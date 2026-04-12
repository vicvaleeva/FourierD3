from typing import Optional, List

import torch
from numpy import unique
from torchpme.lib.kvectors import get_ns_mesh, generate_kvectors_for_ewald

from fastd3.utils import decomp, load_rcov, load_cnref, safe_det_3x3
from fastd3.pair_pot import D3Potential, CNPotential
from fastd3.kspace_filter_d3 import KSpaceFilterD3, KSpaceFilterCN
from fastd3.mesh_interpolator_d3 import MeshInterpolatorD3

class FastD3(torch.nn.Module):
    '''
    species: atoms.numbers List for the configuration
    cell: torch tensor of cell
    pbc: atoms.pbc
    mesh_spacing: parameter controlling mesh size for PME, the smaller the better
    c6tol: parameter controlling accuracy of C6ref eigendecomposition, the smaller the better
    xcfunc: underlying xc functional
    cnfunc: function for CN calculation, options: 'smooth_cut', 'd4'
    device: torch device
    method: either ewald or pme or spme
    k_cutoff: parameter controlling cutoff in the reciprocal space for Ewald, the bigger the better
    interpolation_nodes: number of interpolation nodes used for PPME
    verbose: print stuff or not 
    '''
    
    def __init__(
        self,
        species: List,
        cell,
        pbc: Optional[torch.tensor] = None,
        mesh_spacing: float = 1.2,
        c6tol: float = 1,
        xcfunc: str = 'pbe',
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype = torch.float32, 
        method: str = 'spme',
        params = None,
        cnfunc = 'smooth_cut',
        interpolation_nodes: int = 5,
        k_cutoff: float = 10.0,
        r_cut: float = 6.0,
        cncorr: Optional[torch.tensor] = None,
        verbose = True
    ) -> None:
        super().__init__()
        
        self.device = device
        self.dtype = dtype
        self.method = method
        self.mesh_spacing = mesh_spacing
        self.interpolation_nodes = interpolation_nodes
        self.k_cutoff = k_cutoff
        
        if cncorr is not None:
            self.cncorr = cncorr.to(device)
        else:
            self.cncorr = None

        if pbc is not None:
            assert pbc.all(), "particle-mesh only supports 3d pbc"
        
        if verbose:
            print("Assuming 3D PBC are satisfied")
        
        # Convert species once
        species_unique = unique(species)
        species_map = {species_unique[i]: i for i in range(len(species_unique))}
        converted_species = torch.tensor(
            [species_map[s] for s in species], 
            dtype=torch.long, 
            device=device
        )
        
        # Register as buffer so it moves with model
        self.register_buffer('species', converted_species)
        self.n_species = len(species_unique)
        
        self.xcfunc = xcfunc
        self.cnfunc = cnfunc
        
        # Pre-compute and cache conversion factors
        angstrom_to_bohr = (1 / 0.52917726)
        self.r_cut = r_cut * angstrom_to_bohr
        self.register_buffer(
            'angstrom_to_bohr', 
            torch.tensor(angstrom_to_bohr, dtype=dtype, device=device)
        )
        
        cell = torch.tensor(cell, device=self.device, dtype=dtype)
        
        cell_bohr = cell * self.angstrom_to_bohr
        self.register_buffer('cell', cell_bohr)
        
        volume = torch.abs(safe_det_3x3(cell_bohr))
        self.register_buffer('volume', volume)
        
        # Load and cache eigendecomposition
        eigs, eigvecs = decomp(species_unique, c6tol, verbose, dtype=self.dtype)
        eigs = eigs.to(device=device, dtype=dtype)
        eigvecs = eigvecs.to(device=device, dtype=dtype)
        self.register_buffer('eigs', eigs)
        self.register_buffer('eigvecs', eigvecs)
        self.n_rank = eigvecs.shape[1]
        
        # Pre-reshape eigenvectors for faster access
        v_q_reshaped = eigvecs.view(self.n_species, 7, self.n_rank)
        self.register_buffer('v_q_reshaped', v_q_reshaped)
        
        # Load reference data
        rcov = load_rcov()[species_unique].to(device=device, dtype=dtype)
        cnref = load_cnref()[species_unique, :].to(device=device, dtype=dtype)
        self.register_buffer('rcov', rcov)
        self.register_buffer('cnref', cnref)
        
        # Pre-compute constants
        self.register_buffer('factor_cn', torch.tensor(4.0/3.0, dtype=dtype, device=device))
        self.register_buffer('logit_scale', torch.tensor(-4.0, dtype=dtype, device=device))
        
        # D3 parameters
        if params is not None:
            params = torch.tensor(params, device=device, dtype=dtype)
        elif xcfunc == 'pbe':
            params = torch.tensor([1.0, 0.7875, 0.4289, 4.4407], device=device, dtype=dtype)
        else:
            raise NotImplementedError
        self.potential = D3Potential(species_unique, params, device, method, order=interpolation_nodes)
        if self.cnfunc == 'd4':
            self.cn_potential = CNPotential(species_unique, device, method, order=interpolation_nodes, dtype=dtype)
            self.register_buffer('cn_selfcont', self.cn_potential.selfcont)
        
        # Cache self-interaction terms
        self.register_buffer('selfcont', self.potential.selfcont)
        
        if method == 'pme' or method == 'spme':
            self.ns_mesh = get_ns_mesh(cell_bohr, mesh_spacing * angstrom_to_bohr)
            if verbose:
                print('Using mesh size', self.ns_mesh)
                
            self.kspace_filter = KSpaceFilterD3(
                cell=cell_bohr,
                ns_mesh=self.ns_mesh,
                kernel=self.potential,
                fft_norm="backward",
                ifft_norm="forward"
            )
            
            if self.cn_func == 'd4':
                self.kspace_filter_cn = KSpaceFilterCN(
                    cell=cell_bohr,
                    ns_mesh=self.ns_mesh,
                    kernel=self.cn_potential,
                    fft_norm="backward",
                    ifft_norm="forward"
                )
            
        if method == 'pme':
            self.mesh_interpolator = MeshInterpolatorD3(
                cell=cell_bohr,
                ns_mesh=self.ns_mesh,
                interpolation_nodes=interpolation_nodes,
                method="Lagrange"
            )
            
        if method == 'spme':
            self.mesh_interpolator = MeshInterpolatorD3(
                cell=cell_bohr,
                ns_mesh=self.ns_mesh,
                interpolation_nodes=interpolation_nodes,
                method="P3M"
            )
            
        elif method == 'ewald':
            basis_norms = torch.linalg.norm(cell_bohr, dim=1)
            ns_float = k_cutoff * basis_norms / (2 * torch.pi)
            self.ns = torch.ceil(ns_float).long()
            
            kvectors = generate_kvectors_for_ewald(ns=self.ns, cell=cell_bohr).to(device)
            knorm = torch.linalg.norm(kvectors, dim=1)
            G = self.potential.lr_from_k_sq(knorm)
            if self.cnfunc =='d4':
                G_cn = self.cn_potential.lr_from_k_sq(knorm)
                self.register_buffer('G_cn', G_cn)
            
            self.register_buffer('kvectors', kvectors)
            self.register_buffer('knorm', knorm)
            self.register_buffer('G', G)
    
    def _update_cell(self, cell):
        self.cell = cell * self.angstrom_to_bohr
        self.volume = torch.abs(safe_det_3x3(self.cell))
        if self.method == 'pme' or self.method == 'spme':
            self.mesh_interpolator.update(self.cell)
            self.kspace_filter.update(self.cell)
            if self.cnfunc == 'd4':
                self.kspace_filter_cn.update(self.cell)
                
            
        if self.method == 'ewald':
            self.kvectors = generate_kvectors_for_ewald(ns=self.ns, cell=self.cell)
            self.knorm = torch.linalg.norm(self.kvectors, dim=1)
            self.G = self.potential.lr_from_k_sq(self.knorm)
            if self.cnfunc == 'd4':
                self.G_cn = self.cn_potential.lr_from_k_sq(self.knorm)
            
    def _update_cncorr(self, cncorr):
        if cncorr is not None:
            self.cncorr = cncorr.to(self.device)
        else:
            self.cncorr = None

    @torch.jit.export
    def compute_cn_old(self, positions: torch.Tensor, edge_index: torch.Tensor, 
                   shifts: torch.Tensor, recalc=False) -> torch.Tensor:
        """Compute coordination numbers with fused operations."""
        positions = positions.to(dtype=self.dtype)
        n_atoms = positions.size(0)
        source, target = edge_index
        
        # distance calculation
        vec_ij = positions[target] - positions[source] + shifts
        r_ab = torch.linalg.norm(vec_ij, dim=-1)
        
        # Single mask operation
        mask = r_ab > 1e-6
        source_m = source[mask]
        target_m = target[mask]
        r_ab_m = r_ab[mask]
        
        # covalent radius lookup
        r_cov_sum = self.rcov[self.species[source_m]] + self.rcov[self.species[target_m]]
        
        # sigmoid calculations
        inv_r = 1.0 / r_ab_m
        
        ratio = self.factor_cn * r_cov_sum
        arg_r = -16.0 * (ratio * inv_r - 1.0)
        
        # Combined exponential operations
        exp_r = torch.exp(arg_r)
        
        edge_contributions = 1.0 / (1.0 + exp_r)
        
        # Scatter add
        cn = torch.zeros(n_atoms, device=self.device, dtype=self.dtype)
        cn.index_add_(0, source_m, edge_contributions)
        
        if self.cncorr is not None and not recalc:
            cn = self.cncorr[self.species, 1] * cn + self.cncorr[self.species, 0]
        
        return cn
    
    @torch.jit.export
    def compute_cn_smooth(self, positions: torch.Tensor, edge_index: torch.Tensor, 
                shifts: torch.Tensor, recalc: bool = False) -> torch.Tensor:
        positions = positions.to(dtype=self.dtype)
        n_atoms = positions.size(0)
        source, target = edge_index
        
        vec_ij = positions[target] - positions[source] + shifts
        r_ab = torch.linalg.norm(vec_ij, dim=-1)
        
        r_ab_safe = torch.clamp(r_ab, min=1e-6)
        
        r_cov_sum = self.rcov[self.species[source]] + self.rcov[self.species[target]]
        r1 = 0.5 * r_cov_sum + 0.5 * self.r_cut
        
        r_ab_coeff = torch.where(r_ab_safe > r1, r_ab_safe, r1)
        denom = (self.r_cut - r_ab_coeff).pow(2) + 1e-6
        k_cn = 16.0 + (r_ab_coeff - r1)**2 / denom
        
        inv_r = 1.0 / r_ab_safe
        ratio = self.factor_cn * r_cov_sum
        
        arg_r = -k_cn * (ratio * inv_r - 1.0)
        edge_contributions = torch.sigmoid(-arg_r)
        
        edge_contributions = torch.where(
            r_ab > 1e-6, 
            edge_contributions, 
            torch.tensor(0.0, dtype=self.dtype, device=self.device)
        )
        
        cn = torch.zeros(n_atoms, device=self.device, dtype=self.dtype)
        cn.index_add_(0, source, edge_contributions)
        
        if self.cncorr is not None and not recalc:
            cn = self.cncorr[self.species, 1] * cn + self.cncorr[self.species, 0]
        
        return cn
    
    @torch.jit.export
    def compute_cn_d4(self, positions: torch.Tensor, ex: torch.Tensor = None, nk = None) -> torch.Tensor:
        n_atoms = positions.shape[0]
        # One-hot charges: atom i contributes 1.0 in its species slot.
        cn_charges = torch.zeros(n_atoms, self.n_species, 1, device=self.device, dtype=self.dtype)
        cn_charges[torch.arange(n_atoms, device=self.device), self.species, 0] = 1.0
        if self.method in ('pme', 'spme'):
            rho_mesh = self.mesh_interpolator.points_to_mesh(cn_charges, dtype=self.dtype).squeeze(1)
            # (n_species, nx, ny, nz)

            phi_mesh = self.kspace_filter_cn.forward(rho_mesh).unsqueeze(1)
            # (n_species, 1, nx, ny, nz)

            phi_atoms = self.mesh_interpolator.mesh_to_points(phi_mesh, dtype=self.dtype).squeeze(-1)
            # (n_atoms, n_species)

        else:  # ewald
            # Structure factor per species: (2, nk, n_species)
            S = torch.mm(ex.reshape(2 * nk, n_atoms), self.one_hot_species).view(2, nk, self.n_species)
            # Apply Green's function and back-interpolate
            F = torch.einsum('tkj,kij->tki', S, self.G_cn.permute(2, 0, 1))  # (2, nk, n_species)
            phi_atoms = torch.einsum('tka,tki->ai', ex, F)  # (n_atoms, n_species)
            
        batch = torch.arange(n_atoms, device=self.device)
        cn_raw = phi_atoms[batch, self.species]
        cn = cn_raw  / self.volume - self.cn_selfcont[self.species]
        return cn
    
    @torch.jit.export
    def compute_c6_weights(self, cn: torch.Tensor) -> torch.Tensor:
        """Compute C6 weights from coordination numbers."""
        atom_refs = self.cnref[self.species]
        diff = cn.unsqueeze(1) - atom_refs
        logits = self.logit_scale * torch.square(diff)
        
        # Mask invalid references
        mask = atom_refs == -1
        logits = logits.masked_fill(mask, float('-inf'))
        
        return torch.softmax(logits, dim=1)
        
    def forward(self, positions: torch.Tensor,
                edge_index: torch.Tensor,
                shifts: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optimized operations.
        
        Args:
            positions: Atomic positions [N, 3]
            edge_index: Edge indices [2, E]
            shifts: Periodic shifts [E, 3]
            
        Returns:
            D3 dispersion energy
        """
        
        positions = positions * self.angstrom_to_bohr
        if self.cncfunc == 'smooth_cut':
            shifts = shifts * self.angstrom_to_bohr
        
        n_atoms = positions.size(0)
        
        # Compute coordination numbers
        if self.cnfunc == 'smooth_cut':
            cn = self.compute_cn_smooth(positions, edge_index, shifts)
        if self.cnfunc == 'd4' and self.method in ('pme', 'spme'):
            cn = self.compute_cn_d4(positions)
        if self.cnfunc == 'd4' and self.method == 'ewald':
            trig_args = torch.matmul(self.kvectors, positions.T)  # (nk, n_atoms)
            nk = trig_args.shape[0]
            ex = torch.stack([torch.cos(trig_args), torch.sin(trig_args)], dim=0)
            cn = self.compute_cn_d4(positions, ex, nk)
        
        # Compute weights
        weights = self.compute_c6_weights(cn)
        
        # Compute C6 coefficients (optimized einsum)
        atom_v_qs = self.v_q_reshaped[self.species]
        c6 = torch.einsum('nk,nkr->nr', weights, atom_v_qs)
        
        # One-hot encoding
        batch_idx = torch.arange(n_atoms, device=self.device)

        onehot = torch.zeros(n_atoms, self.n_species, self.n_rank, 
                            device=c6.device, 
                            dtype=self.dtype).index_put(
                                (batch_idx, self.species), 
                                c6
                            )
        
        
        # Self-interaction energy (fused operations)
        c6_sq = torch.square(c6)
        selfcont_atoms = self.selfcont[self.species]
        tmp = torch.sum(c6_sq * selfcont_atoms, dim=0)
        vself = torch.dot(self.eigs, tmp)
        
        if self.method == 'pme' or self.method == 'spme':
            self.mesh_interpolator.compute_weights(positions)
            rho_mesh = self.mesh_interpolator.points_to_mesh(onehot, dtype=self.dtype)
            filtered_hat = self.kspace_filter.forward(rho_mesh)
            energy = torch.dot(self.eigs, filtered_hat)
            
        elif self.method == 'ewald':
            if self.cnfunc == 'smooth_cut':
                trig_args = torch.matmul(self.kvectors, positions.T)
                
                c = torch.cos(trig_args)
                s = torch.sin(trig_args)
                ex = torch.stack([c, s], dim=0)
            sc = torch.einsum('izr,pki->rzpk', onehot, ex)
            sqrt_eigs = torch.sqrt(torch.abs(self.eigs))
            sc_weighted = sc * sqrt_eigs.view(-1, 1, 1, 1)
            convolved = torch.einsum('ijk,ripk,rjpk->jk', self.G, sc_weighted, sc_weighted)
            energy = torch.sum(convolved)
        
        energy = energy / (-2 * self.volume) + vself / 2
        
        return energy