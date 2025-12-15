from typing import Optional, List
from utils import decomp, load_rcov, load_cnref
from numpy import unique
from torchpme.lib.kvectors import get_ns_mesh, generate_kvectors_for_ewald
import matplotlib.pyplot as plt
import time
from pair_pot import D3Potential
from kspace_filter_d3 import KSpaceFilterD3
from mesh_interpolator_d3 import MeshInterpolatorD3

import torch

class FastD3(torch.nn.Module):
    '''
    species: atoms.numbers List for the configuration
    cell: atoms.cell
    pbc: atoms.pbc
    mesh_spacing: parameter controlling mesh size for PME, the smaller the better
    c6tol: parameter controlling accuracy of C6ref eigendecomposition, the smaller the better
    xcfunc: underlying xc functional
    device: torch device
    method: either ewald or pme
    k_cutoff: parameter controlling cutoff in the reciprocal space for Ewald, the bigger the better
    interpolation_nodes: number of interpolation nodes used for PPME
    verbose: print stuff or not 
    '''
    
    def __init__(
        self,
        species: List,
        cell: torch.tensor,
        pbc: Optional[torch.tensor] = None,
        mesh_spacing: float = 1.2,
        c6tol: float = 1,
        xcfunc: str = 'pbe',
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        method: str = 'pme',
        interpolation_nodes: int = 4,
        k_cutoff: float = 1.0,
        verbose = True
    ) -> None:
        super().__init__()
        
        self.device = device
        self.method = method
        
        if pbc is not None:
            assert pbc.all(), "ewald summation only supports 3d pbc"
        
        if verbose:
            print("Assuming 3D PBC are satisfied")
        
        # adapt atoms.numbers for inner use
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
        
        # Pre-compute and cache conversion factors
        angstrom_to_bohr = 1.8897259492972167
        self.register_buffer(
            'angstrom_to_bohr', 
            torch.tensor(angstrom_to_bohr, dtype=torch.float64, device=device)
        )
        
        cell_bohr = torch.from_numpy(cell.array).to(device=device, dtype=torch.float64) * self.angstrom_to_bohr
        self.register_buffer('cell', cell_bohr)
        
        volume = torch.abs(torch.det(cell_bohr))
        self.register_buffer('volume', volume)
        
        # Load and cache eigendecomposition
        eigs, eigvecs = decomp(species_unique, c6tol, verbose)
        eigs = eigs.to(device=device, dtype=torch.float64)
        eigvecs = eigvecs.to(device=device, dtype=torch.float64)
        self.register_buffer('eigs', eigs)
        self.register_buffer('eigvecs', eigvecs)
        self.n_rank = eigvecs.shape[1]
        
        # Pre-reshape eigenvectors for faster access
        _ = eigvecs.shape[0]
        v_q_reshaped = eigvecs.view(self.n_species, 7, self.n_rank)
        self.register_buffer('v_q_reshaped', v_q_reshaped)
        
        # Load reference data
        rcov = load_rcov()[species_unique].to(device=device, dtype=torch.float64)
        cnref = load_cnref()[species_unique, :].to(device=device, dtype=torch.float64)
        self.register_buffer('rcov', rcov)
        self.register_buffer('cnref', cnref)
        
        # Pre-compute constants
        self.register_buffer('k_cn', torch.tensor(16.0, dtype=torch.float64, device=device))
        self.register_buffer('factor_cn', torch.tensor(4.0/3.0, dtype=torch.float64, device=device))
        self.register_buffer('logit_scale', torch.tensor(-4.0, dtype=torch.float64, device=device))
        
        # D3 XC functional parameters
        params = torch.tensor([1.0, 0.7875, 0.4289, 4.4407], device=device, dtype=torch.float64)
        self.potential = D3Potential(species_unique, params, device, method)
        
        # Cache self-interaction terms
        self.register_buffer('selfcont', self.potential.selfcont)
        
        if method == 'pme':
            ns_mesh = get_ns_mesh(cell_bohr, mesh_spacing * angstrom_to_bohr)
            if verbose:
                print('Using mesh size', ns_mesh)
                
            # Define an interpolator
            
            self.mesh_interpolator = MeshInterpolatorD3(
                cell=cell_bohr,
                ns_mesh=ns_mesh,
                interpolation_nodes=interpolation_nodes,
                method="Lagrange"
            )
            
            # Define a reciprocal space filter
            
            self.kspace_filter = KSpaceFilterD3(
                cell=cell_bohr,
                ns_mesh=ns_mesh,
                kernel=self.potential,
                fft_norm="backward",
                ifft_norm="forward"
            )
            
        elif method == 'ewald':
            # get k-vectors and kernel for ewald 
            
            basis_norms = torch.linalg.norm(cell_bohr, dim=1)
            ns_float = k_cutoff * basis_norms / (2 * torch.pi)
            ns = torch.ceil(ns_float).long()
            
            kvectors = generate_kvectors_for_ewald(ns=ns, cell=cell_bohr).to(device)
            knorm = torch.linalg.norm(kvectors, dim=1)
            G = self.potential.lr_from_k_sq(knorm)
            
            self.register_buffer('kvectors', kvectors)
            self.register_buffer('knorm', knorm)
            self.register_buffer('G', G)
    
    @torch.jit.export
    def compute_cn(self, positions: torch.Tensor, edge_index: torch.Tensor, 
                   shifts: torch.Tensor, r_cut: torch.Tensor) -> torch.Tensor:
        """Compute coordination numbers with fused operations."""
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
        
        '''
        Some terms here are commented as smoothing of the CNs can significantly disturb
        calculations at 5-6 Å cutoffs
        '''
        
        # sigmoid calculations
        inv_r = 1.0 / r_ab_m
        #inv_r_cut = 1.0 / r_cut
        
        ratio = self.factor_cn * r_cov_sum
        arg_r = -self.k_cn * (ratio * inv_r - 1.0)
        #arg_cut = -self.k_cn * (ratio * inv_r_cut - 1.0)
        
        # Combined exponential operations
        exp_r = torch.exp(arg_r)
        #exp_cut = torch.exp(arg_cut)
        
        term1 = 1.0 / (1.0 + exp_r)
        #term2 = 1.0 / (1.0 + exp_cut)
        
        # derivative term
        #sigmoid_deriv = exp_cut / torch.square(1.0 + exp_cut)
        #dist_diff = r_ab_m - r_cut
        #prefactor = (64.0 * r_cov_sum) / (3.0 * (r_cut ** 2))
        #term3 = dist_diff * prefactor * sigmoid_deriv
        
        #edge_contributions = term1 - term2 + term3
        edge_contributions = term1
        
        # Scatter add
        cn = torch.zeros(n_atoms, device=self.device, dtype=positions.dtype)
        cn.index_add_(0, source_m, edge_contributions)
        
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
                shifts: torch.Tensor, 
                r_cut: torch.Tensor) -> torch.Tensor:
        """
        
        Args:
            positions: Atomic positions [N, 3]
            edge_index: Edge indices [2, E]
            shifts: Periodic shifts [E, 3]
            r_cut: Cutoff radius
            
        Returns:
            D3 dispersion energy
        """
        
        # unit conversion to Bohr to match tabulated values
        
        positions = positions * self.angstrom_to_bohr
        r_cut = r_cut * self.angstrom_to_bohr
        shifts = shifts * self.angstrom_to_bohr
        
        n_atoms = positions.size(0)
        
        # Compute coordination numbers
        cn = self.compute_cn(positions, edge_index, shifts, r_cut)
        
        # Compute weights
        weights = self.compute_c6_weights(cn)
        
        # Compute C6 coefficients (optimized einsum)
        atom_v_qs = self.v_q_reshaped[self.species]
        c6 = torch.einsum('nk,nkr->nr', weights, atom_v_qs)
        
        # One-hot encoding
        onehot = torch.zeros(n_atoms, self.n_species, self.n_rank, 
                            device=c6.device, dtype=c6.dtype)
        onehot[torch.arange(n_atoms, device=self.device), self.species] = c6
        
        
        # Self-interaction energy (fused operations)
        c6_sq = torch.square(c6)
        selfcont_atoms = self.selfcont[self.species]
        tmp = torch.sum(c6_sq * selfcont_atoms, dim=0)
        vself = torch.dot(self.eigs, tmp)
        
        if self.method == 'pme':
            # compute interpolation weights
            self.mesh_interpolator.compute_weights(positions)
            # interpolate C6 onto meshes
            rho_mesh = self.mesh_interpolator.points_to_mesh(onehot)
            # convolve with the kernel
            filtered_hat = self.kspace_filter.forward(rho_mesh)
            energy = torch.dot(self.eigs, filtered_hat.real)
            
        elif self.method == 'ewald':
            # get k\cdot r
            
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