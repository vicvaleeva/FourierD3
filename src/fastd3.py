from typing import Optional, List
from utils import decomp, load_rcov, load_cnref
from numpy import unique
from torchpme.lib.kvectors import get_ns_mesh, generate_kvectors_for_ewald
import matplotlib.pyplot as plt
from pair_pot import D3Potential
from kspace_filter_d3 import KSpaceFilterD3
from mesh_interpolator_d3 import MeshInterpolatorD3

import torch

class FastD3(torch.nn.Module):
    """
    Fast D3 calculator in the torch interface.
    Uses particle-mesh methods (PME and P3M) to perform fast summation
    of the ~inherently~ long-ranged damped D3 correction potential.
    The C^6AB are untangled using eigendecomposition and the method
    re-uses the neighborlist calculated by the underlying ML potential.
    
    :param elements: list containing atom types for eigendecomposition,
        must contain at least all the atom types present in the cell
    :param cell: tensor containing vectors defining the periodic cell dimensions
    :param pbc: 3xBool tensor to verify pbc are activated in all directions
    :param mesh_spacing: parameter controlling mesh spacing (in Angstrom), 
        biggest influence on accuracy
    :param c6tol: maximum relative error for estimation of C6ref (in %), controls
        the rank of eigendecomposition approximation
    :param xcfunc: string specifying the xc functional used to train
        the ML potential, needed for D3 parameters
        
    """
    
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
        #
        self.mesh_spacing = mesh_spacing
        self.interpolation_nodes = interpolation_nodes
        self.k_cutoff = k_cutoff

        if pbc is not None:
            assert pbc.all(), "particle-mesh only supports 3d pbc, if you have 2d pbc, please make sure there's plenty of empty space in the third direction"
        if verbose:
            print("Assuming 3D PBC are satisfied")
        species_unique = unique(species)
        tmp = {species_unique[i] : i for i in range(len(species_unique))}
        converted_species = [tmp[species[i]] for i in range(len(species))]
        self.species = torch.tensor(converted_species, dtype=torch.long, device=self.device)
        
        self.xcfunc = xcfunc
        
        self.angstrom_to_bohr = torch.tensor(1.8897259492972167, device=self.device)
        self.cell = torch.from_numpy(cell.array) * self.angstrom_to_bohr
        self.cell.to(device)
        
        self.volume = torch.abs(torch.det(self.cell)).to(self.device)
        
        self.eigs, self.eigvecs = decomp(species_unique, c6tol, verbose)
        self.eigs.to(device)
        self.eigvecs.to(device)
        
        self.rcov = load_rcov()[species_unique].to(device=self.device, dtype=torch.float64)
        self.cnref = load_cnref()[species_unique, :].to(device=self.device, dtype=torch.float64)
        
        
        # implement automatic choice from xc functional !
        # these are just for pbe
        params = torch.tensor([1.0, 0.7875, 0.4289, 4.4407], device=device, dtype=torch.float64)
        
        self.potential = D3Potential(species_unique, params, device, method)
        
        if method == 'pme':
        
            ns_mesh = get_ns_mesh(self.cell, mesh_spacing * self.angstrom_to_bohr)
            if verbose:
                print('Using mesh size', ns_mesh.numpy())
            
            self.mesh_interpolator = MeshInterpolatorD3(
                cell=self.cell,
                ns_mesh=ns_mesh,
                interpolation_nodes=interpolation_nodes,
                method="Lagrange"
            )
            
            self.kspace_filter = KSpaceFilterD3(
                cell=self.cell,
                ns_mesh=ns_mesh,
                kernel = self.potential,
                fft_norm="backward",
                ifft_norm="forward"
            )
            
        if method == 'ewald':
            basis_norms = torch.linalg.norm(self.cell, dim=1)
            ns_float = k_cutoff * basis_norms / 2 / torch.pi
            ns = torch.ceil(ns_float).long()
            
            self.kvectors = generate_kvectors_for_ewald(ns=ns, cell=self.cell).to(self.device)
            self.knorm = torch.linalg.norm(self.kvectors, dim=1)
            self.G = self.potential.lr_from_k_sq(self.knorm)
        
    def _update_cell(self, cell):
        self.cell = cell
        if self.method == 'pme':
        
            ns_mesh = get_ns_mesh(self.cell, self.mesh_spacing * self.angstrom_to_bohr)
            if self.verbose:
                print('Using mesh size', ns_mesh.numpy())
            
            self.mesh_interpolator = MeshInterpolatorD3(
                cell=self.cell,
                ns_mesh=ns_mesh,
                interpolation_nodes=self.interpolation_nodes,
                method="Lagrange"
            )
            
            self.kspace_filter = KSpaceFilterD3(
                cell=self.cell,
                ns_mesh=ns_mesh,
                kernel = self.potential,
                fft_norm="backward",
                ifft_norm="forward"
            )
            
        if self.method == 'ewald':
            basis_norms = torch.linalg.norm(self.cell, dim=1)
            ns_float = self.k_cutoff * basis_norms / 2 / torch.pi
            ns = torch.ceil(ns_float).long()
            
            self.kvectors = generate_kvectors_for_ewald(ns=ns, cell=self.cell).to(self.device)
            self.knorm = torch.linalg.norm(self.kvectors, dim=1)
            self.G = self.potential.lr_from_k_sq(self.knorm)
        

    def forward(self, positions,
                edge_index,
                shifts, r_cut):
        if positions.device != self.device:
            raise ValueError(f"Device mismatch: {positions.device} vs {self.device}")
        if edge_index.device != self.device:
            raise ValueError(f"Device mismatch: {edge_index.device} vs {self.device}")
        if shifts.device != self.device:
            raise ValueError(f"Device mismatch: {shifts.device} vs {self.device}")
        
        positions = self.angstrom_to_bohr * positions
        r_cut = self.angstrom_to_bohr * r_cut
        shifts = self.angstrom_to_bohr * shifts
        
        n_atoms = positions.size(0)
        
        
        # calculate CNs
        
        source, target = edge_index
        vec_ij = positions[target] - positions[source] + shifts
        r_ab = torch.norm(vec_ij, dim=-1)
        mask = (r_ab > 1e-6)
        source = source[mask]
        target = target[mask]
        r_ab = r_ab[mask]
        r_cov_sum = self.rcov[self.species[source]] + self.rcov[self.species[target]]
        
        k = 16.0
        factor = 4.0 / 3.0
        
        arg_r = -k * ((factor * r_cov_sum / r_ab) - 1.0)
        term1 = 1.0 / (1.0 + torch.exp(arg_r))
        
        arg_cut = -k * ((factor * r_cov_sum / r_cut) - 1.0)
        exp_cut = torch.exp(arg_cut)
        term2 = 1.0 / (1.0 + exp_cut)
        
        dist_diff = r_ab - r_cut
        prefactor = (64.0 * r_cov_sum) / (3.0 * (r_cut ** 2))
        sigmoid_deriv = exp_cut / torch.square(1.0 + exp_cut)
        term3 = dist_diff * prefactor * sigmoid_deriv
        
        edge_contributions = term1 - term2 + term3
        cn = torch.zeros(n_atoms, device=self.device, dtype=positions.dtype)
        cn.index_add_(0, source, edge_contributions)
        
        # calculate weights
        
        atom_refs = self.cnref[self.species]
        diff = cn.unsqueeze(1) - atom_refs
        logits = -4.0 * torch.square(diff)
        
        mask = (atom_refs == -1)
        logits =  logits.masked_fill(mask, float('-inf'))
        
        weights = torch.softmax(logits, dim=1)
        
        
        # calculate C6
        n_rank = self.eigvecs.shape[1]
        n_total_rows = self.eigvecs.shape[0] 
        n_species = n_total_rows // 7
        
        v_q_reshaped = self.eigvecs.view(n_species, 7, n_rank)
        atom_v_qs = v_q_reshaped[self.species]
        c6 = torch.einsum('nk, nkr -> nr', weights, atom_v_qs)
        
        
        # one-hot encoding for species (since potential is species-dependent)
        
        onehot = torch.zeros(n_atoms, n_species, n_rank, device=c6.device, dtype=c6.dtype)
        onehot[torch.arange(n_atoms), self.species, :] = c6
        
        tmp = torch.sum(torch.square(c6)*self.potential.selfcont[self.species], dim=0)
        vself = torch.dot(self.eigs, tmp)
        
        if self.method == 'pme':
        
            # interpolate particle weights onto a mesh
            
            self.mesh_interpolator.compute_weights(positions)
            rho_mesh = self.mesh_interpolator.points_to_mesh(onehot)
            
            # convolve with potential
            filtered_hat = self.kspace_filter.forward(rho_mesh)
            
            energy = torch.dot(self.eigs, filtered_hat.real)
            
        elif self.method == 'ewald':
            trig_args = self.kvectors @ (positions.T)
            
            c = torch.cos(trig_args)
            s = torch.sin(trig_args)
            
            ex = torch.stack([c, s], dim=0)
            
            sc = torch.einsum('izr, pki -> rzpk', onehot, ex)

            convolved = torch.einsum('ijk, ripk, rjpk -> rijk', self.G, sc, sc)

            energy = torch.dot(self.eigs, torch.sum(convolved, dim=(1, 2, 3)))
        
        energy /= (-2*self.volume)
        energy += vself/2
        return energy
        
        