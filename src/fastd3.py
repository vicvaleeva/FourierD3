from typing import Optional, List
from utils import decomp, load_rcov, load_cnref
from numpy import unique
from torchpme.lib.kvectors import get_ns_mesh

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
        pbc: Optional[torch.tensor],
        mesh_spacing: float = 1.2,
        c6tol: float = 1,
        xcfunc: str = 'pbe',
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        method: str = 'pme',
        interpolation_nodes: int = 4
    ) -> None:
        super().__init__()
        
        self.device = device
        
        if pbc is not None:
            assert pbc.all(), "particle-mesh only supports 3d pbc, if you have 2d pbc, please make sure there's plenty of empty space in the third direction"

        print("Assuming 3D PBC are satisfied")
        species_unique = unique(species)
        tmp = {species_unique[i] : i for i in range(len(species_unique))}
        converted_species = [tmp[species[i]] for i in range(len(species))]
        self.species = torch.tensor(converted_species, dtype=torch.long, device=self.device)
        
        self.xcfunc = xcfunc
        
        self.angstrom_to_bohr = torch.tensor(1.8897161646321, device=self.device)
        self.hartree_to_kcalmol = torch.tensor(627.5094740631, device=self.device)
        self.cell = torch.from_numpy(cell.array) * self.angstrom_to_bohr
        self.cell.to(device)
        
        self.volume = torch.abs(torch.det(self.cell)).to(self.device)
        
        self.eigs, self.eigvecs = decomp(species_unique, c6tol)
        self.eigs.to(device)
        self.eigvecs.to(device)
        
        self.rcov = load_rcov()[species_unique].to(self.device)
        self.cnref = load_cnref()[species_unique, :].to(self.device)
        
        
        # implement automatic choice from xc functional !
        # these are just for pbe
        params = torch.tensor([1.0, 0.7875, 0.4289, 4.4407], device=device)
        
        potential = D3Potential(species_unique, params, device)
        
        ns_mesh = get_ns_mesh(self.cell, mesh_spacing * self.angstrom_to_bohr)
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
            kernel = potential,
            fft_norm="backward",
            ifft_norm="forward"
        )
        
        
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
        weights = torch.exp(-4.0 * torch.square(diff))
        
        mask = (atom_refs == -1)
        weights.masked_fill_(mask, 0.0)
        
        # calculate C6
        n_rank = self.eigvecs.shape[1]
        n_total_rows = self.eigsvecs.shape[0] 
        n_species = n_total_rows // 7
        
        v_q_reshaped = self.eigvecs.view(n_species, 7, n_rank)
        atom_v_qs = v_q_reshaped[self.species]
        numerator = torch.einsum('nk, nkr -> nr', weights, atom_v_qs)
        denominator = weights.sum(dim=1, keepdim=True)
        
        c6 = numerator / denominator
        
        # one-hot encoding for species (since potential is species-dependent)
        
        onehot = torch.zeros(n_atoms, n_species, n_rank, device=c6.device, dtype=c6.dtype)
        onehot[torch.arange(n_atoms), self.species, :] = c6
        
        # interpolate particle weights onto a mesh
        
        self.mesh_interpolator.compute_weights(positions)
        rho_mesh = self.mesh_interpolator.points_to_mesh(c6)
        
        # convolve with potential
        energy = self.kspace_filter.forward(rho_mesh, self.eigs)
        
        energy /= (-self.volume)
        vself = torch.dot(self.eigs, torch.sum(torch.square(c6)*self.potential.selfcont[self.species], dim=-1))
        energy += vself
        energy /= 2
        
        return energy
        
        