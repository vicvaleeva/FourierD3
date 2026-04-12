import numpy as np
import torch
import torch.nn.functional as F

from ase.calculators.calculator import Calculator, all_changes
from matscipy.neighbours import neighbour_list

from fastd3 import FastD3


class FastD3ASECalculator(Calculator):
    '''
    ASE wrapper matching the FastD3 usage pattern in your example.

    - neighbour_list from matscipy
    - explicit strain tensor for stress
    - forces from autograd
    - Hartree → eV conversion
    '''

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        r_cut=6.0,
        method="spme",
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose = False,
        params = None,
        c6tol: float = 1,
        xcfunc: str = 'pbe',
        cnfunc='smooth_cut',
        k_cutoff: float = 10.0,
        mesh_spacing: float = 1.2, # for pme
        interpolation_nodes: int = 5, # for pme
        dtype = torch.float32,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.r_cut = float(r_cut)
        self.method = method
        self.verbose = verbose
        self.params = params

        self.angstrom_to_bohr = (1 / 0.52917726)
        self.HARTREE_TO_EV = 27.21138505
        self.k_cutoff = k_cutoff
        self.xcfunc = xcfunc
        self.cnfunc = cnfunc
        self.c6tol = c6tol
        self.method = method

        # not useful for method = 'ewald'
        self.mesh_spacing = mesh_spacing
        self.interpolation_nodes = interpolation_nodes
        self.dtype= dtype
        # placeholder
        self._model = None

    def _update_cell(self, cell):
        self._model._update_cell(cell=cell)
            

    def _build_model(self, atoms):
        self._model = FastD3(
            species=atoms.numbers,
            cell=atoms.cell.array,
            pbc=torch.tensor(atoms.pbc, device=self.device),
            mesh_spacing=self.mesh_spacing,
            params=self.params,
            c6tol=self.c6tol,
            xcfunc=self.xcfunc,
            cnfunc=self.cnfunc,
            device = self.device,
            method=self.method,
            interpolation_nodes=self.interpolation_nodes,
            k_cutoff = self.k_cutoff,
            verbose=self.verbose,
            r_cut=self.r_cut,
            dtype=self.dtype
        )
        
        

    # ideally this reuse nlist from the MLIP but for now let's keep it this for benchmarking
    def _build_graph(self, atoms, rcut = None):
        if rcut is None:
            rcut = self.r_cut
            
        sender, receiver, unit_shifts = neighbour_list(
            quantities="ijS",
            pbc=atoms.pbc,
            cell=atoms.cell,
            positions=atoms.positions,
            cutoff=rcut,
        )

        edge_index = torch.tensor(
            np.stack((sender, receiver)),
            dtype=torch.long,
            device=self.device,
        )

        unit_shifts = torch.tensor(
            unit_shifts,
            dtype=self.dtype,
            device=self.device,
        )

        return edge_index, unit_shifts

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        cell = torch.tensor(atoms.cell.array, dtype=self.dtype, device=self.device)

        strain = torch.zeros(3, 3, dtype=self.dtype, device=self.device)
        strain.requires_grad_(True)

        strained_cell = cell + torch.einsum("ab,Ab->Aa", strain, cell)
        
        self._update_cell(strained_cell)

        positions = torch.tensor(
            atoms.positions,
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        )
        
        if self.cnfunc == 'smooth_cut':
            strained_pos = positions + torch.einsum("ab,ib->ia", strain, positions)
            edge_index, unit_shifts = self._build_graph(atoms)
            strained_shifts = torch.matmul(unit_shifts, strained_cell)


        # compute energy
        if self.cnfunc == 'smooth_cut':
            energy = self._model(
                strained_pos,
                edge_index,
                strained_shifts,
            )
            
        if self.cnfunc == 'd4':
            energy = self._model(strained_pos)
        # Hartree → eV
        energy_ev = energy * self.HARTREE_TO_EV

        # -------------------------
        # backward and compute forces and stress
        # -------------------------
        energy_ev.backward()

        forces = -positions.grad
        stress = (
            strain.grad
            / self._model.volume * (self.angstrom_to_bohr ** 3)
        )

        # -------------------------
        # store results
        # -------------------------
        self.results["energy"] = energy_ev.detach().cpu().item()
        self.results["forces"] = forces.detach().cpu().numpy()
        self.results["stress"] = stress.detach().cpu().numpy()