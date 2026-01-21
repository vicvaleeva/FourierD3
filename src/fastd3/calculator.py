import numpy as np
import torch

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
        c6tol: float = 1,
        xcfunc: str = 'pbe',
        k_cutoff: float = 10.0,
        mesh_spacing: float = 1.2, # for pme
        interpolation_nodes: int = 4, # for pme
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.r_cut = float(r_cut)
        self.method = method
        self.verbose = verbose

        self.angstrom_to_bohr = (1 / 0.52917726)
        self.HARTREE_TO_EV = 27.21138505
        self.k_cutoff = k_cutoff
        self.xcfunc = xcfunc
        self.c6tol = c6tol
        self.method = method

        # not useful for method = 'ewald'
        self.mesh_spacing = mesh_spacing
        self.interpolation_nodes = interpolation_nodes

        # placeholder
        self._model = None

    def _update_cell(self, cell):
            self._model._update_cell(cell)

    def _build_model(self, atoms):
        self._model = FastD3(
            species=atoms.numbers,
            cell=torch.tensor(atoms.cell.array, device=self.device, dtype=torch.float32),
            pbc=torch.tensor(atoms.pbc, device=self.device),
            # 
            mesh_spacing=self.mesh_spacing,
            c6tol=self.c6tol,
            xcfunc=self.xcfunc,
            device = self.device,
            method=self.method,
            interpolation_nodes=self.interpolation_nodes,
            k_cutoff = self.k_cutoff,
            verbose=self.verbose,
        )

    # ideally this reuse nlist from the MLIP but for now let's keep it this for benchmarking
    def _build_graph(self, atoms):
        sender, receiver, unit_shifts = neighbour_list(
            quantities="ijS",
            pbc=atoms.pbc,
            cell=atoms.cell,
            positions=atoms.positions,
            cutoff=self.r_cut,
        )

        edge_index = torch.tensor(
            np.stack((sender, receiver)),
            dtype=torch.long,
            device=self.device,
        )

        unit_shifts = torch.tensor(
            unit_shifts,
            dtype=torch.float32,
            device=self.device,
        )

        return edge_index, unit_shifts

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        cell = torch.tensor(atoms.cell.array, dtype=torch.float32, device=self.device)
        
        if "cell" in system_changes:
            self._update_cell(cell)

        strain = torch.zeros(3, 3, dtype=torch.float32, device=self.device)
        strain.requires_grad_(True)

        strained_cell = cell + torch.einsum("ab,Ab->Aa", strain, cell)

        # self._build_model(atoms, strained_cell)

        positions = torch.tensor(
            atoms.positions,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )

        strained_pos = positions + torch.einsum("ab,ib->ia", strain, positions)
        edge_index, unit_shifts = self._build_graph(atoms)
        strained_shifts = torch.matmul(unit_shifts, strained_cell)

        r_cut_t = torch.tensor(self.r_cut, dtype=torch.float32, device=self.device)

        # compute energy
        energy = self._model(
            strained_pos,
            edge_index,
            strained_shifts,
            r_cut_t,
        )
        # Hartree → eV
        energy_ev = energy * self.HARTREE_TO_EV

        # -------------------------
        # backward and compute forces and stress
        # -------------------------
        energy_ev.backward()

        forces = -positions.grad
        stress = (
            strain.grad
            / self._model.volume
        )

        # -------------------------
        # store results
        # -------------------------
        self.results["energy"] = energy_ev.detach().cpu().item()
        self.results["forces"] = forces.detach().cpu().numpy()
        self.results["stress"] = stress.detach().cpu().numpy()