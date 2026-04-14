import numpy as np
import torch

from ase.calculators.calculator import Calculator, all_changes
from matscipy.neighbours import neighbour_list

from fastd3 import FastD3


class FastD3ASECalculator(Calculator):
    """Standalone ASE calculator for Fast-D3 dispersion correction.

    Wraps FastD3 in the ASE Calculator interface, providing energy, forces,
    and stress. Intended for benchmarking or for use with non-MACE MLFFs that
    do not have a native integration point.

    For MACE + Fast-D3, prefer MACEFastD3Calculator instead, which reuses
    MACE's neighbour list to avoid redundant computation.

    The stress is computed via automatic differentiation through a strain
    tensor, following the standard approach in ML force field calculators.
    The neighbour list is rebuilt via matscipy at every call.

    Units: ASE uses eV and Å. Fast-D3 computes in Hartree and Bohr internally.
    """

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
        mesh_spacing: float = 1.2,
        interpolation_nodes: int = 5,
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

        # mesh_spacing and interpolation_nodes are only used for method='pme'/'spme'
        self.mesh_spacing = mesh_spacing
        self.interpolation_nodes = interpolation_nodes
        self.dtype = dtype

        # FastD3 model is built lazily on the first calculate() call,
        # once the atomic species and cell are known
        self._model = None

    def _update_cell(self, cell):
        """Forward a new cell to the FastD3 model (called every step for NPT)."""
        self._model._update_cell(cell=cell)

    def _build_model(self, atoms):
        """Instantiate the FastD3 model for the species and cell of `atoms`."""
        self._model = FastD3(
            species=atoms.numbers,
            cell=atoms.cell.array,
            pbc=torch.tensor(atoms.pbc, device=self.device),
            mesh_spacing=self.mesh_spacing,
            params=self.params,
            c6tol=self.c6tol,
            xcfunc=self.xcfunc,
            cnfunc=self.cnfunc,
            device=self.device,
            method=self.method,
            interpolation_nodes=self.interpolation_nodes,
            k_cutoff=self.k_cutoff,
            verbose=self.verbose,
            r_cut=self.r_cut,
            dtype=self.dtype
        )

    def _build_graph(self, atoms, rcut=None):
        """Build a neighbour list for the current atoms object via matscipy.

        Returns edge_index (source/target atom pairs) and unit_shifts (integer
        lattice-vector shifts) as tensors. Only needed for cnfunc='smooth_cut'.
        """
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
        """Compute energy, forces, and stress for the given atoms object.

        Uses a strain tensor with requires_grad=True to obtain the stress via
        backpropagation. The strain deformation maps:
            cell'   = cell + strain @ cell
            positions' = positions + strain @ positions
        so that d(energy)/d(strain)|_{strain=0} is the stress tensor (in eV/Å^3
        after unit conversion from Hartree/Bohr^3).

        For cnfunc='smooth_cut': builds a real-space neighbour list and passes
            it to FastD3.forward together with strained positions and shift vectors.
        For cnfunc='d4': only strained positions are needed; FastD3.forward
            computes CN entirely in k-space.
        """
        super().calculate(atoms, properties, system_changes)

        # Build the model on the first call (lazy initialization)
        if self._model is None:
            self._build_model(atoms)

        cell = torch.tensor(atoms.cell.array, dtype=self.dtype, device=self.device)

        # Strain tensor: zero at evaluation time, but requires_grad for stress
        strain = torch.zeros(3, 3, dtype=self.dtype, device=self.device)
        strain.requires_grad_(True)

        # Apply infinitesimal strain to the cell
        strained_cell = cell + torch.einsum("ab,Ab->Aa", strain, cell)
        self._update_cell(strained_cell)

        positions = torch.tensor(
            atoms.positions,
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        )

        # Apply the same strain to positions and shifts
        strained_pos = positions + torch.einsum("ab,ib->ia", strain, positions)

        if self.cnfunc == 'smooth_cut':
            edge_index, unit_shifts = self._build_graph(atoms)
            strained_shifts = torch.matmul(unit_shifts, strained_cell)
            energy = self._model(strained_pos, edge_index, strained_shifts)
        elif self.cnfunc == 'd4':
            # D4 CN needs only positions; no neighbour list required
            energy = self._model(strained_pos)
        else:
            raise ValueError(f"Unknown cnfunc '{self.cnfunc}'. Expected 'smooth_cut' or 'd4'.")

        # Convert energy from Hartree to eV
        energy_ev = energy * self.HARTREE_TO_EV

        # Backpropagate to get forces and stress in one pass
        energy_ev.backward()

        forces = -positions.grad

        # Stress: (1/Omega) * dE/dstrain, converted from Bohr^3 to Å^3
        stress = (
            strain.grad
            / self._model.volume * (self.angstrom_to_bohr ** 3)
        )

        self.results["energy"] = energy_ev.detach().cpu().item()
        self.results["forces"] = forces.detach().cpu().numpy()
        self.results["stress"] = stress.detach().cpu().numpy()
