import torch
import numpy as np
from ase.calculators.calculator import all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from mace.calculators import MACECalculator
from fourierd3 import FourierD3


class MACEFourierD3Calculator(MACECalculator):
    """Hybrid ASE calculator: MACE short-range energy + Fourier-D3 dispersion.

    MACE provides the short-range energy, forces, and stress. Fourier-D3 adds the
    long-range DFT-D3 dispersion correction on top. When cnfunc='smooth_cut',
    the neighbour list already built by MACE is reused for the CN computation,
    making the marginal cost of Fourier-D3 very small (no extra neighbour list).
    When cnfunc='d4', no neighbour list is needed for Fourier-D3 at all.

    Usage:
        calc = MACEFourierD3Calculator(
            model_paths=["model.pt"],
            d3_kwargs={"xcfunc": "pbe", "method": "spme", "cnfunc": "smooth_cut"},
        )
        atoms.calc = calc

    Args:
        d3_kwargs: keyword arguments forwarded to FourierD3.__init__, e.g.
                   xcfunc, method, cnfunc, mesh_spacing, k_cutoff, c6tol, etc.
                   The r_cut is taken from MACE's r_max if not provided.
        **mace_kwargs: all other keyword arguments are forwarded to MACECalculator.
    """

    def __init__(self, d3_kwargs=None, **mace_kwargs):
        super().__init__(**mace_kwargs)

        self.d3_kwargs = d3_kwargs if d3_kwargs is not None else {}
        self.d3_model = None

        # Internal cache: _atoms_to_batch intercepts the graph MACE builds so we
        # can reuse edge_index and unit_shifts for Fourier-D3 (smooth_cut path).
        self._cached_batch = None

        self.angstrom_to_bohr = (1 / 0.52917726)
        self.HARTREE_TO_EV = 27.21138505

    def _atoms_to_batch(self, atoms):
        """Intercept the graph construction in MACECalculator.

        Saves the batch dict (including edge_index and unit_shifts) so that
        `calculate` can pass the same neighbour list to Fourier-D3 for free,
        without rebuilding it.
        """
        batch = super()._atoms_to_batch(atoms)
        self._cached_batch = batch
        return batch

    def _ensure_d3_model(self, atoms, system_changes):
        """Initialize or reinitialize the FourierD3 model when needed.

        Rebuilds the model if the atomic species change (e.g., different structure).
        For cell-only changes the model is not rebuilt here; `_update_cell` handles that.
        If r_cut is not specified in d3_kwargs, defaults to MACE's r_max so that the
        neighbour list MACE builds covers the CN cutoff exactly.
        """
        if self.d3_model is None or "numbers" in system_changes:
            if "r_cut" not in self.d3_kwargs:
                self.d3_kwargs["r_cut"] = self.r_max  # r_max comes from the MACE model
                print('cutoff = ', self.d3_kwargs["r_cut"])

            self.d3_model = FourierD3(
                species=atoms.numbers,
                cell=atoms.cell.array,
                pbc=torch.tensor(atoms.pbc, device=self.device),
                device=self.device,
                **self.d3_kwargs
            )

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """Compute MACE energy/forces/stress and add Fourier-D3 dispersion on top.

        The strain tensor trick is used to compute stress via automatic differentiation:
        a zero strain tensor with requires_grad=True is applied to both positions and
        cell, so that d(energy)/d(strain) gives the stress tensor directly.

        For cnfunc='smooth_cut': the neighbour list from MACE's batch is reused.
        For cnfunc='d4': only positions (and cell via _update_cell) are needed;
                         edge_index and shifts are not passed to Fourier-D3.
        """
        # Run MACE forward pass (also populates self._cached_batch via _atoms_to_batch)
        super().calculate(atoms, properties, system_changes)

        if self._cached_batch is None:
            print("CACHE NOT BATCHED")

        self._ensure_d3_model(atoms, system_changes)

        batch = self._cached_batch
        cnfunc = self.d3_kwargs.get('cnfunc', 'smooth_cut')

        # Set up strained positions and cell for stress computation via autograd
        cell_d3 = batch["cell"].clone().detach().to(dtype=self.d3_model.dtype)

        strain = torch.zeros(3, 3, dtype=self.d3_model.dtype, device=self.device)
        strain.requires_grad_(True)

        # Strained cell: cell' = cell + strain @ cell  (infinitesimal deformation)
        strained_cell = cell_d3 + torch.einsum("ab,Ab->Aa", strain, cell_d3)
        self.d3_model._update_cell(strained_cell)

        positions_d3 = batch["positions"].clone().detach().to(dtype=self.d3_model.dtype)
        positions_d3.requires_grad_(True)

        # Apply the same strain to positions: pos' = pos + strain @ pos
        strained_pos = positions_d3 + torch.einsum("ab,ib->ia", strain, positions_d3)

        # Compute Fourier-D3 energy
        if cnfunc == 'smooth_cut':
            # Reuse MACE's neighbour list: shift vectors in Cartesian coords
            unit_shifts = batch["unit_shifts"].to(dtype=self.d3_model.dtype)
            strained_shifts = torch.matmul(unit_shifts, strained_cell)
            edge_index = batch["edge_index"]  # [2, n_edges]

            energy_hartree = self.d3_model(
                positions=strained_pos,
                edge_index=edge_index,
                shifts=strained_shifts,
            )
        elif cnfunc == 'd4':
            # D4 CN is computed entirely in k-space; no edge_index or shifts needed
            energy_hartree = self.d3_model(positions=strained_pos)
        else:
            raise ValueError(f"Unknown cnfunc '{cnfunc}'. Expected 'smooth_cut' or 'd4'.")

        energy_ev = energy_hartree * self.HARTREE_TO_EV

        # Backpropagate to get forces (dE/dpos) and stress (dE/dstrain)
        energy_ev.backward()

        d3_forces = -positions_d3.grad.detach().cpu().numpy()

        # Stress tensor: (1/Omega) * dE/dstrain, converted from Bohr^3 to Å^3
        d3_stress_3x3 = (
            strain.grad
            / self.d3_model.volume * (self.angstrom_to_bohr ** 3)
        ).detach().cpu().numpy()

        # Add D3 contributions to the MACE results already stored in self.results
        d3_e = energy_ev.item()
        self.results["energy"] += d3_e
        if "free_energy" in self.results:
            self.results["free_energy"] += d3_e

        self.results["forces"] += d3_forces

        if "stress" in self.results:
            d3_stress_voigt = full_3x3_to_voigt_6_stress(d3_stress_3x3)
            self.results["stress"] += d3_stress_voigt

        self._cached_batch = None
