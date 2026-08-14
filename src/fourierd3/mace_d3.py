import numpy as np
import torch
from ase.calculators.calculator import all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from mace.calculators import MACECalculator
from fourierd3 import FourierD3
from fourierd3.neighborlist import SkinNeighborList


class MACEFourierD3Calculator(MACECalculator):
    """Hybrid ASE calculator: MACE short-range energy + Fourier-D3 dispersion.

    MACE provides the short-range energy, forces, and stress. Fourier-D3 adds the
    long-range DFT-D3 dispersion correction on top. When cnfunc='smooth_cut' and
    the CN cutoff equals MACE's r_max (the default, since r_cut defaults to r_max),
    the neighbour list already built by MACE is reused for the CN computation,
    making the marginal cost of Fourier-D3 very small (no extra neighbour list).
    If r_cut differs from r_max, MACE's graph is the wrong list — too short to
    cover the CN sum, or longer than needed — so a separate list is built here at
    r_cut, optionally a cached skin list. When cnfunc='d4', no neighbour list is
    needed at all.

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
        skin:      buffer width in Å for the separate CN neighbour list (only
                   used when r_cut != r_max, since MACE rebuilds its own graph
                   every step regardless). 0 rebuilds every step.
        every:     only consider rebuilding the fallback list every `every` calls.
        delay:     never rebuild the fallback list until `delay` calls after the
                   last rebuild.
        check:     rebuild only when atoms may have left the buffer region.
        **mace_kwargs: all other keyword arguments are forwarded to MACECalculator.
    """

    def __init__(self, d3_kwargs=None, skin=0.0, every=1, delay=0, check=True,
                 **mace_kwargs):
        super().__init__(**mace_kwargs)

        self.d3_kwargs = d3_kwargs if d3_kwargs is not None else {}
        self.d3_model = None

        # Species the current d3_model was built for; a change forces a rebuild
        self._d3_species = None

        # Skin-list settings for the fallback neighbour list (see _ensure_d3_model)
        self._skin_kwargs = dict(skin=skin, every=every, delay=delay, check=check)
        self.neighbor_list = None

        # Whether MACE's own graph covers the CN cutoff (set in _ensure_d3_model)
        self._reuse_mace_graph = True

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

        The rebuild test compares the actual atomic numbers rather than trusting
        `system_changes`, which is `all_changes` whenever `calculate` is invoked
        directly (as MD drivers and benchmarks often do). Keying off the species
        array keeps the C6 tensor decomposition — the expensive part of
        construction — out of the per-step path.

        Also decides where the CN neighbour list comes from. MACE always builds
        its graph at r_max, so it is the right list only when r_cut == r_max;
        otherwise a separate list is built here at r_cut.
        """
        species_changed = (
            self._d3_species is None
            or not np.array_equal(atoms.numbers, self._d3_species)
        )

        if self.d3_model is None or species_changed:
            self._d3_species = atoms.numbers.copy()

            if "r_cut" not in self.d3_kwargs:
                self.d3_kwargs["r_cut"] = self.r_max  # r_max comes from the MACE model

            r_cut = float(self.d3_kwargs["r_cut"])
            cnfunc = self.d3_kwargs.get("cnfunc", "smooth_cut")

            # MACE's graph is always built at r_max. Reuse it only when it is
            # exactly the CN cutoff. If r_cut > r_max, reusing it would silently
            # truncate the CN sum; if r_cut < r_max, the graph carries pairs the
            # CN does not want (they are masked off inside FourierD3, but the
            # list is needlessly large). Either way, build our own.
            self._reuse_mace_graph = (
                cnfunc != "smooth_cut" or abs(r_cut - self.r_max) < 1e-8
            )

            if self._reuse_mace_graph:
                self.neighbor_list = None
            else:
                print(
                    f"[fourierd3] CN r_cut ({r_cut:.3f} A) differs from the MACE "
                    f"cutoff r_max ({self.r_max:.3f} A); building a separate CN "
                    f"neighbour list at r_cut instead of reusing MACE's graph."
                )
                self.neighbor_list = SkinNeighborList(
                    cutoff=r_cut,
                    device=self.device,
                    dtype=self.d3_kwargs.get("dtype", torch.float32),
                    verbose=True,
                    **self._skin_kwargs,
                )

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

        For cnfunc='smooth_cut': the neighbour list from MACE's batch is reused
                         when it covers r_cut, otherwise a separate list is built.
        For cnfunc='d4': only positions (and cell via _update_cell) are needed;
                         edge_index and shifts are not passed to Fourier-D3.
        """
        # Run MACE forward pass (also populates self._cached_batch via _atoms_to_batch)
        super().calculate(atoms, properties, system_changes)

        if self._cached_batch is None:
            raise RuntimeError(
                "MACE did not populate the graph cache; _atoms_to_batch was not "
                "called during the MACE forward pass."
            )

        atoms = atoms if atoms is not None else self.atoms

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
            if self._reuse_mace_graph:
                # r_cut == r_max: MACE's graph is exactly the list we need
                unit_shifts = batch["unit_shifts"].to(dtype=self.d3_model.dtype)
                edge_index = batch["edge_index"]  # [2, n_edges]
            else:
                # r_cut != r_max: MACE's graph is the wrong cutoff, use our own
                edge_index, unit_shifts = self.neighbor_list.get(atoms)
                unit_shifts = unit_shifts.to(dtype=self.d3_model.dtype)

            strained_shifts = torch.matmul(unit_shifts, strained_cell)

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
