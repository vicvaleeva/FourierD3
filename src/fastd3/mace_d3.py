import torch
import numpy as np
from ase.calculators.calculator import all_changes
from ase.stress import full_3x3_to_voigt_6_stress

# Import from your provided files
from mace.calculators import MACECalculator
from fastd3 import FastD3

class MACEFastD3Calculator(MACECalculator):
    """
    Hybrid Calculator: MACE (Energy/Forces) + FastD3 (Dispersion).
    Reuses the neighbor list (edge_index, unit_shifts) from MACE to save time.
    """
    def __init__(self, d3_kwargs=None, **mace_kwargs):
        super().__init__(**mace_kwargs)
        
        self.d3_kwargs = d3_kwargs if d3_kwargs is not None else {}
        self.d3_model = None
        
        # Internal state to intercept MACE's graph
        self._cached_batch = None
        
        # Constants from your original calculator.py
        self.angstrom_to_bohr = (1 / 0.52917726)
        self.HARTREE_TO_EV = 27.21138505

    def _atoms_to_batch(self, atoms):
        """
        Intercept the batch creation in MACECalculator.
        This saves the graph (edge_index, unit_shifts) constructed by MACE
        so we can reuse it for FastD3.
        """
        batch = super()._atoms_to_batch(atoms)
        self._cached_batch = batch
        return batch

    def _ensure_d3_model(self, atoms, system_changes):
        """
        Initialize or update the FastD3 model.
        Only updates the cell if 'cell' is in system_changes.
        """
        if self.d3_model is None or "numbers" in system_changes:
            if "r_cut" not in self.d3_kwargs:
                self.d3_kwargs["r_cut"] = self.r_max  # self.r_max comes from MACE models
                print('cutoff = ', self.d3_kwargs["r_cut"])

            self.d3_model = FastD3(
                species=atoms.numbers,
                cell=atoms.cell.array,
                pbc=torch.tensor(atoms.pbc, device=self.device),
                device=self.device,
                **self.d3_kwargs
            )

    
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        
        if self._cached_batch is None:
            print("CACHE NOT BATCHED")

        self._ensure_d3_model(atoms, system_changes)
        
        batch = self._cached_batch
        
        edge_index = batch["edge_index"]            # [2, n_edges]
        unit_shifts = batch["unit_shifts"].to(dtype=self.d3_model.dtype)          # [n_edges, 3]
        
        positions_d3 = batch["positions"].clone().detach().to(dtype=self.d3_model.dtype)
        positions_d3.requires_grad_(True)
        
        cell_d3 = batch["cell"].clone().detach().to(dtype=self.d3_model.dtype)
        
        strain = torch.zeros(3, 3, dtype=self.d3_model.dtype, device=self.device)
        strain.requires_grad_(True)
        
        strained_cell = cell_d3 + torch.einsum("ab,Ab->Aa", strain, cell_d3)
        
        self.d3_model._update_cell(strained_cell)
        
        strained_pos = positions_d3 + torch.einsum("ab,ib->ia", strain, positions_d3)
        
        strained_shifts = torch.matmul(unit_shifts, strained_cell)

        energy_hartree = self.d3_model(
            positions=strained_pos,
            edge_index=edge_index,
            shifts=strained_shifts
        )
        
        energy_ev = energy_hartree * self.HARTREE_TO_EV
        
        energy_ev.backward()
        
        d3_forces = -positions_d3.grad.detach().cpu().numpy()
        
        d3_stress_3x3 = (
            strain.grad
            / self.d3_model.volume * (self.angstrom_to_bohr ** 3)
        ).detach().cpu().numpy()
        
        d3_e = energy_ev.item()
        self.results["energy"] += d3_e
        if "free_energy" in self.results:
            self.results["free_energy"] += d3_e
            
        self.results["forces"] += d3_forces
        
        if "stress" in self.results:
            d3_stress_voigt = full_3x3_to_voigt_6_stress(d3_stress_3x3)
            self.results["stress"] += d3_stress_voigt

        self._cached_batch = None