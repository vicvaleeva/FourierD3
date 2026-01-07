import numpy as np
from ase.build import molecule
from fastd3 import FastD3ASECalculator
import torch

conf = molecule("C6H6", vacuum=5.0)
conf.set_pbc(True)

calc = FastD3ASECalculator(
    r_cut=6.0,
    method="ewald",
    device=torch.device("cpu"),
)
calc._build_model(conf)

conf.calc = calc

strains = np.linspace(-0.02, 0.02, 9)  # ±2%
energies = []

cell0 = conf.cell.copy()

for eps in strains:
    cell = cell0.copy()
    cell[0, 0] *= (1.0 + eps)   # uniaxial x strain
    conf.set_cell(cell, scale_atoms=True)

    E = conf.get_potential_energy()
    energies.append(E)

    print(f"eps = {eps:+.4f}, E = {E:.6f} eV")
