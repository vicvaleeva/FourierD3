from ase.build import molecule
from calculator import FastD3ASECalculator
import torch

conf = molecule("C6H6", vacuum=5.0)
conf.set_pbc(True)

# setup abstract object
calc = FastD3ASECalculator(
    r_cut=6.0,
    method="ewald",
    device = torch.device('cpu'),
    rebuild=True
)

# build model for specific config
calc._build_model(conf)

# set calculator
conf.calc = calc

# get results
E = conf.get_potential_energy()
F = conf.get_forces()
S = conf.get_stress(voigt=False)

# check
print("Energy (eV):", E)
print("Forces (eV/Å):", F)
print("Stress:", S)
