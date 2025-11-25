from fastd3 import FastD3
from ase import Atoms
from ase.build import molecule

w = molecule('H2O')
w.center(vacuum=2.0)

# REPEAT: (6, 6, 6)
# 6 * 6 * 6 = 216 molecules
atoms = w.repeat((6, 6, 6))
atoms.set_pbc(True)

# Density scaling: ~3.1 Angstroms per molecule length
box_len = 6 * 3.1 
atoms.set_cell([box_len, box_len, box_len])

calc = FastD3(species=atoms.numbers, cell=atoms.cell, pbc=atoms.pbc)
print(calc.kspace_filter._kfilter.shape)