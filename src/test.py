from fastd3 import FastD3
from ase.build import molecule
from ase import Atoms
from ase.units import Bohr
import torch
import time
import numpy as np
from matscipy.neighbours import neighbour_list
from ase.build import add_adsorbate, fcc111, bulk
from ase.visualize import view
from ase.io import read

atoms = read('SiO2.xyz', index=-1)
atoms.set_pbc(True)
atoms0 = atoms.repeat((1, 1, 1))
view(atoms0)


calc = FastD3(species=atoms0.numbers, cell=atoms0.cell, pbc=atoms0.pbc, c6tol=0.01, method='ewald', k_cutoff=4.0)

sender, receiver, unit_shifts = neighbour_list(
        quantities="ijS",
        pbc=atoms0.pbc,
        cell=atoms0.cell,
        positions=atoms0.positions,
        cutoff=5.0
)

edge_index = np.stack((sender, receiver))
shifts = np.dot(unit_shifts, atoms0.cell)

positions = torch.from_numpy(atoms0.positions)
r_cut = torch.tensor(5.0)
edge_index = torch.from_numpy(edge_index)
shifts = torch.from_numpy(shifts)
positions.requires_grad_(True)
start = time.time()
energy = calc.forward(positions, edge_index, shifts, r_cut)
energy = energy*27.21138505
#print(energy)
energy.backward()
pme_time = time.time() - start
a = -positions.grad*1000

from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

atoms1 = atoms.repeat((1, 1, 1))
#view(atoms1)

calc = TorchDFTD3Calculator(atoms=atoms1, device="cpu", damping="bj", xc='pbe')
start = time.time()
forces = atoms1.get_forces()
d3_time = time.time() - start
b = torch.tensor(forces)*1000

print()
print('Force MAE:', torch.mean(torch.abs(a-b)).item(), 'meV/A')
print('D3 time:', d3_time)
print()
print('Ewald-D3 time:', pme_time)
print()

calc = TorchDFTD3Calculator(atoms=atoms1, device="cpu", damping="bj", xc='pbe', cnthr = 18 * Bohr, cutoff = 18 * Bohr)
start = time.time()
forces = atoms1.get_forces()
d3_time = time.time() - start
c = torch.tensor(forces)*1000
print('Truncated D3 error:', torch.mean(torch.abs(b-c)).item(), 'meV/A')
print('Truncated D3 Time:', d3_time)

