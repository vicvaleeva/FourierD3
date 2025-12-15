from fastd3 import FastD3
from ase.visualize import view
import torch
import time
import numpy as np
from matscipy.neighbours import neighbour_list
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from ase.io import read

atoms = read('boxes/hea/500.xyz', index=-1)
view(atoms)
atoms.set_pbc(True)

calc = FastD3(species=atoms.numbers, cell=atoms.cell, verbose=True, c6tol=1, method='pme', mesh_spacing=2.0, device='gpu')

sender, receiver, unit_shifts = neighbour_list(
        quantities="ijS",
        pbc=atoms.pbc,
        cell=atoms.cell,
        positions=atoms.positions,
        cutoff=5.0
)

edge_index = np.stack((sender, receiver))
shifts = np.dot(unit_shifts, atoms.cell)

positions = torch.from_numpy(atoms.positions)
r_cut = torch.tensor(5.0)
edge_index = torch.from_numpy(edge_index)
shifts = torch.from_numpy(shifts)
positions.requires_grad_(True)


fastd3_st = time.time()
energy_fastd3 = calc.forward(positions, edge_index, shifts, r_cut)
energy_fastd3 *= 27.21138505
energy_fastd3.backward()
fastd3_end = time.time() - fastd3_st
forces_fastd3 = -positions.grad*1000
positions.requires_grad_(False)

calc_long = TorchDFTD3Calculator(atoms=atoms, device="cpu", damping="bj", xc='pbe', cnthr=40, cutoff=40)
forces_longd3 = atoms.get_forces()*1000
forces_longd3 = torch.tensor(forces_longd3)

calc_short = TorchDFTD3Calculator(atoms=atoms, device="cpu", damping="bj", xc='pbe', cnthr=9, cutoff=9)
shortd3_st = time.time()
forces_shortd3 = atoms.get_forces()*1000
shortd3_end = time.time() - shortd3_st
forces_shortd3 = torch.tensor(forces_shortd3)

print('Fast D3 error', torch.mean(torch.abs(forces_fastd3-forces_longd3)).item())
print('Short D3 error', torch.mean(torch.abs(forces_shortd3 - forces_longd3)).item())
print()
print('Fast D3 time', fastd3_end)
print('Short D3 time', shortd3_end)