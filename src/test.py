from fastd3 import FastD3
from ase.build import molecule
from ase import Atoms
import torch
import numpy as np
from matscipy.neighbours import neighbour_list

atoms = molecule('H2O')
box_len=3.1
atoms.set_cell([box_len, box_len, box_len])
atoms.center()
atoms = atoms.repeat((4, 4, 4))
atoms.set_pbc(True)

calc = FastD3(species=atoms.numbers, cell=atoms.cell, pbc=atoms.pbc, mesh_spacing=0.5, c6tol=1)

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
energy = calc.forward(positions, edge_index, shifts, r_cut)
energy = energy*27.211396
energy.backward()
print(-positions.grad*1000)