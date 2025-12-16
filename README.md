# fastd3

## Installation

To install this package, clone this directory and run 

```sh
pip install -e .
```

You can use `requirements.txt` to configure the environment, but you mainly need torch, ase, torch-pme, and matscipy. Additionally, you can install torch-dftd to run comparison tests with classic D3

## Example usage

```python
import numpy as np
import torch

from fastd3 import FastD3
from ase.build import molecule
from matscipy.neighbours import neighbour_list

device = torch.device('cuda')
r_cut = torch.tensor(6.0).to(device)

# one would normally re-use the neighbour list from an MLIP,
# but here we build it manually for demostrations

def helper(conf):
    sender, receiver, unit_shifts = neighbour_list(
        quantities="ijS",
        pbc=conf.pbc,
        cell=conf.cell,
        positions=conf.positions,
        cutoff=6.0
    )

    edge_index = np.stack((sender, receiver))
    
    edge_index = torch.from_numpy(edge_index).to(device)
    unit_shifts = torch.from_numpy(unit_shifts).to(device, dtype=torch.float64)
    
    return edge_index, unit_shifts


# needed to compute stress

strain = torch.zeros(3, 3, dtype=torch.float64)
strain.requires_grad_(True)

# get a benzene molecule

conf = molecule('C6H6', vacuum=5.0)
conf.set_pbc(True)
cell = torch.from_numpy(atoms.cell.array).to(device)
strained_cell = cell + torch.einsum("ab,Ab->Aa", strain, cell)

# get positions

positions = torch.from_numpy(conf.positions).to(device)
positions.requires_grad_(True)

strained_pos = positions + torch.einsum("ab,ib->ia", strain, positions)

edge_index, unit_shifts = helper(conf)
strained_shifts = torch.matmul(unit_shifts, strained_cell)

# initialize the calculator

calc = FastD3(species=conf.numbers, cell=strained_cell, method='pme')

energy_fastd3 = calc.forward(strained_pos, edge_index, strained_shifts, r_cut)
energy_fastd3 *= 27.21138505 # convert from Hartree to eV
energy_fastd3.backward()
forces_calc = -positions.grad*1000 # forces in meV/Å
stress_calc = strain.grad / calc.volume # stress
```