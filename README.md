# fastd3

## Installation

To install this package, clone this directory and run 

```sh
pip install -e .
```

You can use `requirements.txt` to configure the environment, but you mainly need torch, ase, torch-pme, and matscipy. Additionally, you can install torch-dftd to run comparison tests with classic D3

## ASE calculator interface

To use `FastD3` with ASE

```python
import numpy as np
from ase.build import molecule
from fastd3 import FastD3ASECalculator
import torch

conf = molecule("C60", vacuum=5.0)
conf.set_pbc(True)

# the r_cut is for calculating the coordination number
calc = FastD3ASECalculator(
    r_cut=6.0,
    method="spme",
    interpolation_nodes=5,
    mesh_spacing=1.2,
    device=torch.device("cpu"),
)
calc._build_model(conf)
conf.calc = calc

conf.get_potential_energy()
conf.get_forces()
conf.get_stress()
```
