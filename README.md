# FourierD3

## Installation

To install this package, clone this directory and run

```sh
pip install -e .
```

This pulls in all runtime dependencies (`torch`, `ase`, `torch-pme`, `matscipy`, ...) automatically. To run comparison tests against the classic D3 implementation, install the optional `dftd` extra:

```sh
pip install -e ".[dftd]"
```

For MACE support, use the `mace` extra (`pip install -e ".[mace]"`). The pinned `requirements.txt` is also provided if you want to reproduce the exact tested environment.

## ASE calculator interface

To use `FourierD3` with ASE

```python
import numpy as np
from ase.build import molecule
from fourierd3 import FourierD3ASECalculator
import torch

conf = molecule("C60", vacuum=5.0)
conf.set_pbc(True)

# the r_cut is for calculating the coordination number
calc = FourierD3ASECalculator(
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
