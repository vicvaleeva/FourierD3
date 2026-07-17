# FourierD3

## Citation

If you use this work in any manner, please cite:

> Valeeva, V.; Ho, CH; Geiger, M.; Pellegrini, F.; Csányi, G.; Kucukbenli, E.; Ortner, C.
> *A fast summation method for the DFT-D3 dispersion correction.* arXiv:2607.15103 (2026).

```bibtex
@article{fourierd3,
  title   = {A fast summation method for the DFT-D3 dispersion correction},
  author  = {Valeeva, V. and Ho, CH. and Geiger, M. and Pellegrini, F.
             and Cs{\'a}nyi, G. and Kucukbenli, E. and Ortner, C.},
  journal = {arXiv preprint arXiv:2607.15103},
  year    = {2026},
  url     = {https://arxiv.org/abs/2607.15103}
}
```

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
