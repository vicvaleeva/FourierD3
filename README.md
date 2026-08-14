# FourierD3

## Citation

If you use this work in any manner, please cite:

> Valeeva, V.; Ho, C.H.; Geiger, M.; Pellegrini, F.; Csányi, G.; Kucukbenli, E.; Ortner, C.
> *A fast summation method for the DFT-D3 dispersion correction.* arXiv:2607.15103 (2026).

```bibtex
@article{fourierd3,
  title   = {A fast summation method for the DFT-D3 dispersion correction},
  author  = {Valeeva, V. and Ho, C.H. and Geiger, M. and Pellegrini, F.
             and Cs{\'a}nyi, G. and Kucukbenli, E. and Ortner, C.},
  journal = {arXiv preprint arXiv:2607.15103},
  year    = {2026},
  url     = {https://arxiv.org/abs/2607.15103}
}
```

## Installation

To install this package, clone this directory and run

```sh
pip install .
```

This pulls in all runtime dependencies (`torch`, `ase`, `torch-pme`, `matscipy`, ...) automatically. To run comparison tests against the classic D3 implementation, install the optional `dftd` extra:

```sh
pip install ".[dftd]"
```

For MACE support, use the `mace` extra (`pip install ".[mace]"`).

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

## [Experimental] Using skin-type neighborlist

Rebuilding the coordination-number neighbour list at every step can be slow,
particularly when `r_cut` is large. A neighbour list with a skin layer is
provided, with hyperparameters similar to those in
[LAMMPS](https://docs.lammps.org/neigh_modify.html#description):

```python
calc = FourierD3ASECalculator(r_cut=10.0, method="spme",
    every=1, delay=0, check=True, skin=1.0)  # skin-type nlist parameters
```

The list is built at `r_cut + skin` and reused for as long as `every`, `delay`
and `check` allow; `skin=0` (the default) rebuilds every step. The same
parameters exist on `MACEFourierD3Calculator`, where they apply only if
`r_cut != r_max` (otherwise MACE's own graph is reused).

However, we would like to stress that using a large coordination number neighbour list 
is ill-defined and not necessarily better, as shown in section A of the Appendix.
The intended use of the package is to re-use the neighbour list from the underlying
MLIP for the calculation of the coordination number, rather than compute a new one.

Adapted from the skin neighborlist in Jerry Ho's fork of
[torch-dftd](https://github.com/CheukHinHoJerry/torch-dftd).
