from fourierd3 import FourierD3ASECalculator
from fourierd3 import FourierD3
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
import torch
import time
from ase.io import read
from matscipy.neighbours import neighbour_list
import numpy as np
from mace.calculators import MACECalculator

angstrom_to_bohr = (1 / 0.52917726)

mats = ['hea', 'benzene', 'water']
  
for mat in mats:
    confs = read('boxes/'+mat+'/1000.xyz', ':')
    calc = MACECalculator(model_paths='../mace-omat-0-medium.model', device="cuda", default_dtype="float64", enable_cueq=True)
    times = []
    conf = confs[0]
    conf.calc = calc
    for i in range(0, 110):
        conf.positions = confs[i].positions
        torch.cuda.synchronize()
        t0 = time.time()
        _ = conf.get_forces()*1000
        _ = conf.get_potential_energy() / len(conf)
        torch.cuda.synchronize()
        t1 = time.time()
        times.append((t1-t0)*1000)
        if (i)%20 == 0:
            print('mace', mat, 20*((i)//20), '% done')
    
    print(np.mean(times[-100:]))