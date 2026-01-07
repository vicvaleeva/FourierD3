from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from ase.io import read
import torch
import time

# This script generates reference forces with long-cutoff classic D3 for benchmarking

mats = ['water', 'benzene', 'hea']
sizes = ['100', '250', '500', '750', '1000', '2500', '5000', '7500', '10000', '25000']
forces = {}
times = {}
for mat in mats:
    forces[mat] = {}
    times[mat] = {}
    for size in sizes:
        forces[mat][size] = []
        times[mat][size] = []
        confs = read('boxes/'+mat+'/'+size+'.xyz', index=":")
        conf = confs[0]
        conf.set_pbc(True)
        calc_long = TorchDFTD3Calculator(atoms=conf, device="cuda", damping="bj", xc='pbe', cnthr=25.0, cutoff=50.0)
        for i in range(10):
            conf.positions = confs[i].positions
            _ = conf.get_forces()
            
        print(mat + size + ' warmup done')
        
        for i in range(10, len(confs)):
            conf.positions = confs[i].positions
            torch.cuda.synchronize()
            t0 = time.time()
            
            forces_calc = conf.get_forces()*1000
            
            torch.cuda.synchronize()
            t1 = time.time()
            
            forces[mat][size].append(torch.tensor(forces_calc, dtype=torch.float64))
            times[mat][size].append((t1 - t0)*1000)
            
            if i % 20 == 0:
                print(mat + size + ' ' +  str(i) + '% done')

        print('Saving ' + mat + size + ' results')
        torch.save(torch.stack(forces[mat][size]), 'results/'+mat+'/'+size+'_forces')
        torch.save(torch.tensor(times[mat][size], dtype=torch.float64), 'results/'+mat+'/'+size+'_times')
            
        
            
            