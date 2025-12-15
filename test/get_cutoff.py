from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from ase.io import read
import torch
import time

mats = ['water', 'benzene', 'hea']
cutoffs = [6, 8, 10, 12, 14, 16, 18, 20]
forces = {}
for mat in mats:
    forces[mat] = {}
    for cutoff in cutoffs:
        forces[mat][cutoff] = []
        confs = read('boxes/'+mat+'/2500.xyz', index=":")
        conf = confs[0]
        conf.set_pbc(True)
        calc_long = TorchDFTD3Calculator(atoms=conf, device="cuda", damping="bj", xc='pbe', cnthr=cutoff, cutoff=cutoff)
        
        for i in range(10, len(confs)):
            conf.positions = confs[i].positions
            
            forces_calc = conf.get_forces()*1000
            
            forces[mat][cutoff].append(torch.tensor(forces_calc, dtype=torch.float64))
            
            if i % 20 == 0:
                print(mat + str(cutoff) + ' ' +  str(i) + '% done')

        print('Saving ' + mat + str(cutoff) + ' results')
        torch.save(torch.stack(forces[mat][cutoff]), 'results/'+mat+'/cutoff_'+str(cutoff))
            
        
            
            