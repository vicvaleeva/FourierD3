from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from ase.io import read
import torch
import time
from fastd3 import FastD3
import numpy as np
from matscipy.neighbours import neighbour_list

device = torch.device('cuda')
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

def helper(conf):
    sender, receiver, unit_shifts = neighbour_list(
        quantities="ijS",
        pbc=conf.pbc,
        cell=conf.cell,
        positions=conf.positions,
        cutoff=6.0
    )

    edge_index = np.stack((sender, receiver))
    shifts = np.dot(unit_shifts, conf.cell)
    
    edge_index = torch.from_numpy(edge_index).to(device)
    shifts = torch.from_numpy(shifts).to(device)
    
    return edge_index, shifts

mats = ['water', 'benzene', 'hea']
mesh_spacings = [12.0, 6.0, 2.5, 1.2, 0.6, 0.3]
forces = {}
times = {}
for mat in mats:
    forces[mat] = {}
    times[mat] = {}
    for mesh_spacing in mesh_spacings:
        forces[mat][mesh_spacing] = []
        times[mat][mesh_spacing] = []
        confs = read('boxes/'+mat+'/2500.xyz', index=":")
        conf = confs[0]
        conf.set_pbc(True)
        calc = FastD3(species=conf.numbers, cell=conf.cell, verbose=True, c6tol=1, method='pme', mesh_spacing=mesh_spacing, device=device)
        r_cut = torch.tensor(6.0).to(device)
        for i in range(10):
            conf.positions = confs[i].positions
            
            positions = torch.from_numpy(conf.positions).to(device)
            edge_index, shifts = helper(conf)
            
            _ = calc.forward(positions, edge_index, shifts, r_cut)
        
        for i in range(10, len(confs)):
            conf.positions = confs[i].positions
            
            positions = torch.from_numpy(conf.positions).to(device)
            edge_index, shifts = helper(conf)
            
            positions.requires_grad_(True)
            
            torch.cuda.synchronize()
            start_event.record()
            
            energy_fastd3 = calc.forward(positions, edge_index, shifts, r_cut)
            energy_fastd3 *= 27.21138505
            energy_fastd3.backward()
            
            end_event.record()
            torch.cuda.synchronize()
            
            forces_calc = -positions.grad*1000
            positions.requires_grad_(False)
            
            time = start_event.elapsed_time(end_event)
            
            times[mat][mesh_spacing].append(time)
            forces[mat][mesh_spacing].append(forces_calc.cpu())
            
            if i % 50 == 0:
                print('PME' + mat + str(mesh_spacing) + ' ' +  str(i) + '% done')
            

        print('Saving ' + mat + str(mesh_spacing) + ' results')
        torch.save(torch.stack(forces[mat][mesh_spacing]), 'results/'+mat+'/f_meshspacing_'+str(mesh_spacing))
        torch.save(torch.tensor(times[mat][mesh_spacing]), 'results/'+mat+'/t_meshspacing_'+str(mesh_spacing))
            
        
            
            