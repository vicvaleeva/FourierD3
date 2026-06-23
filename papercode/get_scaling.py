from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from fastd3 import FastD3
from matscipy.neighbours import neighbour_list
from ase.io import read
import torch
import time
import numpy as np


device = torch.device('cuda')
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

def helper(atoms):
    sender, receiver, unit_shifts = neighbour_list(
        quantities="ijS",
        pbc=atoms.pbc,
        cell=atoms.cell,
        positions=atoms.positions,
        cutoff=6.0,
    )

    edge_index = torch.tensor(
        np.stack((sender, receiver)),
        dtype=torch.long,
        device=device,
    )

    unit_shifts = torch.tensor(
        unit_shifts,
        dtype=torch.float32,
        device=device,
    )

    return edge_index, unit_shifts
    

mats = ['water', 'benzene', 'hea']
sizes = [100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 25000]
mesh_spacings = [2, 1.5, 2, 2, 2, 1.5, 1.5, 1.5, 2.0, 1.5]
times_pme = {}
for mat in mats:
    times_pme[mat] = {}
    for s in range(len(sizes)):
        size = sizes[s]
        mesh_spacing = mesh_spacings[s]
        times_pme[mat][size] = []
        
        confs = read('boxes/'+mat+'/'+str(size)+'.xyz', index=":")
        conf = confs[0]
        conf.set_pbc(True)
        calc = FastD3(
            species=conf.numbers,
            cell=conf.cell.array,
            pbc=torch.tensor(conf.pbc, device=device),
            c6tol=0.0001,
            device = device,
            method='spme',
            mesh_spacing=mesh_spacing,
            interpolation_nodes=4,
            dtype=torch.float32
        )
            
        for i in range(0, len(confs)):
            conf.positions = confs[i].positions
            
            
            
            edge_index, unit_shifts = helper(conf)
            
            torch.cuda.synchronize()
            start_event.record()
            
            positions = torch.tensor(
                conf.positions,
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            )
            
            cell = torch.tensor(conf.cell.array, dtype=torch.float32, device=device)

            strain = torch.zeros(3, 3, dtype=torch.float32, device=device)
            strain.requires_grad_(True)

            strained_cell = cell + torch.einsum("ab,Ab->Aa", strain, cell)
            
            shifts = torch.matmul(unit_shifts, strained_cell)
            
            calc._update_cell(cell=strained_cell)
            
            energy = calc.forward(
                positions,
                edge_index,
                shifts,
            )
            energy_ev = energy * 27.21138505
            energy_ev.backward()
            forces_calc = -positions.grad * 1000
            stress_calc = strain.grad / calc.volume

            end_event.record()
            torch.cuda.synchronize()
            
            time = start_event.elapsed_time(end_event)
            times_pme[mat][size].append(time)
            
            if i % 20 == 0:
                print('PME' + mat + str(size) + ' ' +  str(i) + '% done')
        
        torch.save(torch.tensor(times_pme[mat][size]), 'results/'+mat+'/'+str(size)+'_pmetimes_withoutNblist')
            
            
        
