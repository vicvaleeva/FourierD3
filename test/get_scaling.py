from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from ase.io import read
import torch
import time
from fastd3 import FastD3
import numpy as np
from matscipy.neighbours import neighbour_list

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

device = torch.device('cuda')
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
mats = ['hea', 'water', 'benzene']
sizes = [100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 25000]
mesh_spacings = [0.5, 0.5, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 2.0, 2.5]
times_pme = {}
times_trunc = {}
for mat in mats:
    times_pme[mat] = {}
    times_trunc[mat] = {}
    for s in range(len(sizes)):
        tmp_forces = []
        size = sizes[s]
        mesh_spacing = mesh_spacings[s]
        
        times_pme[mat][size] = []
        times_trunc[mat][size] = []
        
        confs = read('boxes/'+mat+'/'+str(size)+'.xyz', index=":")
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
            
            time_pme = start_event.elapsed_time(end_event)
            
            times_pme[mat][size].append(time_pme)
            tmp_forces.append(forces_calc.cpu())
            
            if i % 50 == 0:
                print('PME' + mat + str(size) + ' ' +  str(i) + '% done')
        
        tmp_forces = torch.stack(tmp_forces)        
        ref_forces = torch.load('results/'+mat+'/'+str(size)+'_forces')
        pme_error = torch.median(torch.amax(torch.abs(tmp_forces - ref_forces), dim=(1, 2))).item()

        print('Saving PME ' + mat + str(size) + ' results: accuracy ' + str(pme_error) + 'meV/A')
        torch.save(torch.tensor(times_pme[mat][size], dtype=torch.float64), 'results/'+mat+'/'+str(size)+'_pmetimes')
        
        print('Finding truncated cutoff...')
        trunc_error = 1e12
        cutoff = 6.0
        conf.positions = confs[-1].positions
        while trunc_error > pme_error:
            calc_d3 = TorchDFTD3Calculator(atoms=conf, device="cuda", damping="bj", xc='pbe', cnthr=cutoff, cutoff=cutoff)
            forces_calc = conf.get_forces()*1000
            trunc_error = torch.max(torch.abs(ref_forces[-1] - forces_calc))
            cutoff += 0.5
            
        cutoff -= 0.5
        print('Running truncated D3 with cutoff', cutoff, 'accuracy', trunc_error, 'mev/A')
        
        for i in range(10):
            conf.positions = confs[i].positions
            _ = conf.get_forces()
            
        for i in range(10, len(confs)):
            conf.positions = confs[i].positions
            torch.cuda.synchronize()
            t0 = time.time()
            
            forces_calc = conf.get_forces()*1000
            
            torch.cuda.synchronize()
            t1 = time.time()
            
            times_trunc[mat][size].append((t1 - t0)*1000)
            
            if i % 50 == 0:
                print(mat + str(size) + ' ' + str(i) + '% done')
                
        print('Saving truncD3 ' + mat + str(size))
        torch.save(torch.tensor(times_trunc[mat][size], dtype=torch.float64), 'results/'+mat+'/'+str(size)+'_trunctimes')
        
        
            
            