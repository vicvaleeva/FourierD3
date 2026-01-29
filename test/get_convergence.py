from fastd3 import FastD3ASECalculator
from fastd3 import FastD3
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
import torch
import time
from ase.io import read
from matscipy.neighbours import neighbour_list
import numpy as np

# reference ev/atom

mats = ['benzene', 'hea', 'water']

'''

calc = FastD3ASECalculator(r_cut=20.0, method="ewald", 
                           device=torch.device("cuda"), 
                           c6tol=0.01, k_cutoff=20.0,
                           verbose=False)
for mat in mats:
    confs = read('boxes/'+mat+'/250.xyz', ':')
    energies_ref = []
    forces_ref = []
    for i in range(0, 110):
        conf = confs[i]
        calc._build_model(conf)
        conf.calc = calc
        forces_ref.append(torch.tensor(conf.get_forces()*1000))
        energies_ref.append(conf.get_potential_energy() / len(conf))
        
        if (i-10)%20 == 0:
            print('reference', mat, 20*((i-10) // 20), '% done')
    torch.save(torch.tensor(energies_ref), 'results/'+mat+'/250_energies_ewaldref')
    torch.save(torch.stack(forces_ref), 'results/'+mat+'/250_forces_ewaldref')
  


cutoffs = [5, 10, 15]
for mat in mats:
    confs = read('boxes/'+mat+'/250.xyz', ':')
    for cutoff in cutoffs:
        calc = TorchDFTD3Calculator(device="cuda", 
                             damping="bj", xc='pbe', 
                             cnthr=20.0, cutoff=cutoff)
        energies = []
        forces = []
        times = []
        for i in range(0, 110):
            conf = confs[i]
            conf.calc = calc
            torch.cuda.synchronize()
            t0 = time.time()
            force_calc = conf.get_forces()*1000
            energy_calc = conf.get_potential_energy() / len(conf)
            torch.cuda.synchronize()
            t1 = time.time()
            energies.append(energy_calc)
            forces.append(torch.tensor(force_calc))
            times.append((t1-t0)*1000)
            if (i)%20 == 0:
                print('torch d3', mat, cutoff, 20*((i-1)//20), '% done')
        
        torch.save(torch.tensor(energies), 'results/'+mat+'/250_energies_torchd3_'+str(cutoff))
        torch.save(torch.stack(forces), 'results/'+mat+'/250_forces_torchd3_'+str(cutoff))
        torch.save(torch.tensor(times), 'results/'+mat+'/250_times_torchd3_'+str(cutoff))

'''
#20 cn SPME
mesh_spacings = [8, 3.5, 1.5, 0.75, 0.25]
for mat in mats:
    confs = read('boxes/'+mat+'/250.xyz', ':')
    for mesh_spacing in mesh_spacings:
        calc = FastD3ASECalculator(r_cut=20.0, method="spme", 
                           device=torch.device("cuda"), interpolation_nodes=5,
                           c6tol=0.01, mesh_spacing=mesh_spacing, verbose=False)
        energies = []
        forces = []
        times = []
        conf = confs[0]
        calc._build_model(confs[0])
        conf.calc = calc
        for i in range(0, 110):
            conf = confs[i]
            calc._build_model(conf)
            conf.calc = calc
            torch.cuda.synchronize()
            t0 = time.time()
            force_calc = conf.get_forces()*1000
            energy_calc = conf.get_potential_energy() / len(conf)
            torch.cuda.synchronize()
            t1 = time.time()
            energies.append(energy_calc)
            forces.append(torch.tensor(force_calc))
            times.append((t1-t0)*1000)
            
            if (i)%20 == 0:
                print('20cn SPME', mat, mesh_spacing, 20*((i)//20), '% done')
        torch.save(torch.tensor(energies), 'results/'+mat+'/250_energies_fastd3_'+str(mesh_spacing))
        torch.save(torch.stack(forces), 'results/'+mat+'/250_forces_fastd3_'+str(mesh_spacing))
        torch.save(torch.tensor(times), 'results/'+mat+'/250_timesWithNblist_fastd3_'+str(mesh_spacing))
      

# 6cn SPME
mesh_spacings = [8, 3.5, 1.5, 0.75, 0.25]
for mat in mats:
    confs = read('boxes/'+mat+'/250.xyz', ':')
    for mesh_spacing in mesh_spacings:
        calc = FastD3ASECalculator(r_cut=6.0, method="spme", 
                           device=torch.device("cuda"), interpolation_nodes=5,
                           c6tol=0.01, mesh_spacing=mesh_spacing, verbose=False)
        energies = []
        forces = []
        for i in range(0, 110):
            conf = confs[i]
            calc._build_model(conf)
            conf.calc = calc
            force_calc = conf.get_forces()*1000
            energy_calc = conf.get_potential_energy() / len(conf)
            energies.append(energy_calc)
            forces.append(torch.tensor(force_calc))
            if i == 0 and mat == 'benzene':
                print(force_calc[0, 0])
            if (i)%20 == 0:
                print('6cn SPME', mat, mesh_spacing, 20*((i)//20), '% done')
        torch.save(torch.tensor(energies), 'results/'+mat+'/250_energies_fastd3_6cn_'+str(mesh_spacing))
        torch.save(torch.stack(forces), 'results/'+mat+'/250_forces_fastd3_6cn_'+str(mesh_spacing))

# timings for 20cn SPME without nblist

def helper(atoms):
    sender, receiver, unit_shifts = neighbour_list(
        quantities="ijS",
        pbc=atoms.pbc,
        cell=atoms.cell,
        positions=atoms.positions,
        cutoff=20.0,
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

device = torch.device('cuda')
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

mesh_spacings = [8, 3.5, 1.5, 0.75, 0.25]
for mat in mats:
    confs = read('boxes/'+mat+'/250.xyz', ':')
    for mesh_spacing in mesh_spacings:
        conf = confs[0]
        calc = FastD3(
            species=conf.numbers,
            cell=torch.tensor(conf.cell.array, device=device, dtype=torch.float32),
            pbc=torch.tensor(conf.pbc, device=device),
            c6tol=0.01,
            device = device,
            method='spme',
            mesh_spacing=mesh_spacing,
            interpolation_nodes=5,
        )
        times = []

        for i in range(0, 110):
            conf.positions = confs[i].positions
            positions = torch.tensor(
                conf.positions,
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            )
            edge_index, unit_shifts = helper(conf)
            shifts = torch.matmul(unit_shifts, torch.tensor(conf.cell.array, device=device, dtype=torch.float32))
            
            torch.cuda.synchronize()
            start_event.record()
            
            energy = calc.forward(
                positions,
                edge_index,
                shifts,
            )
            energy_ev = energy * 27.21138505
            energy_ev.backward()
            forces_calc = -positions.grad * 1000

            end_event.record()
            torch.cuda.synchronize()
            
            time = start_event.elapsed_time(end_event)
            
            times.append(time)
            
            if (i-10)%20 == 0:
                print('20cn SPME', mat, mesh_spacing, 20*((i)//20), '% done')

        torch.save(torch.tensor(times), 'results/'+mat+'/250_timesWithoutNblist_fastd3_'+str(mesh_spacing))
        
