from fastd3 import FastD3ASECalculator
from fastd3 import FastD3
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
import torch
import time
from ase.io import read
from matscipy.neighbours import neighbour_list
import numpy as np

torch.set_default_dtype(torch.float64)

mats = ['hea', 'benzene', 'water']

'''

# reference 
calc = FastD3ASECalculator(r_cut=6.0, method="ewald", 
                           device=torch.device("cuda"), 
                           c6tol=0.0001, k_cutoff=9.0,
                           verbose=False, dtype=torch.float64)
for mat in mats:
    confs = read('boxes/'+mat+'/1000.xyz', ':')
    energies_ref = []
    forces_ref = []
    calc._build_model(confs[0])
    for i in range(0, 110):
        conf = confs[i]
        conf.calc = calc
        forces_ref.append(torch.tensor(conf.get_forces()*1000))
        energies_ref.append(conf.get_potential_energy() / len(conf))
        
        if (i)%20 == 0:
            print('reference', mat, 20*((i) // 20), '% done')
    torch.save(torch.tensor(energies_ref), 'results/'+mat+'/1000_energies_ewaldref')
    torch.save(torch.stack(forces_ref), 'results/'+mat+'/1000_forces_ewaldref')
 




cutoffs = [40, 30, 20, 15, 10, 6]
for mat in mats:
    confs = read('boxes/'+mat+'/1000.xyz', ':')
    for cutoff in cutoffs:
        calc = TorchDFTD3Calculator(device="cuda", 
                             damping="bj", xc='pbe', 
                             cnthr=6, cutoff=cutoff, dtype=torch.float32)
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
                print('torch d3', mat, cutoff, 20*((i)//20), '% done')
        
        torch.save(torch.tensor(energies), 'results/'+mat+'/1000_energies_torchd3_'+str(cutoff))
        torch.save(torch.stack(forces), 'results/'+mat+'/1000_forces_torchd3_'+str(cutoff))
        torch.save(torch.tensor(times), 'results/'+mat+'/1000_times_torchd3_'+str(cutoff))
        



  
# 6cn SPME
mesh_spacings = [13, 6, 2.5, 1.2, 0.4]
for mat in mats:
    confs = read('boxes/'+mat+'/1000.xyz', ':')
    for mesh_spacing in mesh_spacings:
        calc = FastD3ASECalculator(r_cut=6.0, method="spme", 
                           device=torch.device("cuda"), interpolation_nodes=5,
                           c6tol=0.001, mesh_spacing=mesh_spacing, verbose=True, dtype=torch.float64)
        energies = []
        forces = []
        times = []
        calc._build_model(confs[-1])
        for i in range(0, 110):
        
            conf = confs[i]
            conf.calc = calc
            torch.cuda.synchronize()
            t0 = time.time()
            stress_calc = conf.get_stress()
            force_calc = conf.get_forces()*1000
            energy_calc = conf.get_potential_energy() / len(conf)
            torch.cuda.synchronize()
            t1 = time.time()
            energies.append(energy_calc)
            forces.append(torch.tensor(force_calc))
            times.append((t1-t0)*1000)
            if (i)%20 == 0:
                print('6cn SPME', mat, mesh_spacing, 20*((i)//20), '% done')
        torch.save(torch.tensor(energies), 'results/'+mat+'/1000_energies_fastd3_'+str(mesh_spacing))
        torch.save(torch.stack(forces), 'results/'+mat+'/1000_forces_fastd3_'+str(mesh_spacing))
        torch.save(torch.tensor(times), 'results/'+mat+'/1000_timesWithNblist_fastd3_'+str(mesh_spacing)) 
        


# timings for 6cn SPME without nblist
mesh_spacings = [13, 6, 2.5, 1.2, 0.4]
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
        dtype=torch.float64,
        device=device,
    )

    return edge_index, unit_shifts
        

for mat in mats:
    confs = read('boxes/'+mat+'/1000.xyz', ':')
    for mesh_spacing in mesh_spacings:
        conf = confs[0]
        calc = FastD3(
            species=conf.numbers,
            cell=torch.tensor(conf.cell.array, device=device, dtype=torch.float64),
            pbc=torch.tensor(conf.pbc, device=device),
            c6tol=0.001,
            device = device,
            dtype=torch.float64,
            method='spme',
            mesh_spacing=mesh_spacing,
            interpolation_nodes=5,
            verbose = False
        )
        times = []

        for i in range(0, 110):
            conf.positions = confs[i].positions

            edge_index, unit_shifts = helper(conf)

            torch.cuda.synchronize()
            start_event.record()

            positions = torch.tensor(
                conf.positions,
                dtype=torch.float64,
                device=device,
                requires_grad=True,
            )

            cell = torch.tensor(conf.cell.array, dtype=torch.float64, device=device)

            strain = torch.zeros(3, 3, dtype=torch.float64, device=device)
            strain.requires_grad_(True)

            strained_cell = cell + torch.einsum("ab,Ab->Aa", strain, cell)

            shifts = torch.matmul(unit_shifts, strained_cell)

            calc._update_cell(cell=strained_cell)

            energy = calc(
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

            times.append(time)

            if (i-10)%20 == 0:
                print('6cn SPME', mat, mesh_spacing, 20*((i)//20), '% done')

        torch.save(torch.tensor(times), 'results/'+mat+'/1000_timesWithoutNblist_fastd3_'+str(mesh_spacing))

'''