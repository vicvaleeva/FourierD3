from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from ase.io import read
import torch
import time
import numpy as np


device = torch.device('cuda')
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)


mats = ['water', 'benzene', 'hea']
sizes = [100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 25000]
times_torchd3 = {}
for mat in mats:
    times_torchd3[mat] = {}
    for size in sizes:
        times_torchd3[mat][size] = []

        confs = read('boxes/'+mat+'/'+str(size)+'.xyz', index=":")
        conf = confs[0]
        conf.set_pbc(True)

        calc = TorchDFTD3Calculator(
            device="cuda",
            damping="bj",
            xc='pbe',
            cnthr=6,
            cutoff=15,
            dtype=torch.float32,
        )

        oom = False
        for i in range(0, len(confs)):
            conf.positions = confs[i].positions
            conf.calc = calc

            try:
                torch.cuda.synchronize()
                start_event.record()

                force_calc = conf.get_forces() * 1000
                energy_calc = conf.get_potential_energy() / len(conf)
                stress_calc = conf.get_stress()

                end_event.record()
                torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError:
                print('torchd3 OOM ' + mat + ' ' + str(size) + ' at frame ' + str(i) + ', skipping rest of this size')
                # drop references and reclaim memory before moving on
                force_calc = energy_calc = stress_calc = None
                calc.results = {}
                torch.cuda.empty_cache()
                # NaN sentinel marks this size as OOM so the plot skips the point
                times_torchd3[mat][size].append(float('nan'))
                oom = True
                break

            time = start_event.elapsed_time(end_event)
            times_torchd3[mat][size].append(time)

            if i % 20 == 0:
                print('torchd3 ' + mat + str(size) + ' ' + str(i) + '% done')

        # free the calculator/model for this size before allocating the next
        del calc
        torch.cuda.empty_cache()

        if times_torchd3[mat][size]:
            torch.save(torch.tensor(times_torchd3[mat][size]), 'results/'+mat+'/'+str(size)+'_torchd3times_withNblist')

        # larger systems of the same material will also OOM, so skip them
        if oom:
            print('torchd3 skipping remaining sizes for ' + mat + ' after OOM at ' + str(size))
            break
