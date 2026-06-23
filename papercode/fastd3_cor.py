#!/usr/bin/env python3

import os
import copy
import ase.io
from pathlib import Path
import numpy as np
import torch
import time  

# Calculators
from fastd3 import FastD3ASECalculator

# ===========================================================
# HELPER FUNCTION: KENDALL TAU DISTANCE
# ===========================================================
def count_pairwise_inversions(ref_ranking, current_ranking):
    """
    Counts how many pairs of structures have the wrong relative stability
    compared to the reference ranking.
    """
    ref_pos = {name: idx for idx, name in enumerate(ref_ranking)}
    inversions = 0
    n = len(current_ranking)
    
    for i in range(n):
        for j in range(i + 1, n):
            name_i = current_ranking[i]
            name_j = current_ranking[j]
            if ref_pos[name_i] > ref_pos[name_j]:
                inversions += 1
                
    return inversions

# ===========================================================
# HELPER FUNCTION: CUDA SYNC
# ===========================================================
def sync_device(device):
    """Ensures GPU tasks finish before the timer stops."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)

# ===========================================================
# CONFIG
# ===========================================================
INPUT_XYZ = "sio2.xyz" 
DEVICE = torch.device("cuda")
CUTOFF_CN = 6.0

# Number of times to repeat the energy calculation for averaging timings
NUM_REPEATS = 10  

# Progressing from low accuracy (1.0) to reference accuracy (30.0)
K_CUTOFFS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]

# ===========================================================
# 1. LOAD AND PREPROCESS STRUCTURES
# ===========================================================
ats0 = ase.io.read(INPUT_XYZ, ":")
print(f"Loaded {len(ats0)} structures from {INPUT_XYZ}.")

valid_ats = []
pbe_energies = []
names = []

for i, at_orig in enumerate(ats0):
    at = copy.deepcopy(at_orig)
    name = at.info.get('name', at.info.get('material_id', f"Phase_{i}"))
    
    try:
        if "total_energy_eV" in at.info:
            pbe_total_e = float(at.info["total_energy_eV"])
        elif "energy_per_atom_eV" in at.info:
            pbe_total_e = float(at.info["energy_per_atom_eV"]) * len(at)
        else:
            raise KeyError("PBE energy metadata not found in atoms.info")
            
        valid_ats.append(at)
        pbe_energies.append(pbe_total_e)
        names.append(name)
    except KeyError as e:
        print(f"[ERROR] Skipping {name}: {e}. Ensure the XYZ has the energy data.")

print(f"Proceeding with {len(valid_ats)} valid structures.")
total_possible_pairs = (len(valid_ats) * (len(valid_ats) - 1)) // 2
print(f"Total possible pairwise comparisons: {total_possible_pairs}")

# ===========================================================
# 2. COMPUTE REFERENCE ORDER (FastD3)
# ===========================================================
print("\n" + "="*60)
print(" COMPUTING REFERENCE ENERGIES (FastD3 Ewald, k_cutoff=30.0)")
print("="*60)

ref_calc = FastD3ASECalculator(
    r_cut=6.0, 
    method="ewald", 
    device=DEVICE, 
    c6tol=0.0001, 
    k_cutoff=30.0,
    verbose=False
)

ref_e_per_atom = []
ref_summary = []

for i, at in enumerate(valid_ats):
    ref_calc._build_model(at)
    at.calc = ref_calc
    d3_total_e = at.get_potential_energy()
    
    final_e = pbe_energies[i] + d3_total_e
    e_per_atom = final_e / len(at)
    
    ref_e_per_atom.append(e_per_atom)
    ref_summary.append([names[i], final_e, e_per_atom, len(at)])

# Determine Reference Ranking
ref_summary.sort(key=lambda x: x[2])
ref_ranking = [row[0] for row in ref_summary]
print("[OK] Reference energies computed.")

# ===========================================================
# 3. OUTER LOOP: EWALD K-CUTOFFS (FastD3)
# ===========================================================
cutoff_rankings = {}
pairwise_inversions = []
average_timings = []  
prev_ranking = None

for k_cut in K_CUTOFFS:
    print(f"\n" + "#"*60)
    print(f" PROCESSING K_CUTOFF: {k_cut} 1/A")
    print("#"*60)

    OUT_DIR = Path(f"../results/sio2/single_points/PBE_fastd3_cn{CUTOFF_CN}_k{k_cut}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ENERGY_FILE = OUT_DIR / f"sp_energies_k{k_cut}.txt"
    
    d3_calc = FastD3ASECalculator(
        r_cut=6.0, 
        method="ewald", 
        device=DEVICE, 
        c6tol=0.0001, 
        k_cutoff=k_cut,
        verbose=False
    )

    summary_data = []
    total_time_for_kcut = 0.0  

    for i, at in enumerate(valid_ats):
        
        d3_calc._build_model(at)
        at.calc = d3_calc

        # --- GPU/COMPUTE WARM-UP ---
        # Run once to compile kernels and allocate memory.
        # This run caches the energy, but we will clear it before the loop.
        _ = at.get_potential_energy()

        # --- START REPEATED TIMING BLOCK ---
        sync_device(DEVICE)
        start_time = time.perf_counter()
        
        for _ in range(NUM_REPEATS):
            # CLEAR THE ASE CACHE SO IT FORCES A RECALCULATION
            at.calc.results.clear() 
            d3_total_e = at.get_potential_energy()
            
        sync_device(DEVICE)
        end_time = time.perf_counter()
        # --- END REPEATED TIMING BLOCK ---
        
        avg_time_for_this_struct = (end_time - start_time) / NUM_REPEATS
        total_time_for_kcut += avg_time_for_this_struct
        
        final_e = pbe_energies[i] + d3_total_e
        e_per_atom = final_e / len(at)
        
        summary_data.append([names[i], final_e, e_per_atom, len(at)])

    avg_time_per_struct = total_time_for_kcut / len(valid_ats)
    average_timings.append((k_cut, avg_time_per_struct))

    summary_data.sort(key=lambda x: x[2])
    current_ranking = [row[0] for row in summary_data]
    cutoff_rankings[k_cut] = current_ranking

    num_inversions = count_pairwise_inversions(ref_ranking, current_ranking)
    pairwise_inversions.append((k_cut, num_inversions))

    print(f"=> Incorrect Relative Pairings (vs Ref): {num_inversions} / {total_possible_pairs}")
    print(f"=> Avg Compute Time per Structure:     {avg_time_per_struct:.6f} sec (averaged over {NUM_REPEATS} true runs)") 

    if prev_ranking is not None:
        prev_indices = {name: idx for idx, name in enumerate(prev_ranking)}
        changed_ids = [name for idx, name in enumerate(current_ranking) if prev_indices[name] != idx]
        
        if changed_ids:
            print(f"=> IDs that changed position since last k_cutoff: {', '.join(changed_ids)}")
        else:
            print("=> IDs that changed position since last k_cutoff: None (Stable)")
    else:
        print("=> First k_cutoff (no previous ranking to compare).")

    prev_ranking = current_ranking

    with open(ENERGY_FILE, "w") as f:
        f.write("Name                Total_Energy_eV    Energy_per_Atom_eV    Num_Atoms\n")
        for row in summary_data:
            f.write(f"{row[0]:<20} {row[1]:<18.8f} {row[2]:<21.8f} {row[3]:<10}\n")

# ===========================================================
# 4. SAVE INVERSION COUNTS, TIMINGS, AND STABILITY REPORT
# ===========================================================
print("\n" + "="*60)
print(" ANALYSIS & RANKING STABILITY REPORT ")
print("="*60)

RESULTS_DIR = Path("../results/sio2/single_points")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Save Inversions
ERROR_FILE = RESULTS_DIR / "pairwise_inversions_ewald_kcut.txt"
inversions_array = np.array(pairwise_inversions)
np.savetxt(ERROR_FILE, inversions_array, fmt=["%.2f", "%d"], header="K_Cutoff(1/A) Pairwise_Inversions")
print(f"Saved pairwise inversion counts to: {ERROR_FILE}")

# Save Timings
TIMING_FILE = RESULTS_DIR / "average_timings_ewald_kcut.txt"
timings_array = np.array(average_timings)
np.savetxt(TIMING_FILE, timings_array, fmt=["%.2f", "%.6f"], header="K_Cutoff(1/A) Avg_Time_Per_Structure(s)")
print(f"Saved average timings to:           {TIMING_FILE}")

final_ranking = cutoff_rankings[K_CUTOFFS[-1]]
stable_kcut = None

for i, k_cut in enumerate(K_CUTOFFS):
    is_stable = all(cutoff_rankings[k] == final_ranking for k in K_CUTOFFS[i:])
    if is_stable:
        stable_kcut = k_cut
        break

if stable_kcut is not None:
    print(f"-> The exact relative ranking stops changing at a k_cutoff of: {stable_kcut} 1/A")
else:
    print("-> WARNING: The exact ranking never stabilized completely. It was still fluctuating at the maximum k_cutoff.")