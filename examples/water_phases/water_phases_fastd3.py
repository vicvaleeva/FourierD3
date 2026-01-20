#!/usr/bin/env python3

import ase.io
import numpy as np
from pathlib import Path

# ASE
from ase.filters import ExpCellFilter
from ase.optimize import LBFGS
from ase.calculators.mixing import SumCalculator
from ase.io import write

# MACE
from mace.calculators.mace import MACECalculator

# FastD3
from fastd3 import FastD3ASECalculator
import torch

import copy
from ase.build import make_supercell


# ===========================================================
# CONFIG
# ===========================================================

INPUT_XYZ = "../../chho_data/water_phases/all_water_phases.xyz"


# MACE model
mace_model_path = "/home/jerry528/foundation_models/mace-omat-0-small.model"

# Supercell
n_supercell = 2
P = np.diag([n_supercell, n_supercell, n_supercell])

# FastD3
cutoff_A = 6.0

for k_cutoff in [1.0, 7.0]:
#k_cutoff = 5.0

    # Optimisation
    fmax = 1e-3
    max_steps = 2000

    OUT_DIR = Path(
        f"../results/water_phases/relax_trajs/"
        f"MACE_fastd3_sc{n_supercell}_rcut{cutoff_A}_kcut{k_cutoff}_fmax{fmax}"
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ENERGY_FILE = OUT_DIR / (
        f"relaxed_energies_sc{n_supercell}_"
        f"rcut{cutoff_A}_kcut{k_cutoff}_"
        f"fmax{fmax}.txt"
    )

    DEVICE = "cuda"


    # ===========================================================
    # 1. Load structures
    # ===========================================================

    ats = ase.io.read(INPUT_XYZ, ":")
    print(f"Loaded {len(ats)} structures")


    # ===========================================================
    # 2. Build calculators
    # ===========================================================

    mace_calc = MACECalculator(
        model_path=mace_model_path,
        device=DEVICE,
        energies=True,
        forces=True,
        stresses=True,
        enable_cueq=True,
    )

    d3_calc = FastD3ASECalculator(
        r_cut=cutoff_A,
        method="ewald",
        device=torch.device(DEVICE),
        k_cutoff=k_cutoff,
    )

    print("Calculators ready.")


    # ===========================================================
    # 3. Relax each structure and write its OWN trajectory
    # ===========================================================

    final_energies = []

    for i, at0 in enumerate(ats):
        print(f"\n======================================")
        print(f" Relaxing structure {i:4d}")
        print("======================================")

        # Output file for THIS structure
        traj_file = OUT_DIR / f"relax_frame_{i:04d}.extxyz"

        # Remove old trajectory if exists
        if traj_file.exists():
            traj_file.unlink()

        # Build supercell
        at = copy.deepcopy(at0)
        at.pbc = True
        at = make_supercell(at, P)

        # Build D3 model for this structure
        d3_calc._build_model(at)

        # Combined calculator
        combined_calc = SumCalculator([mace_calc, d3_calc])
        at.set_calculator(combined_calc)

        # Flexible cell relaxation
        ucf = ExpCellFilter(at, hydrostatic_strain=True)

        # -------------------------------------------------------
        # Collect relaxation steps for THIS structure
        # -------------------------------------------------------
        frames = []
        step_counter = 0

        def log_step():
            global step_counter
            step_counter += 1

            a = ucf.atoms

            # attach metadata
            a.info["frame_id"] = i
            a.info["step"] = step_counter

            # store energy explicitly
            try:
                a.info["energy"] = a.get_potential_energy()
            except Exception:
                pass

            frames.append(a.copy())

        # Optimiser
        opt = LBFGS(ucf)
        opt.attach(log_step, interval=1)

        # Run
        opt.run(fmax=fmax, steps=max_steps)

        # Final energy
        E = at.get_potential_energy()
        final_energies.append(E)

        print(f"[frame {i:4d}] final E = {E:.8f} eV")
        print(f"             steps collected = {len(frames)}")

        # -------------------------------------------------------
        # Write THIS relaxation trajectory
        # -------------------------------------------------------
        write(traj_file, frames, format="extxyz")

        print(f"[OK] wrote trajectory -> {traj_file}")


    # ===========================================================
    # 4. Save final energies summary
    # ===========================================================

    np.savetxt(
        ENERGY_FILE,
        np.column_stack([np.arange(len(final_energies)), final_energies]),
        header="index   relaxed_energy_eV",
    )

    print("\n======================================")
    print("All relaxations finished.")
    print(f"[OK] Final energies -> {ENERGY_FILE}")
    print(f"[OK] Trajectories in -> {OUT_DIR}")
    print("======================================")
