import os
from pathlib import Path
from ase.io import read, write
from ase.optimize import LBFGS
from ase.calculators.mixing import SumCalculator
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from mace.calculators.mace import MACECalculator

# ======================================================
# User settings
# ======================================================
configs_name = "graphite_ABC_4x4x4"
config_index = 0
mace_model_path = "/home/jerry528/foundation_models/mace-omat-0-small.model"

# ======================================================
# Paths
# ======================================================
configs_path = f"/home/jerry528/fastd3/chho_data/graphite/{configs_name}.xyz" #"/home/jerry528/fastd3/results/carbon_minimization/com_inter_0/cutoff_10A/minimized.xyz" #f"/home/jerry528/fastd3/chho_data/graphite/{configs_name}.xyz"
base_res_path = Path(f"results/carbon_minimization/{configs_name}_{config_index}/")
base_res_path.mkdir(parents=True, exist_ok=True)

bohr_per_ang = 1.88972612546


# ======================================================
# MACE calculator (ACE part)
# ======================================================
ace = MACECalculator(
    model_path=mace_model_path,
    device="cuda",
    energies=True,
    forces=True,
    stresses=True,
    enable_cueq=True,
)


# ======================================================
# Loop over cutoff values
# ======================================================
for cutoff_A in [8, 10, 12, 16, 20]:

    # ----- Prepare folder -----
    cutoff_folder = base_res_path / f"cutoff_{cutoff_A}A"
    cutoff_folder.mkdir(exist_ok=True)

    traj_file        = cutoff_folder / "opt.traj"
    traj_xyz         = cutoff_folder / "opt.xyz"
    minimizer_file   = cutoff_folder / "minimized.xyz"

    # ----- Create D3 calculator -----
    cutoff_bohr = cutoff_A * bohr_per_ang

    d3_calc = TorchDFTD3Calculator(
        atoms=None,
        damping="zero", # bj
        cutoff=cutoff_bohr,
        device="cuda",
    )

    # SumCalculator requires calculators in a list
    calculator = SumCalculator([ace, d3_calc])

    # ----- Load config -----
    atoms = read(configs_path, index=config_index)
    atoms.calc = calculator

    # ----- Run optimization -----
    dyn = LBFGS(atoms=atoms, trajectory=str(traj_file))
    dyn.run(fmax=0.001, steps=1000)

    # ----- Save results -----
    traj_atoms = read(traj_file, ":")

    write(traj_xyz, traj_atoms)          # write all frames as xyz
    write(minimizer_file, traj_atoms[-1])   # final relaxed structure

    print(f"[Done] cutoff {cutoff_A}Å → results saved to {cutoff_folder}")
