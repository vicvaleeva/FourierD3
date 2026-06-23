import numpy as np
import os

# --- Helper Functions ---
def get_pbc_distance(pos, box_length):
    """Wraps coordinates into the box [0, box_length)."""
    return pos % box_length

def write_frame(f, atoms, box_length, frame_num):
    """Writes a single frame in Extended XYZ format."""
    f.write(f"{len(atoms)}\n")
    # Extended XYZ header (Lattice="ax ay az bx by bz cx cy cz")
    f.write(f'Lattice="{box_length:.4f} 0.0 0.0 0.0 {box_length:.4f} 0.0 0.0 0.0 {box_length:.4f}" '
            f'Properties=species:S:1:pos:R:3 Frame={frame_num}\n')
    
    for atom_type, pos in atoms:
        f.write(f"{atom_type:<2} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")

# ==========================================
# 1. WATER GENERATOR
# ==========================================
def generate_water(n_atoms_target, output_dir, n_frames=100, density=1.0, perturbation=0.25):
    # Adjust atoms to be divisible by 3 (H2O)
    if n_atoms_target % 3 != 0:
        n_atoms_actual = n_atoms_target - (n_atoms_target % 3)
    else:
        n_atoms_actual = n_atoms_target
        
    n_mols = int(n_atoms_actual / 3)
    if n_mols < 1: return

    # Filename is just the target number (e.g., "100.xyz")
    filename = f"{n_atoms_target}.xyz"
    filepath = os.path.join(output_dir, filename)
    
    # Geometry (TIP3P-like)
    bond = 0.9572
    angle = 104.52 * np.pi / 180.0
    rel_O  = np.array([0.0, 0.0, 0.0])
    rel_H1 = np.array([bond*np.sin(angle/2), bond*np.cos(angle/2), 0.0])
    rel_H2 = np.array([-bond*np.sin(angle/2), bond*np.cos(angle/2), 0.0])
    template = [("O", rel_O), ("H", rel_H1), ("H", rel_H2)]
    
    # Box Size
    molar_mass = 18.015
    vol_cm3 = (n_mols * molar_mass) / (6.022e23 * density)
    box_length = (vol_cm3 * 1e24) ** (1/3)
    
    # Lattice Grid
    k = int(np.ceil(n_mols**(1/3)))
    spacing = box_length / k
    base_mols = []
    
    count = 0
    for x in range(k):
        for y in range(k):
            for z in range(k):
                if count >= n_mols: break
                center = np.array([(x+0.5)*spacing, (y+0.5)*spacing, (z+0.5)*spacing])
                base_mols.append(center)
                count += 1

    # Write
    with open(filepath, 'w') as f:
        for frame in range(n_frames):
            current_atoms = []
            for center in base_mols:
                # Perturb Molecule Center
                shift = (np.random.rand(3) - 0.5) * 2 * perturbation
                mol_center = center + shift
                for atom_type, rel_pos in template:
                    pos = get_pbc_distance(mol_center + rel_pos, box_length)
                    current_atoms.append((atom_type, pos))
            write_frame(f, current_atoms, box_length, frame)
            
    print(f"   -> Wrote {filename} (Actual: {n_atoms_actual} atoms)")

# ==========================================
# 2. BENZENE GENERATOR
# ==========================================
def generate_benzene(n_atoms_target, output_dir, n_frames=100, density=0.88, perturbation=0.2):
    # Adjust atoms to be divisible by 12 (C6H6)
    if n_atoms_target % 12 != 0:
        n_atoms_actual = n_atoms_target - (n_atoms_target % 12)
    else:
        n_atoms_actual = n_atoms_target
        
    n_mols = int(n_atoms_actual / 12)
    if n_mols < 1: return

    # Filename is just the target number (e.g., "100.xyz")
    filename = f"{n_atoms_target}.xyz"
    filepath = os.path.join(output_dir, filename)
    
    # Geometry
    template = []
    for i in range(6):
        th = i * np.pi / 3
        template.append(("C", np.array([1.40*np.cos(th), 1.40*np.sin(th), 0.0])))
        template.append(("H", np.array([2.48*np.cos(th), 2.48*np.sin(th), 0.0])))

    # Box Size
    molar_mass = 78.11
    vol_cm3 = (n_mols * molar_mass) / (6.022e23 * density)
    box_length = (vol_cm3 * 1e24) ** (1/3)

    # Lattice Grid
    k = int(np.ceil(n_mols**(1/3)))
    spacing = box_length / k
    base_mols = []
    
    count = 0
    for x in range(k):
        for y in range(k):
            for z in range(k):
                if count >= n_mols: break
                center = np.array([(x+0.5)*spacing, (y+0.5)*spacing, (z+0.5)*spacing])
                base_mols.append(center)
                count += 1

    # Write
    with open(filepath, 'w') as f:
        for frame in range(n_frames):
            current_atoms = []
            for center in base_mols:
                shift = (np.random.rand(3) - 0.5) * 2 * perturbation
                mol_center = center + shift
                for atom_type, rel_pos in template:
                    pos = get_pbc_distance(mol_center + rel_pos, box_length)
                    current_atoms.append((atom_type, pos))
            write_frame(f, current_atoms, box_length, frame)

    print(f"   -> Wrote {filename} (Actual: {n_atoms_actual} atoms)")

# ==========================================
# 3. HEA GENERATOR (Al-Cu-Ag-Au-Ni-Pd-Pt)
# ==========================================
def generate_hea(n_atoms_target, output_dir, n_frames=100, perturbation=0.15):
    # Vegard's Law Setup
    elements = ['Al', 'Cu', 'Ag', 'Au', 'Ni', 'Pd', 'Pt']
    lats     = [4.05, 3.615, 4.09, 4.08, 3.52, 3.89, 3.92]
    a_mix = sum(lats) / len(lats)
    
    # Supercell Logic (4 atoms per cell)
    k = int(np.round((n_atoms_target / 4) ** (1/3)))
    if k < 1: k = 1
    n_atoms_actual = 4 * (k**3)
    
    # Filename is just the target number (e.g., "100.xyz")
    filename = f"{n_atoms_target}.xyz"
    filepath = os.path.join(output_dir, filename)
    
    box_length = k * a_mix
    
    # Build FCC Grid
    base_pos = []
    basis = np.array([[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]]) * a_mix
    
    for x in range(k):
        for y in range(k):
            for z in range(k):
                offset = np.array([x, y, z]) * a_mix
                for b in basis:
                    base_pos.append(offset + b)
    
    # Assign Elements Randomly
    atom_types = np.random.choice(elements, size=n_atoms_actual)
    
    # Write
    with open(filepath, 'w') as f:
        for frame in range(n_frames):
            current_atoms = []
            for i, pos in enumerate(base_pos):
                shift = (np.random.rand(3) - 0.5) * 2 * perturbation
                final_pos = get_pbc_distance(pos + shift, box_length)
                current_atoms.append((atom_types[i], final_pos))
            write_frame(f, current_atoms, box_length, frame)

    print(f"   -> Wrote {filename} (Actual: {n_atoms_actual} atoms)")

# ==========================================
# MAIN EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    
    # 1. Configuration
    root_dir = "boxes"
    targets = [100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 25000]
    systems = ["water", "benzene", "hea"]
    
    print(f"Starting generation for targets: {targets}")
    
    # 2. Iterate through Systems
    for sys_name in systems:
        # Create system folder (e.g., boxes/water)
        sys_path = os.path.join(root_dir, sys_name)
        os.makedirs(sys_path, exist_ok=True)
        
        print(f"\n--- Processing System: {sys_name.upper()} ---")
        
        # 3. Iterate through Targets
        for n in targets:
            # We no longer create a subfolder for 'n'.
            # We pass sys_path as the output directory.
            
            if sys_name == "water":
                generate_water(n, sys_path, n_frames=110)
            elif sys_name == "benzene":
                generate_benzene(n, sys_path, n_frames=110)
            elif sys_name == "hea":
                generate_hea(n, sys_path, n_frames=110)
                
    print("\nAll tasks completed successfully.")