from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.sparse.linalg import eigsh
'''
# --- DATA GENERATION BLOCK ---
# load C6 reference tensor (104, 104, 7, 7)
def load_c6ref(types: List) -> torch.Tensor:
    current_dir = Path(__file__).parent.resolve()
    c6ref = torch.load(current_dir / '..' / 'data' / 'reference-c6.pt', weights_only=True)
    return c6ref[types][:, types].permute(0, 2, 1, 3).reshape(len(types)*7, len(types)*7).numpy()

# compute maximum relative error between reference c6 and low-rank approximation
def maxrel_err(ref, approx) -> float:
    mask = np.where(ref == 0, 1.0, ref)
    return np.max(np.abs(ref-approx)*np.where(ref == 0, 0, 1/mask))

def decomp(types: List, c6tol: float, seed: int = 42):
    c6ref_mat = load_c6ref(types)
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(c6ref_mat.shape[0])
    k = 1
    eigs, eigvecs = eigsh(c6ref_mat, k=k, v0=v0)
    while maxrel_err(c6ref_mat, eigvecs @ np.diag(eigs) @ eigvecs.T)*100 >= c6tol:
        k += 1
        eigs, eigvecs = eigsh(c6ref_mat, k=k, v0=v0)
    return k

data = {}
for n_types in range(1, 101):
    data[n_types] = []
    for it in range(100):
        samples = np.random.choice(np.arange(1, 104), size=n_types, replace=False)
        data[n_types].append(decomp(samples, 0.01))
        
    print(n_types, 'done')

mins = []
maxes = []
medians = []
for n_types in range(1, 101):
    mins.append(np.min(data[n_types]))
    medians.append(np.median(data[n_types]))
    maxes.append(np.max(data[n_types]))

# Updated to save all three arrays so the plot can use them
np.savez('decomp_data.npz', mins=mins, medians=medians, maxes=maxes)

'''
# --- PLOTTING BLOCK ---

# Load the generated data arrays
# (Make sure you run the generation block once with np.savez to create this file)
loaded_data = np.load('decomp_data.npz')
mins = loaded_data['mins']
medians = loaded_data['medians']
maxes = loaded_data['maxes']

x_axis = np.arange(1, 101)

# Set up the figure
fig, ax = plt.subplots(figsize=(4, 4))

# Use standard matplotlib 'tab10' blue
main_color = 'tab:blue'

# 1. Plot the shaded range (min to max)
ax.fill_between(x_axis, mins, maxes, color=main_color, alpha=0.25, edgecolor='none')

# 2. Plot the median line on top
ax.plot(x_axis, medians, color=main_color, linewidth=2.5)

# 3. Formatting and styling
ax.set_xlabel('# of species', fontsize=12, fontweight='medium')
ax.set_ylabel('Rank of Approximation', fontsize=12, fontweight='medium')

# Clean tick marks
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_yticks([0, 5, 10, 15, 20, 25])
ax.tick_params(axis='both', which='major', labelsize=10)

# Light horizontal grid for readability
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Remove top and right spines for a modern look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add a legend
ax.legend(loc='lower right', frameon=False)

# Ensure tight layout
plt.tight_layout()

# Save and display
plt.savefig('decomp.pdf', dpi=300, bbox_inches='tight')
plt.show()