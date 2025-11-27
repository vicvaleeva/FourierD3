from typing import List
from pathlib import Path

import torch
import numpy as np
from scipy.sparse.linalg import eigsh


# load C6 reference tensor (104, 104, 7, 7)
def load_c6ref(types: List) -> torch.Tensor:
    current_dir = Path(__file__).parent.resolve()
    c6ref = torch.load(current_dir / '..' / 'data' / 'reference-c6.pt', weights_only=True)
    return c6ref[types][:, types].permute(0, 2, 1, 3).reshape(len(types)*7, len(types)*7).numpy()

# compute maximum relative error between reference c6 and low-rank approximation
def maxrel_err(ref, approx) -> float:
    mask = np.where(ref == 0, 1.0, ref)
    return np.max(np.abs(ref-approx)*np.where(ref == 0, 0, 1/mask))

def decomp(types: List, c6tol: float):
    c6ref_mat = load_c6ref(types)
    k = 1
    eigs, eigvecs = eigsh(c6ref_mat, k=k)
    while maxrel_err(c6ref_mat, eigvecs @ np.diag(eigs) @ eigvecs.T)*100 >= c6tol:
        k += 1
        eigs, eigvecs = eigsh(c6ref_mat, k=k)
    err = maxrel_err(c6ref_mat, eigvecs @ np.diag(eigs) @ eigvecs.T)
    print(f'Using {k}-rank decomposition with maximum relative error: {err*100} %')   
    return torch.tensor(eigs, dtype=torch.float64), torch.tensor(eigvecs, dtype=torch.float64)

def load_sqrtQz(types: List, device) -> torch.Tensor:
    current_dir = Path(__file__).parent.resolve()
    return torch.load(current_dir / '..' / 'data' / 'sqrtQz.pt', weights_only=True)[types].to(device)

def load_rcov() -> torch.Tensor:
    current_dir = Path(__file__).parent.resolve()
    return torch.load(current_dir / '..' / 'data' / 'rcov.pt', weights_only=True)

def load_cnref() -> torch.Tensor:
    current_dir = Path(__file__).parent.resolve()
    return torch.load(current_dir / '..' / 'data' / 'cnref.pt', weights_only=True)