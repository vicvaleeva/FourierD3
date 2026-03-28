from pathlib import Path
from typing import List

import numpy as np
import torch
from scipy.sparse.linalg import eigsh

# load C6 reference tensor (104, 104, 7, 7)
def load_c6ref(types: List) -> torch.Tensor:
    current_dir = Path(__file__).parent.resolve()
    c6ref = torch.load(current_dir / '..' / '..' / 'data' / 'reference-c6.pt', weights_only=True)
    return c6ref[types][:, types].permute(0, 2, 1, 3).reshape(len(types)*7, len(types)*7).numpy()

# compute maximum relative error between reference c6 and low-rank approximation
def maxrel_err(ref, approx) -> float:
    mask = np.where(ref == 0, 1.0, ref)
    return np.max(np.abs(ref-approx)*np.where(ref == 0, 0, 1/mask))

def decomp(types: List, c6tol: float, verbose: bool, seed: int = 42, dtype=torch.float32):
    c6ref_mat = load_c6ref(types)
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(c6ref_mat.shape[0])
    k = 1
    eigs, eigvecs = eigsh(c6ref_mat, k=k, v0=v0)
    while maxrel_err(c6ref_mat, eigvecs @ np.diag(eigs) @ eigvecs.T)*100 >= c6tol:
        k += 1
        eigs, eigvecs = eigsh(c6ref_mat, k=k, v0=v0)
    err = maxrel_err(c6ref_mat, eigvecs @ np.diag(eigs) @ eigvecs.T)
    if verbose:
        print(f'Using {k}-rank decomposition with maximum relative error: {err*100} %')   
    return torch.tensor(eigs, dtype=dtype), torch.tensor(eigvecs, dtype=dtype)

def load_sqrtQz(types: List, device) -> torch.Tensor:
    current_dir = Path(__file__).parent.resolve()
    return torch.load(current_dir / '..' / '..' / 'data' / 'sqrtQz.pt', weights_only=True)[types].to(device)

def load_rcov() -> torch.Tensor:
    current_dir = Path(__file__).parent.resolve()
    return torch.load(current_dir / '..' / '..' /  'data' / 'rcov.pt', weights_only=True)

def load_cnref() -> torch.Tensor:
    current_dir = Path(__file__).parent.resolve()
    return torch.load(current_dir / '..' /  '..' / 'data' / 'cnref.pt', weights_only=True)

def safe_det_3x3(m: torch.Tensor) -> torch.Tensor:
    c00 = m[1,1]*m[2,2] - m[1,2]*m[2,1]
    c10 = m[1,2]*m[2,0] - m[1,0]*m[2,2]
    c20 = m[1,0]*m[2,1] - m[1,1]*m[2,0]
    return m[0,0]*c00 + m[0,1]*c10 + m[0,2]*c20

def safe_inv_3x3(m: torch.Tensor) -> torch.Tensor:
    c00 = m[1,1]*m[2,2] - m[1,2]*m[2,1]
    c01 = m[0,2]*m[2,1] - m[0,1]*m[2,2]
    c02 = m[0,1]*m[1,2] - m[0,2]*m[1,1]
    
    c10 = m[1,2]*m[2,0] - m[1,0]*m[2,2]
    c11 = m[0,0]*m[2,2] - m[0,2]*m[2,0]
    c12 = m[0,2]*m[1,0] - m[0,0]*m[1,2]
    
    c20 = m[1,0]*m[2,1] - m[1,1]*m[2,0]
    c21 = m[0,1]*m[2,0] - m[0,0]*m[2,1]
    c22 = m[0,0]*m[1,1] - m[0,1]*m[1,0]
    
    adj = torch.stack([
        torch.stack([c00, c01, c02]),
        torch.stack([c10, c11, c12]),
        torch.stack([c20, c21, c22])
    ])
    
    det = m[0,0]*c00 + m[0,1]*c10 + m[0,2]*c20
    return adj / det