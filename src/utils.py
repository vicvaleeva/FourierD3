from typing import List
from warnings import catch_warnings, simplefilter

import torch
import numpy as np
from scipy.sparse.linalg import eigsh


# load C6 reference tensor (104, 104, 7 ,7 )
def load_c6ref(types: List) -> torch.Tensor:
    c6ref = torch.load('../data/reference-c6.pt', weights_only=True)
    return c6ref[types][:, types].permute(0, 2, 1, 3).reshape(len(types)*7, len(types)*7).numpy()

# compute maximum relative error between reference c6 and low-rank approximation
def maxrel_err(ref, approx) -> float:
    with catch_warnings:
        simplefilter('ignore')
        return np.max(np.abs(ref-approx)*np.where(ref == 0, 0, 1/ref))

def decomp(types: List, c6tol: float):
    c6ref_mat = load_c6ref(types)
    k = 1
    eigs, eigvecs = eigsh(c6ref_mat, k=k)
    while maxrel_err(c6ref_mat, eigvecs @ np.diag(eigs) @ eigvecs.T)*100 >= c6tol:
        k += 1
        eigs, eigvecs = eigsh(c6ref_mat, k=k)
    err = maxrel_err(c6ref_mat, eigvecs @ np.diag(eigs) @ eigvecs.T)
    print(f'Using {k}-rank decomposition with maximum relative error: {err*100} %')   
    return eigs, eigvecs
        

    