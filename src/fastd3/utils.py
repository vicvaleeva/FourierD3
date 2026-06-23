from pathlib import Path
from typing import List

import numpy as np
import torch
from scipy.sparse.linalg import eigsh


def load_c6ref(types: List) -> torch.Tensor:
    """Load the C6^ref reference tensor and extract the block for the given species.

    The full tensor has shape (104, 104, 7, 7): for each pair of elements (up to
    element 104) and each pair of reference coordination environments (up to 7 per
    element), it stores the reference C6 dispersion coefficient C6^ref_{Z_i Z_j}
    (theta^ref_p_{Z_i}, theta^ref_q_{Z_j}) from the Grimme et al. DFT-D3 dataset
    The block is reindexed to the unique species present, permuted, and reshaped
    into a symmetric (n_species*7, n_species*7) matrix M, ready for the
    eigendecomposition in `decomp`.
    """
    current_dir = Path(__file__).parent.resolve()
    c6ref = torch.load(current_dir / '..' / '..' / 'data' / 'reference-c6.pt', weights_only=True)
    return c6ref[types][:, types].permute(0, 2, 1, 3).reshape(len(types)*7, len(types)*7).numpy()


def maxrel_err(ref, approx) -> float:
    """Maximum relative error between the reference C6 tensor and its low-rank approximation.

    Zero entries in `ref` are excluded from the relative error (they contribute 0).
    Used as the convergence criterion in `decomp`.
    """
    mask = np.where(ref == 0, 1.0, ref)
    return np.max(np.abs(ref-approx)*np.where(ref == 0, 0, 1/mask))


def decomp(types: List, c6tol: float, verbose: bool, seed: int = 42, dtype=torch.float32):
    """Compute the low-rank eigendecomposition of the C6^ref block matrix M.

    M is the symmetric (n_species*7, n_species*7) matrix built from the DFT-D3
    reference C6 coefficients for the species present in the system. We seek the
    smallest rank r such that the rank-r approximation

        M ≈ V diag(lambda) V^T,    V in R^{n_species*7 x r}

    has maximum relative error below `c6tol` percent. The Lanczos
    algorithm (eigsh) is used because M is large and sparse.

    Returns:
        eigs:    eigenvalues lambda_r, shape (r,)
        eigvecs: eigenvectors V, shape (n_species*7, r); rows are indexed by
                 (species, reference CN index) pairs and will later be reshaped
                 to (n_species, 7, r) to build the per-atom C6 weights.
    """
    # The decomposition is an offline preprocessing step, so always compute it in
    # full fp64 precision (regardless of the requested runtime `dtype`) to keep the
    # eigensolver well-conditioned, then cast the result to `dtype` at the end.
    c6ref_mat = load_c6ref(types).astype(np.float64)
    n = c6ref_mat.shape[0]
    # eigsh can return at most n-1 eigenpairs; cap the rank there to avoid an
    # infinite loop if the requested tolerance can never be reached.
    max_rank = n - 1
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(n).astype(np.float64)
    k = 1
    eigs, eigvecs = eigsh(c6ref_mat, k=k, v0=v0)
    while maxrel_err(c6ref_mat, eigvecs @ np.diag(eigs) @ eigvecs.T)*100 >= c6tol and k < max_rank:
        k += 1
        eigs, eigvecs = eigsh(c6ref_mat, k=k, v0=v0)
    err = maxrel_err(c6ref_mat, eigvecs @ np.diag(eigs) @ eigvecs.T)
    if verbose:
        print(f'Using {k}-rank decomposition with maximum relative error: {err*100} %')
        if k == max_rank and err*100 >= c6tol:
            print(f'Warning: reached maximum rank {max_rank} without meeting tolerance {c6tol} %')
    return torch.tensor(eigs, dtype=dtype), torch.tensor(eigvecs, dtype=dtype)


def load_sqrtQz(types: List, device, dtype=torch.float32) -> torch.Tensor:
    """Load sqrt(Q_Z) for the given species, used in the C8 recursive relation.

    Q_Z = (1/2) * sqrt(Z) * <r^4>_Z / <r^2>_Z  are element-specific quadrupole
    moment ratios derived from atomic densities. The C8 coefficients are then
    C8_{ij} = 3 * C6_{ij} * sqrt(Q_{Z_i} * Q_{Z_j}).
    """
    current_dir = Path(__file__).parent.resolve()
    return torch.load(current_dir / '..' / '..' / 'data' / 'sqrtQz.pt', weights_only=True)[types].to(device=device, dtype=dtype)


def load_rcov() -> torch.Tensor:
    """Load covalent radii R^cov_Z (Pyykko et al.) for all elements.

    Used in the smooth coordination number function: the pairwise covalent radius
    sum R^cov_{ij} = R^cov_i + R^cov_j sets the
    distance scale at which two atoms are considered bonded.
    """
    current_dir = Path(__file__).parent.resolve()
    return torch.load(current_dir / '..' / '..' /  'data' / 'rcov.pt', weights_only=True)


def load_cnref() -> torch.Tensor:
    """Load the reference coordination numbers theta^ref_{Z,p} for all elements.

    Shape: (104, 7). Each element Z has up to 7 reference coordination environments
    used to interpolate the pairwise C6 coefficients. Entries equal to -1
    indicate unused reference environments and are masked during softmax.
    """
    current_dir = Path(__file__).parent.resolve()
    return torch.load(current_dir / '..' /  '..' / 'data' / 'cnref.pt', weights_only=True)


def load_en() -> torch.Tensor:
    """Load Pauling electronegativities EN_Z for all elements.

    Used in the D4 coordination number pair weight delta_{AB}: the electronegativity
    difference |EN_A - EN_B| controls how strongly
    pairs of unlike elements contribute to each other's coordination number.
    """
    current_dir = Path(__file__).parent.resolve()
    return torch.load(current_dir / '..' / '..' / 'data' / 'en.pt', weights_only=True)


def load_rcov_cn() -> torch.Tensor:
    """Load covalent radii R^cov_Z used specifically for the D4 CN pair potential.

    These may differ slightly from the radii in `load_rcov` (which come from Pyykko
    et al.) because the D4 CN function uses a different reference dataset.
    """
    current_dir = Path(__file__).parent.resolve()
    return torch.load(current_dir / '..' / '..' /  'data' / 'rcov_cn.pt', weights_only=True)


def safe_det_3x3(m: torch.Tensor) -> torch.Tensor:
    """Compute the determinant of a 3x3 matrix via explicit cofactor expansion.

    Avoids calling torch.linalg.det, which is not supported under TorchScript
    and may not be differentiable in all backends. Used to compute the cell volume Omega.
    """
    c00 = m[1,1]*m[2,2] - m[1,2]*m[2,1]
    c10 = m[1,2]*m[2,0] - m[1,0]*m[2,2]
    c20 = m[1,0]*m[2,1] - m[1,1]*m[2,0]
    return m[0,0]*c00 + m[0,1]*c10 + m[0,2]*c20


def safe_inv_3x3(m: torch.Tensor) -> torch.Tensor:
    """Compute the inverse of a 3x3 matrix via explicit adjugate/determinant formula.

    Avoids torch.linalg.inv for TorchScript compatibility. Used to convert between
    Cartesian and fractional coordinates in the mesh interpolator.
    """
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
