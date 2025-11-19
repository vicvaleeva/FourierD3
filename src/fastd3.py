from typing import Optional, List

import torch

class FastD3(torch.nn.Module):
    """
    Fast D3 calculator in the torch interface.
    Uses particle-mesh methods (PME and P3M) to perform fast summation
    of the ~inherently~ long-ranged damped D3 correction potential.
    The C^6AB are untangled using eigendecomposition and the method
    re-uses the neighborlist calculated by the underlying ML potential.
    
    :param elements: list containing atom types for eigendecomposition,
        must contain at least all the atom types present in the cell
    :param cell: Tensor containing vectors defining the periodic cell dimensions
    :param pbc: 3bBool tensor to verify pbc are activated in all directions
    :param fourierspacing: parameter controlling mesh spacing (in Angstrom), 
        biggest influence on accuracy
    :param c6tol: maximum allowed deviation for estimation of C6ref (in %), controls
        the rank of eigendecomposition approximation
    :param xc: string specifying the xc functional used to train
        the ML potential, needed for D3 parameters
        
    """
    
    def __init__(
        self,
        types: List,
        cell: torch.Tensor,
        pbc: Optional[torch.Tensor],
        fourierspacing: float = 1.2,
        c6tol: float = 0.1,
        xcfunctional: str = 'PBE'
    ):
        super().__init__()
        return 0
        