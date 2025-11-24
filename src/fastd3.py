from typing import Optional, List
from utils import decomp
from torchpme.lib.kvectors import get_ns_mesh
from torchpme.lib.mesh_interpolator import MeshInterpolator
from torchpme.lib.kspace_filter import KSpaceFilter

from pair_pot import D3Potential

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
    :param mesh_spacing: parameter controlling mesh spacing (in Angstrom), 
        biggest influence on accuracy
    :param c6tol: maximum relative error for estimation of C6ref (in %), controls
        the rank of eigendecomposition approximation
    :param xcfunc: string specifying the xc functional used to train
        the ML potential, needed for D3 parameters
        
    """
    
    def __init__(
        self,
        types: List,
        cell: torch.Tensor,
        pbc: Optional[torch.Tensor],
        mesh_spacing: float = 1.2,
        c6tol: float = 0.01,
        xcfunc: str = 'pbe',
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        method: str = 'pme',
        interpolation_nodes: int = 4
    ) -> None:
        super().__init__()
        
        if pbc is not None:
            assert pbc.all(), "particle-mesh only supports 3d pbc, if you have 2d pbc, please make sure there's plenty of empty space in the third direction"

        print("Assuming 3D PBC are satisfied")
        self.device = device
        self.xcfunc = xcfunc
        self.volume = torch.abs(torch.det(cell)).to(self.device)
        
        self.eigs, self.eigvecs = decomp(types, c6tol)
        self.eigs.to(device)
        self.eigvecs.to(device)
        
        # implement automatic choice from xc functional !
        # these are just for pbe
        params = torch.Tensor([1.0, 0.7875, 0.4289, 4.4407], device=device)
        
        self.potential = D3Potential(self.types, params, device)
        
        ns_mesh = get_ns_mesh(cell, mesh_spacing)
        
        self.mesh_interpolator = MeshInterpolator(
            cell=cell,
            ns_mesh=ns_mesh,
            interpolation_nodes=interpolation_nodes
        )
        
        self.kspace_filter = KSpaceFilter(
            cell=cell,
            ns_mesh=ns_mesh,
            kernel = self.potential,
            fft_norm="backward",
            ifft_norm="forward"
        )
        
        
        
        
        

        
        