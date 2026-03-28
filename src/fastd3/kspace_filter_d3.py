import torch

from torchpme.lib.kspace_filter import KSpaceFilter
from torchpme.lib.kvectors import _generate_kvectors

from fastd3.utils import safe_inv_3x3

class KSpaceFilterD3(KSpaceFilter):
    
    def forward(self, mesh_values: torch.Tensor) -> torch.Tensor:
        
        # mesh_values has (n_species, n_rank, nx, ny, nz) dimensions
        
        if mesh_values.dim() != 5:
            raise ValueError(
                "`mesh_values` needs to be a 5 dimensional tensor, got "
                f"{mesh_values.dim()}"
            )

        if mesh_values.device != self._kfilter.device:
            raise ValueError(
                "`mesh_values` and the k-space filter are on different devices, got "
                f"{mesh_values.device} and {self._kfilter.device}"
            ) 

        mesh_hat = torch.fft.rfftn(mesh_values, norm=self._fft_norm, dim=(2, 3, 4))
    
        mesh_ready = mesh_hat.flatten(2).permute(2, 0, 1).contiguous()
        
        interm = torch.bmm(self._kfilter, mesh_ready.conj())
        filter_hat = (mesh_ready * interm).real.sum(dim=(0, 1))
        
        return filter_hat
    
    @torch.jit.export
    def _prep_kvectors(self, cell, ns_mesh):
        if cell is not None:
            if cell.shape != (3, 3):
                raise ValueError(
                    f"cell of shape {list(cell.shape)} should be of shape (3, 3)"
                )
            self.cell = cell

        if ns_mesh is not None:
            if ns_mesh.shape != (3,):
                raise ValueError(
                    f"shape {list(ns_mesh.shape)} of `ns_mesh` has to be (3,)"
                )
            self.ns_mesh = ns_mesh

        if self.cell.device != self.ns_mesh.device:
            raise ValueError(
                "`cell` and `ns_mesh` are on different devices, got "
                f"{self.cell.device} and {self.ns_mesh.device}"
            )

        if cell is not None or ns_mesh is not None:
            self._kvectors = _generate_kvectors(ns=self.ns_mesh, cell=self.cell, for_ewald=False)
            self._k_sq = torch.linalg.norm(self._kvectors, dim=-1)
        