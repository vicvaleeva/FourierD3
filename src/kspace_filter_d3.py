from torchpme.lib.kspace_filter import KSpaceFilter, KSpaceKernel

import torch

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
            
        mesh_hat = torch.fft.rfftn(mesh_values, norm=self._fft_norm, dims=(2, 3, 4))
        
        if mesh_hat.shape[-3:] != self._kfilter.shape[-3:]:
            raise ValueError(
                "The particle weight mesh is inconsistent with the k-space grid."
            )
        
        filter_hat = torch.einsum('ikxyz,kjxyz->ijxyz', self._kfilter, mesh_hat)
        
        result = torch.fft.irfftn(filter_hat, norm=self._ifft_norm, dim=(2, 3, 4), s=mesh_hat.shape)
        
        if torch.isnan(result).any():
            raise ValueError(
                "NaNs detected in the k-space filter result. This are probably caused "
                "by an unsuitable `mesh_spacing`, resulting in a problematic grid of "
                f"shape: {list(mesh_values.shape)}. Try adjsuting the grid by using a "
                "different `mesh_spacing` value."
            )
            
        return result
        