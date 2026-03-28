import torch
from typing import Optional
from torchpme.lib.mesh_interpolator import MeshInterpolator

from fastd3.utils import safe_inv_3x3

class MeshInterpolatorD3(MeshInterpolator):
    
    @torch.jit.export
    def update(
        self,
        cell: Optional[torch.Tensor] = None,
        ns_mesh: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Update buffers and derived attributes of the instance.

        Call this to reuse a ``MeshInterpolator`` object when the ``cell`` parameters or
        the mesh resolution changes. If neither ``cell`` nor ``ns_mesh`` are passed
        there is nothing to be done.

        :param cell: torch.tensor of shape ``(3, 3)``, where ``cell[i]`` is the i-th basis
            vector of the unit cell
        :param ns_mesh: toch.tensor of shape ``(3,)`` Number of mesh points to use along
            each of the three axes
        """
        if cell is not None:
            if cell.shape != (3, 3):
                raise ValueError(
                    f"cell of shape {list(cell.shape)} should be of shape (3, 3)"
                )
            self.cell = cell
            self.inverse_cell = cell.clone()
            self._dtype = cell.dtype
            self._device = cell.device

            self.inverse_cell = safe_inv_3x3(cell)

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
    
    @torch.jit.export
    def points_to_mesh(self, particle_weights: torch.Tensor, dtype) -> torch.Tensor:
        if particle_weights.device != self._device:
            raise ValueError(f"Device mismatch: {particle_weights.device} vs {self._device}")
        
        if particle_weights.dim() != 3:
            raise ValueError("particle_weights must be (n_atoms, n_species, n_rank)")

        n_atoms, n_species, n_rank = particle_weights.shape
        n_channels = n_species * n_rank
        nx, ny, nz = int(self.ns_mesh[0]), int(self.ns_mesh[1]), int(self.ns_mesh[2])

        w_x = self.interpolation_weights[self.x_shifts, :, 0]
        w_y = self.interpolation_weights[self.y_shifts, :, 1]
        w_z = self.interpolation_weights[self.z_shifts, :, 2]
        
        w_geo = (w_x * w_y * w_z).t() 

        idx_x = self.x_indices.t()
        idx_y = self.y_indices.t()
        idx_z = self.z_indices.t()
        
        linear_indices = (idx_z + nz * (idx_y + ny * idx_x)).flatten()

        flat_particles = particle_weights.reshape(n_atoms, n_channels)
        
        values_to_add = (
            flat_particles.unsqueeze(1) * w_geo.unsqueeze(2)
        ).flatten(0, 1)

        rho_flat_accum = torch.zeros(
            (nx * ny * nz, n_channels), 
            dtype=dtype, 
            device=self._device
        )
        
        rho_flat_accum.index_add_(0, linear_indices, values_to_add)

        rho_mesh = rho_flat_accum.t().contiguous().view(n_species, n_rank, nx, ny, nz)
        
        return rho_mesh