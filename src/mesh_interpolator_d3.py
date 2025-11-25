from torchpme.lib.mesh_interpolator import MeshInterpolator

import torch

class MeshInterpolatorD3(MeshInterpolator):
    
    def points_to_mesh(self, particle_weights):
        if particle_weights.device != self._device:
            raise ValueError(f"Device mismatch: {particle_weights.device} vs {self._device}")
        
        if particle_weights.dim() != 3:
            raise ValueError("particle_weights must be (n_points, n_species, n_rank)")

        _, n_species, n_rank = particle_weights.shape
        nx, ny, nz = int(self.ns_mesh[0]), int(self.ns_mesh[1]), int(self.ns_mesh[2])

        spatial_weights = (
            self.interpolation_weights[self.x_shifts, :, 0]
            * self.interpolation_weights[self.y_shifts, :, 1]
            * self.interpolation_weights[self.z_shifts, :, 2]
        )

        values_to_scatter = (
            particle_weights.unsqueeze(1) * spatial_weights.unsqueeze(-1).unsqueeze(-1)
        )
        
        values_flat = values_to_scatter.view(-1, n_species, n_rank)

        idx_x = self.x_indices.view(-1)
        idx_y = self.y_indices.view(-1)
        idx_z = self.z_indices.view(-1)

        rho_mesh_permuted = torch.zeros(
            (nx, ny, nz, n_species, n_rank), 
            dtype=self._dtype, 
            device=self._device
        )

        rho_mesh_permuted.index_put_(
            (idx_x, idx_y, idx_z),
            values_flat,
            accumulate=True
        )

        # (n_species, n_rank, nx, ny, nz)
        return rho_mesh_permuted.permute(3, 4, 0, 1, 2)
    
    def mesh_to_points(self, mesh_vals: torch.Tensor, species: torch.Tensor) -> torch.Tensor:
        if mesh_vals.dim() != 5:
            raise ValueError(
                f"`mesh_vals` must be 5D (n_species, n_rank, nx, ny, nz), got {mesh_vals.dim()}"
            )
        
        if species.dim() != 1:
            raise ValueError("`species` must be 1D (n_points,)")
        
        spatial_weights = (
            self.interpolation_weights[self.x_shifts, :, 0]
            * self.interpolation_weights[self.y_shifts, :, 1]
            * self.interpolation_weights[self.z_shifts, :, 2]
        )

        n_interp = self.x_indices.shape[1]
        idx_species = species.view(-1, 1).expand(-1, n_interp)

        grid_values = mesh_vals[
            idx_species,
            :,
            self.x_indices,
            self.y_indices,
            self.z_indices
        ]

        interpolated_values = (grid_values * spatial_weights.unsqueeze(-1)).sum(dim=1)
        # (n_atoms, n_rank)
        return interpolated_values
