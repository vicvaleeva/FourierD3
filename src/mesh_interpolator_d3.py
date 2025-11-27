from torchpme.lib.mesh_interpolator import MeshInterpolator

import torch

class MeshInterpolatorD3(MeshInterpolator):
    
    def points_to_mesh(self, particle_weights):
        if particle_weights.device != self._device:
            raise ValueError(f"Device mismatch: {particle_weights.device} vs {self._device}")
        
        if particle_weights.dim() != 3:
            raise ValueError("particle_weights must be (n_atoms, n_species, n_rank)")

        n_atoms, n_species, n_rank = particle_weights.shape
        n_channels = n_species * n_rank
        nx, ny, nz = int(self.ns_mesh[0]), int(self.ns_mesh[1]), int(self.ns_mesh[2])
        
        flat_weights = particle_weights.reshape(n_atoms, n_channels)

        rho_mesh_flat = torch.zeros(
            (n_channels, nx, ny, nz), 
            dtype=self._dtype, 
            device=self._device
        )
        
        for a in range(n_channels):
            rho_mesh_flat[a].index_put_(
                (self.x_indices, self.y_indices, self.z_indices),
                (
                    flat_weights[:, a]
                    * self.interpolation_weights[self.x_shifts, :, 0]
                    * self.interpolation_weights[self.y_shifts, :, 1]
                    * self.interpolation_weights[self.z_shifts, :, 2]
                ),
                accumulate=True,
            )

        # (n_species, n_rank, nx, ny, nz)
        rho_mesh = rho_mesh_flat.view(n_species, n_rank, nx, ny, nz)
        return rho_mesh
    
    def mesh_to_points(self, mesh_vals: torch.Tensor) -> torch.Tensor:
        
        if mesh_vals.device != self._device:
            raise ValueError(f"Device mismatch: {mesh_vals.device} vs {self._device}")
        
        if mesh_vals.dim() != 5:
            raise ValueError(
                f"`mesh_vals` must be 5D (n_species, n_rank, nx, ny, nz), got {mesh_vals.dim()}"
            )
        
        n_species, n_rank, nx, ny, nz = mesh_vals.shape
        n_channels = n_species * n_rank
        
        mesh_vals_flat = mesh_vals.view(n_channels, nx, ny, nz)

        flat_results = (
            (
                mesh_vals_flat[:, self.x_indices, self.y_indices, self.z_indices]
                * self.interpolation_weights[self.x_shifts, :, 0]
                * self.interpolation_weights[self.y_shifts, :, 1]
                * self.interpolation_weights[self.z_shifts, :, 2]
            )
            .sum(dim=1)
            .T 
        ) # (n_atoms, n_channels)
        
        n_atoms = flat_results.shape[0]
        return flat_results.view(n_atoms, n_species, n_rank)
        
        