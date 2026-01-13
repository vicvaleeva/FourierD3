import torch

from torchpme.lib.mesh_interpolator import MeshInterpolator

class MeshInterpolatorD3(MeshInterpolator):
    
    def _compute_1d_weights(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == 'Euler':
            return self._compute_1d_weights_Euler(x)
        if self.method == "Lagrange":
            return self._compute_1d_weights_Lagrange(x)
        raise ValueError("Only `method` `Lagrange` and `Euler` are allowed")
    
    @torch.jit.export
    def _compute_1d_weights_Euler(self, x: torch.Tensor) -> torch.Tensor:
        return self._compute_1d_weights_P3M(x)
    
    @torch.jit.export
    def points_to_mesh(self, particle_weights: torch.Tensor) -> torch.Tensor:
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
            dtype=self._dtype, 
            device=self._device
        )
        
        rho_flat_accum.index_add_(0, linear_indices, values_to_add)

        rho_mesh = rho_flat_accum.t().contiguous().view(n_species, n_rank, nx, ny, nz)
        
        return rho_mesh