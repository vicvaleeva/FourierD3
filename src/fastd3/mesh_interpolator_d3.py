import torch
from typing import Optional
from torchpme.lib.mesh_interpolator import MeshInterpolator

from fastd3.utils import safe_inv_3x3


class MeshInterpolatorD3(MeshInterpolator):
    """Mesh interpolator for the D3 multi-channel charge density.

    Extends the base `MeshInterpolator` from torchpme to handle the
    (n_species, n_rank) channel structure of the D3 C6 weights.

    In the SPME/PME approach, the structure factor is approximated by spreading
    atom-centered weights onto an equispaced mesh:

        S^r_X(k) ≈ tilde{S}^r_X(k) = sum_{x in M} [sum_{i in X} C^6_{r,i} * W_i(x)] * e^{ik.x}

    where W_i(x) are B-spline (SPME) or Lagrange (PME) interpolation weights
    evaluated at the mesh points nearest to atom i. The mesh charge density
    rho^r_X[ix, iy, iz] = sum_i C^6_{r,i} * W_i(ix, iy, iz) is computed by
    `points_to_mesh`, and the FFT of this density gives the approximate structure factor.

    The adjoint operation `mesh_to_points` interpolates a potential field from
    the mesh back to atom positions; this is used for the D4 CN calculation,
    where the CN potential field is computed in k-space and must be evaluated at
    each atom's position.
    """

    @torch.jit.export
    def update(
        self,
        cell: Optional[torch.Tensor] = None,
        ns_mesh: Optional[torch.Tensor] = None,
    ) -> None:
        """Update the cell and/or mesh resolution, recomputing the inverse cell.

        Must be called whenever the simulation cell changes (e.g., during NPT MD).
        The inverse cell is needed to convert Cartesian positions to fractional
        coordinates for the B-spline weight computation.

        Args:
            cell:    (3, 3) tensor; cell[i] is the i-th lattice vector.
            ns_mesh: (3,) tensor; number of mesh points along each axis.
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

            # Recompute the inverse cell for fractional coordinate conversion.
            # Uses the explicit cofactor formula (safe_inv_3x3) for TorchScript compatibility.
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
        """Spread per-atom C6 weights onto the mesh (particle-to-mesh step).

        For each atom i, distributes its C6 weight C^6_{r,i} across the nearest
        mesh points using precomputed B-spline or Lagrange interpolation weights
        (stored in `self.interpolation_weights`, `x_indices`, etc., computed by
        `compute_weights` in the parent class). The result is the mesh charge
        density rho^r_X[ix, iy, iz].

        Args:
            particle_weights: C6 decomposition coefficients, shape (n_atoms, n_species, n_rank).
                              Entry [i, X, r] = C^6_{r,i} * delta(species_i, X).
                              (Most entries are zero; only atom i's species slot is nonzero.)
            dtype: dtype for the output mesh.

        Returns:
            rho_mesh: shape (n_species, n_rank, nx, ny, nz).
                      rho_mesh[X, r, ix, iy, iz] = sum_i C^6_{r,i} * W_i(ix, iy, iz) * delta(species_i, X).
        """
        if particle_weights.device != self._device:
            raise ValueError(f"Device mismatch: {particle_weights.device} vs {self._device}")

        if particle_weights.dim() != 3:
            raise ValueError("particle_weights must be (n_atoms, n_species, n_rank)")

        n_atoms, n_species, n_rank = particle_weights.shape
        n_channels = n_species * n_rank
        nx, ny, nz = int(self.ns_mesh[0]), int(self.ns_mesh[1]), int(self.ns_mesh[2])

        # Retrieve precomputed 1D interpolation weights along each axis.
        # w_x[i, p] = B-spline weight for atom i at its p-th node in the x-direction.
        w_x = self.interpolation_weights[self.x_shifts, :, 0]
        w_y = self.interpolation_weights[self.y_shifts, :, 1]
        w_z = self.interpolation_weights[self.z_shifts, :, 2]

        # 3D interpolation weight for each (atom, node) pair: product of 1D weights.
        # w_geo shape: (n_atoms, n_nodes^3), where n_nodes = interpolation_nodes^3.
        w_geo = (w_x * w_y * w_z).t()

        # Mesh point indices for each (atom, node) triple, flattened to a linear index.
        idx_x = self.x_indices.t()
        idx_y = self.y_indices.t()
        idx_z = self.z_indices.t()
        linear_indices = (idx_z + nz * (idx_y + ny * idx_x)).flatten()  # (n_atoms * n_nodes,)

        # Flatten the (n_species, n_rank) channel dimension for scatter-add efficiency.
        flat_particles = particle_weights.reshape(n_atoms, n_channels)  # (n_atoms, n_channels)

        # values_to_add[i*n_nodes + p, c] = flat_particles[i, c] * w_geo[i, p]
        values_to_add = (
            flat_particles.unsqueeze(1) * w_geo.unsqueeze(2)
        ).flatten(0, 1)  # (n_atoms * n_nodes, n_channels)

        # Accumulate contributions into the flattened mesh.
        rho_flat_accum = torch.zeros(
            (nx * ny * nz, n_channels),
            dtype=dtype,
            device=self._device
        )
        rho_flat_accum.index_add_(0, linear_indices, values_to_add)

        # Reshape to (n_species, n_rank, nx, ny, nz) for the FFT in KSpaceFilterD3.
        rho_mesh = rho_flat_accum.t().contiguous().view(n_species, n_rank, nx, ny, nz)

        return rho_mesh

    @torch.jit.export
    def mesh_to_points(self, mesh_values: torch.Tensor, dtype) -> torch.Tensor:
        """Interpolate a mesh field back to atom positions (mesh-to-particle step).

        This is the adjoint of `points_to_mesh`: for each atom i, it computes a
        weighted sum of the mesh field over the atom's interpolation nodes. Used
        to evaluate the CN potential field phi_X(r_i) at each atom position after
        the k-space convolution in `KSpaceFilterCN`.

        Args:
            mesh_values: shape (n_species, n_rank, nx, ny, nz). Typically the
                         real-space potential field obtained by back-transforming
                         G^CN(k) * rho(k).
            dtype: output dtype.

        Returns:
            output: shape (n_atoms, n_species, n_rank).
                    output[i, X, r] = sum_{nodes p} W_i(p) * mesh_values[X, r, ix_p, iy_p, iz_p].
        """
        if mesh_values.dim() != 5:
            raise ValueError("mesh_values must be (n_species, n_rank, nx, ny, nz)")

        n_species, n_rank, nx, ny, nz = mesh_values.shape
        n_channels = n_species * n_rank

        # Same interpolation weights as in points_to_mesh
        w_x = self.interpolation_weights[self.x_shifts, :, 0]
        w_y = self.interpolation_weights[self.y_shifts, :, 1]
        w_z = self.interpolation_weights[self.z_shifts, :, 2]
        w_geo = (w_x * w_y * w_z).t()  # (n_atoms, n_nodes)

        idx_x = self.x_indices.t()
        idx_y = self.y_indices.t()
        idx_z = self.z_indices.t()
        linear_indices = (idx_z + nz * (idx_y + ny * idx_x)).flatten()  # (n_atoms * n_nodes,)

        n_atoms = w_geo.shape[0]
        n_nodes = w_geo.shape[1]

        # Flatten to (nx*ny*nz, n_channels) to gather by linear index
        mesh_flat = mesh_values.reshape(n_channels, nx * ny * nz).t()

        # Gather mesh values at each (atom, node) location
        gathered = mesh_flat[linear_indices, :]           # (n_atoms * n_nodes, n_channels)
        gathered = gathered.view(n_atoms, n_nodes, n_channels)

        # Weighted sum over nodes: adjoint of the scatter-add in points_to_mesh
        output = (w_geo.unsqueeze(2) * gathered).sum(dim=1)  # (n_atoms, n_channels)

        return output.view(n_atoms, n_species, n_rank)
