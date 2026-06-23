import torch

from torchpme.lib.kspace_filter import KSpaceFilter
from torchpme.lib.kvectors import _generate_kvectors

from fourierd3.utils import safe_inv_3x3


class KSpaceFilterD3(KSpaceFilter):
    """Reciprocal-space convolution for the DFT-D3 dispersion energy.

    Computes the k-space contribution to the D3 energy:

        E_k = -(1/(2*Omega)) * sum_r lambda_r
              * sum_{X,Y} sum_k G_{X,Y}(k) * S^r_X(k) * conj(S^r_Y(k))

    where G_{X,Y}(k) is the Green's function (Fourier transform of the
    BJ-damped potential) stored in `_kfilter`, and S^r_X(k) is the structure
    factor for rank-r component and species X.

    In the PME/SPME path, the structure factors are computed by spreading the
    per-atom C6 weights onto a mesh via `MeshInterpolatorD3.points_to_mesh`,
    then performing a 3D rFFT. The Green's function G is applied by matrix
    multiplication (coupling species X and Y) for each k-point, and the result
    is summed over k. The output is a vector of length n_rank, one dot product
    per eigenvalue lambda_r; the final energy is `dot(eigs, output) / (-2*Omega)`.
    """

    def forward(self, mesh_values: torch.Tensor) -> torch.Tensor:
        """Apply the Green's function in k-space and return the per-rank energy contributions.

        Args:
            mesh_values: C6-weighted charge density on the mesh, shape
                         (n_species, n_rank, nx, ny, nz).

        Returns:
            filter_hat: real tensor of shape (n_rank,); entry r is
                sum_{X,Y,k} G_{X,Y}(k) * S^r_X(k) * conj(S^r_Y(k)).
                Multiply by -1/(2*Omega) and dot with eigenvalues to get E_D3.
        """
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

        # Compute 3D rFFT to get structure factors S^r_X(k).
        # mesh_hat shape: (n_species, n_rank, nx, ny, nz//2+1)
        mesh_hat = torch.fft.rfftn(mesh_values, norm=self._fft_norm, dim=(2, 3, 4))

        # Flatten spatial dims and permute: (nk_flat, n_species, n_rank)
        # Each k-point is a (n_species x n_rank) matrix of structure factors.
        mesh_ready = mesh_hat.flatten(2).permute(2, 0, 1).contiguous()

        # For each k-point, compute sum_{X,Y} G_{X,Y}(k) * conj(S^r_Y(k)).
        # _kfilter has shape (nk_flat, n_species_X, n_species_Y) (after dealiasing).
        # bmm contracts over the Y (species) dimension.
        interm = torch.bmm(self._kfilter, mesh_ready.conj())  # (nk, n_species, n_rank)

        # Multiply by S^r_X(k) and sum over k and species to get per-rank totals.
        # Taking the real part discards the imaginary part which cancels by symmetry.
        filter_hat = (mesh_ready * interm).real.sum(dim=(0, 1))  # (n_rank,)

        return filter_hat

    @torch.jit.export
    def _prep_kvectors(self, cell, ns_mesh):
        """Update k-vectors and recompute the Green's function when the cell changes."""
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
            # Generate k-vectors on the PME mesh grid (not for Ewald, which uses explicit list)
            self._kvectors = _generate_kvectors(ns=self.ns_mesh, cell=self.cell, for_ewald=False)
            self._k_sq = torch.linalg.norm(self._kvectors, dim=-1)


class KSpaceFilterCN(KSpaceFilter):
    """Reciprocal-space convolution for the D4 coordination number field.

    Computes the potential field phi_{X}(r_i) at each atom position due to the
    CN pair potential, by convolving the species density with the Green's function
    G^{CN}_{AB}(k) = hat_f_{AB}(k) (from CNPotential.lr_from_k_sq) in k-space.

    The CN of atom i of species X is then:

        CN_i = sum_Y phi_{XY}(r_i) / Omega - f_{XX}(0)   (self-interaction subtracted)

    where phi is the result of this convolution (computed via reciprocal space).
    This avoids building a real-space neighbour list for the CN calculation entirely.
    """

    def forward(self, mesh_values: torch.Tensor) -> torch.Tensor:
        """Convolve the species density with the CN Green's function in k-space.

        Args:
            mesh_values: species density on the mesh, shape (n_species, nx, ny, nz).
                         Entry [X, ix, iy, iz] is the sum of interpolation weights
                         of all atoms of species X at mesh point (ix, iy, iz).

        Returns:
            phi_mesh: real-space potential field, shape (n_species, nx, ny, nz).
                      phi_mesh[X, ...] = sum_Y G^CN_{XY}(k) * rho_Y(k) (back-transformed).
                      Interpolated at atom positions by `mesh_to_points` to give CN values.
        """
        if mesh_values.dim() != 4:
            raise ValueError(
                "`mesh_values` needs to be a 4 dimensional tensor, got "
                f"{mesh_values.dim()}"
            )

        if mesh_values.device != self._kfilter.device:
            raise ValueError(
                "`mesh_values` and the k-space filter are on different devices, got "
                f"{mesh_values.device} and {self._kfilter.device}"
            )

        # 3D rFFT of the species density: rho_X(k), shape (n_species, nx, ny, nz//2+1)
        mesh_hat = torch.fft.rfftn(mesh_values, norm=self._fft_norm, dim=(1, 2, 3))
        # Flatten to (n_species, nk_flat)
        struc = mesh_hat.flatten(1)

        # Apply Green's function: phi_X(k) = sum_Y G^CN_{XY}(k) * rho_Y(k)
        # _kfilter shape: (nk_flat, n_species_X, n_species_Y)
        interm = torch.einsum('kab,bk->ak', self._kfilter, struc)  # (n_species, nk)

        # Back-transform to real space to get the potential field phi_X(r)
        phi_mesh = torch.fft.irfftn(
            interm.view(mesh_hat.shape),
            norm=self._ifft_norm,
            s=mesh_values.shape[-3:],
            dim=(1, 2, 3)
        )  # (n_species, nx, ny, nz)

        return phi_mesh

    @torch.jit.export
    def _prep_kvectors(self, cell, ns_mesh):
        """Update k-vectors and recompute the CN Green's function when the cell changes."""
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
