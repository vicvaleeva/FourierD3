from typing import List

import torch
from torchpme.potentials import Potential

from fastd3.utils import load_sqrtQz, load_en, load_rcov_cn


class D3Potential(Potential):
    """Reciprocal-space Green's function for the BJ-damped DFT-D3 dispersion potential.

    The real-space DFT-D3 energy per rank-r term is a pairwise sum:

        phi_{X,Y}(r) = s6 * f6^damp(r, X, Y) / r^6
                     + 3*sqrt(Q_X*Q_Y) * s8 * f8^damp(r, X, Y) / r^8

    with Becke-Johnson (BJ) damping:

        f_n^damp(r, X, Y) = s_n * r^n / (r^n + R0(X,Y)^n),
        R0(X, Y) = a1 * sqrt(3 * Q_X * Q_Y) + a2

    This class provides `lr_from_k_sq`, which returns the 3D spherical Fourier
    transform of phi_{X,Y}(r) evaluated on the reciprocal-space grid. This
    Fourier transform is the Green's function G_{X,Y}(k).

    The FT is derived analytically via contour integration. The result involves
    exponential-trigonometric combinations that are evaluated in two branches
    to avoid numerical cancellation near k=0 (where a Taylor expansion is used).

    `selfcont` stores the k->0 (r->0) limit of the potential needed for the
    self-interaction correction V_self.
    """

    def __init__(
        self,
        species: List,
        params: torch.tensor,
        device,
        method: str,
        order: int = None,
        dtype = torch.float32
    ):
        super().__init__()
        self.device = device
        self.order = order
        self.species = species
        self.params = params.to(device)  # [s6, s8, a1, a2] for BJ damping
        self.method = method
        self.dtype = dtype

        # sqrt(Q_Z) for each unique species, used in R0 and C8
        self.sqrtQz = load_sqrtQz(species, device=device)
        # QzQz[X, Y] = sqrt(Q_X) * sqrt(Q_Y), shape (n_species, n_species)
        self.QzQz = torch.outer(self.sqrtQz, self.sqrtQz)

        # Shape for broadcasting over (n_species_X, n_species_Y, k-grid...)
        if self.method in ('pme', 'spme'):
            self.view_shape = (len(species), len(species), 1, 1, 1)
        else:
            self.view_shape = (len(species), len(species), 1)

        # R0(X, Y) = a1 * sqrt(3 * Q_X * Q_Y) + a2; shape broadcasts over k
        self.Rab = (params[2]*torch.sqrt(3*self.QzQz) + params[3]).view(*self.view_shape)
        self.QzQz_view = self.QzQz.view(*self.view_shape)

        # Precompute powers of R0 needed in the analytical FT expressions
        self.Rab2 = torch.pow(self.Rab, 2)
        self.Rab3 = torch.pow(self.Rab, 3)
        self.Rab4 = self.Rab * self.Rab3
        self.Rab5 = self.Rab * self.Rab4
        self.Rab6 = torch.pow(self.Rab3, 2)
        self.Rab8 = self.Rab3 * self.Rab5

        # Constants used in the analytical FT formulas
        self.pi = torch.tensor(torch.pi, device=device)
        self.pi2 = self.pi**2
        self.sq2 = torch.sqrt(torch.tensor(2.0, device=device))
        self.sq3 = torch.sqrt(torch.tensor(3.0, device=device))
        self.sin8 = torch.sin(self.pi / 8)   # sin(pi/8), appears in the C8 FT
        self.cos8 = torch.cos(self.pi / 8)   # cos(pi/8), appears in the C8 FT
        self.thresh = torch.tensor(1e-15, device=device)  # threshold for small-k branch

        # Self-interaction V_self: limit of phi_{X,X}(r) as r -> 0+.
        # For BJ damping, phi_{X,X}(r) -> s6/R0^6 + 3*s8*Q_X/R0^8 as r->0.
        diag6i = 1 / torch.diagonal(self.Rab6)   # 1/R0_{X,X}^6
        diag8i = 1 / torch.diagonal(self.Rab8)   # 1/R0_{X,X}^8
        diagQz = torch.diagonal(self.QzQz)        # Q_X
        self.selfcont = (self.params[0] * diag6i + 3 * self.params[1] * diagQz * diag8i).view(-1, 1)

        # Precomputed coefficient groups for the C6 FT (1/r^6 with BJ damping).
        # The analytical 3D spherical FT of s6 * r^6/(r^6 + R0^6) / r^6 is derived
        # via partial fractions over the 6th roots of -R0^6 and contour integration.
        # Small-k Taylor: hatf6(k) ≈ (2*pi^2)/(3*R0^3) - (2*pi^2*R0)/(9) * k^2
        self.ft6_small_c1 = (2 * self.pi2) / (3 * self.Rab3)
        self.ft6_small_c2 = (2 * self.pi2 * self.Rab) / 9
        # Large-k prefactor: (2*pi^2)/(3*R0^4 * k)
        self.pre6_c = (2 * self.pi2) / (3 * self.Rab4)

        # Precomputed coefficient groups for the C8 FT (1/r^8 with BJ damping).
        # Small-k Taylor: hatf8(k) ≈ (pi^2*sqrt(2)*sin(pi/8))/R0^5 - ... * k^2
        self.ft8_small_c1 = (self.pi2 * self.sq2 * self.sin8) / self.Rab5
        self.ft8_small_c2 = self.ft8_small_c1 * self.Rab2 / 6
        # Large-k prefactor: -pi^2 / R0^6
        self.pre8_c = - (self.pi2) / self.Rab6

        # Cache for the SPME aliasing correction (recomputed only if mesh shape changes)
        self._cached_mesh_shape = None
        self._cached_sc = None
        self._cached_weights = None


    def lr_from_k_sq(self, _k: torch.tensor) -> torch.tensor:
        """Evaluate the Green's function G_{X,Y}(k) for each species pair.

        Returns the 3D spherical Fourier transform of phi_{X,Y}(r) at each
        k-vector magnitude in `_k`. This enters the reciprocal-space energy as:

            E_D3 = V_self - (1/(2*Omega)) * sum_r lambda_r
                   * sum_{X,Y} sum_k G_{X,Y}(k) * S^r_X(k) * conj(S^r_Y(k))

        The total Green's function splits into C6 and C8 contributions:

            G_{X,Y}(k) = s6 * hat_f6_{X,Y}(k) + 3*sqrt(Q_X*Q_Y)*s8 * hat_f8_{X,Y}(k)

        Two numerical branches are used:
          - Large k (k*R0 >= thresh): closed-form exponential-trig expression.
          - Small k (k*R0 < thresh):  Taylor expansion to avoid 0/0 cancellation.

        For SPME, divides by sinc(m_x/N_x)^(2p) * sinc(m_y/N_y)^(2p) * sinc(m_z/N_z)^(2p)
        to correct for aliasing introduced by B-spline interpolation of order p
        (the P3M dealiasing scheme).

        Args:
            _k: k-vector norms, shape (nk,) for Ewald or (nx, ny, nz/2+1) for PME/SPME.

        Returns:
            For Ewald: tensor of shape (n_species, n_species, nk).
            For PME/SPME: complex tensor of shape (nk_flat, n_species, n_species),
                permuted and contiguous for batched matrix multiplication.
        """
        k = _k.view(1, 1, *_k.shape)  # broadcast over (n_species_X, n_species_Y, ...)
        ksq = torch.square(k)

        kRab = k * self.Rab
        small = kRab < self.thresh  # use Taylor branch where k*R0 is tiny

        # Substitute k=1 in the denominator of large-k formulas to avoid 0/0
        k_safe = torch.where(small, 1.0, k)
        kRab_safe = k_safe * self.Rab

        # --- C6 contribution: FT of s6 * r^6/(r^6 + R0^6) / r^6 ---
        # Taylor branch: hatf6 ≈ c1 - c2*k^2
        ft6_small = self.ft6_small_c1 - self.ft6_small_c2 * ksq
        # Large-k branch: derived via residues at the 6th roots of -R0^6.
        # The roots with positive imaginary part contribute; the result is:
        #   hatf6 = (2*pi^2)/(3*R0^4*k) * [exp(-k*R0) - 2*exp(-k*R0/2)*cos(pi/3 + k*R0*sqrt(3)/2)]
        num6 = torch.exp(-kRab_safe) - 2 * torch.exp(-kRab_safe / 2) * torch.cos(self.pi/3 + kRab_safe * self.sq3 / 2)
        ft6_large = (self.pre6_c / k_safe) * num6
        ft6 = self.params[0] * torch.where(small, ft6_small, ft6_large)

        # --- C8 contribution: FT of 3*s8*Q_{XY} * r^8/(r^8 + R0^8) / r^8 ---
        # Taylor branch: hatf8 ≈ c1 - c2*k^2
        ft8_small = self.ft8_small_c1 - self.ft8_small_c2 * ksq
        # Large-k branch: residues at 8th roots of -R0^8 with positive imaginary part.
        # These split into two groups: exp(-k*R0*sin(pi/8)) and exp(-k*R0*cos(pi/8)).
        exp1 = torch.exp(-kRab_safe * self.sin8)
        arg1 = (self.pi / 4) + (kRab_safe * self.cos8)

        exp2 = torch.exp(-kRab_safe * self.cos8)
        arg2 = (3 * self.pi / 4) + (kRab_safe * self.sin8)

        num8 = exp1 * torch.cos(arg1) + exp2 * torch.cos(arg2)
        ft8_large = (self.pre8_c / k_safe) * num8
        ft8 = 3 * self.params[1] * torch.where(small, ft8_small, ft8_large)

        # Combined Green's function: G_{X,Y}(k) = hatf6_{X,Y}(k) + Q_{XY} * hatf8_{X,Y}(k)
        kfilter = ft6 + self.QzQz_view * ft8

        if self.method == 'spme':
            # P3M dealiasing: divide by the squared B-spline window function.
            # For B-splines of order p, the interpolation introduces a smoothing
            # factor sinc(m/N)^p per dimension. Dividing by sinc^(2p) corrects this.
            mesh_shape = kfilter.shape[-3:]

            if self._cached_mesh_shape != mesh_shape:
                mesh_nx, mesh_ny, mesh_nz_raw = mesh_shape
                mesh_nz = (mesh_nz_raw - 1) * 2  # rfft stores only half the z-frequencies

                # Miller indices (fractional frequencies) for each dimension
                miller_x = torch.fft.fftfreq(mesh_nx, d=1.0 / mesh_nx, device=self.device)
                miller_y = torch.fft.fftfreq(mesh_ny, d=1.0 / mesh_ny, device=self.device)
                miller_z = torch.fft.rfftfreq(mesh_nz, d=1.0 / mesh_nz, device=self.device)

                sinc_x = torch.sinc(miller_x / mesh_nx)
                sinc_y = torch.sinc(miller_y / mesh_ny)
                sinc_z = torch.sinc(miller_z / mesh_nz)

                # Squared window function (product over dimensions)
                sc = sinc_x[:, None, None] * sinc_y[None, :, None] * sinc_z[None, None, :]
                sc = torch.pow(sc, 2*self.order)   # raised to 2p for dealiasing
                sc = torch.where(sc < 1e-10, 1e-10, sc)  # clamp to avoid division by zero

                self._cached_sc = sc
                self._cached_mesh_shape = mesh_shape

            kfilter = kfilter / self._cached_sc

        if self.method in ('pme', 'spme'):
            # For PME/SPME we need to account for the Hermitian symmetry of the rfftn
            # output: interior z-frequencies appear twice (once for +k, once for -k).
            # Weighting by 2 (with 1 at endpoints) avoids double-counting.
            last_dim = kfilter.shape[-1]
            if self._cached_weights is None or self._cached_weights.shape[0] != last_dim:
                weights = torch.full((last_dim,), 2.0, device=kfilter.device, dtype=kfilter.dtype)
                weights[0] = 1.0
                weights[-1] = 1.0
                self._cached_weights = weights

            k_weighted = kfilter * self._cached_weights
            # Flatten spatial dimensions and permute to (nk_flat, n_species_X, n_species_Y)
            # for batched matrix multiplication in KSpaceFilterD3.forward
            k_flat = k_weighted.flatten(2)

            complex_dtype = torch.complex128 if self.dtype == torch.float64 else torch.complex64
            return k_flat.to(dtype=complex_dtype).permute(2, 0, 1).contiguous()

        return kfilter

    def self_contribution(self) -> torch.tensor:
        """Return the diagonal self-contribution, shape (n_species, 1)."""
        return self.selfcont


class CNPotential(Potential):
    """
    D4 coordination number pair potential for use in periodic Ewald/PME summation.

    The D4 EN-weighted CN pair function for species pair (A, B) is:

        f_AB(r) = (delta_AB / 2) * erfc(k0 * (r - r0_AB) / r0_AB)

    where r0_AB = r_cov_A + r_cov_B is the sum of covalent radii, and the
    electronegativity weight is:

        delta_AB = k1 * exp(-(|EN_A - EN_B| + k2)^2 / k3)

    This function decays to zero at large r and converges smoothly, making it
    suitable for Ewald summation without a real-space cutoff.

    The 3D spherical Fourier transform (derived analytically via integration by
    parts, extending the lower limit to -inf with negligible error ~ exp(-k0^2)):

        hat_f_AB(k) = 4*pi * delta_AB * exp(-(k*r0)^2 / (4*k0^2))
                      * [(sin(k*r0) - k*r0*cos(k*r0)) / k^3
                         + r0^2 * sin(k*r0) / (2*k0^2 * k)]

    Small-k Taylor expansion (for numerical stability, using sin(x)-x*cos(x) = x^3/3 - x^5/30 + ...):

        hat_f_AB(k -> 0) = 4*pi * delta_AB * r0^3 * (1/3 + 1/(2*k0^2))
                           - 4*pi * delta_AB * r0^5 * (1/30 + 1/(6*k0^2) + 1/(8*k0^4)) * k^2 + ...

    The coordination number for atom i is obtained by convolving the species
    density (structure factor) with this potential in reciprocal space, then
    reading off the value at atom i's position (see `compute_cn_d4` in core.py).
    """

    def __init__(
        self,
        species: List,
        device,
        method: str,
        order: int = None,
        k0: float = 7.5,       # erfc width; large k0 makes the function close to a step
        k1: float = 4.10451,   # overall EN-weight scale
        k2: float = 19.08857,  # EN-weight offset
        k3: float = 2 * 11.28174**2,  # EN-weight variance
        dtype = torch.float32
    ):
        super().__init__()
        self.device = device
        self.order = order
        self.method = method
        self.dtype = dtype

        n = len(species)

        if self.method in ('pme', 'spme'):
            self.view_shape = (n, n, 1, 1, 1)
        else:
            self.view_shape = (n, n, 1)

        # r0_AB = r_cov_A + r_cov_B: pairwise covalent radius sum, shape (n, n, ...)
        rcov_full = load_rcov_cn()
        r_cov = rcov_full[list(species)].to(device=device, dtype=dtype)
        r0 = (r_cov.unsqueeze(0) + r_cov.unsqueeze(1)).view(*self.view_shape)
        self.r0 = r0
        self.r0sq = r0 * r0

        # delta_AB: electronegativity-weighted pair strength, shape (n, n, ...)
        en_full = load_en()
        en = en_full[list(species)].to(device=device, dtype=dtype)
        en_diff = torch.abs(en.unsqueeze(0) - en.unsqueeze(1))
        delta = k1 * torch.exp(-torch.square(en_diff + k2) / k3)
        self.delta = delta.view(*self.view_shape)

        # erfc width parameter k0
        k0t = torch.tensor(k0, device=device, dtype=dtype)
        self.k0 = k0t
        self.k0sq = k0t * k0t

        self.pi = torch.tensor(torch.pi, device=device, dtype=dtype)
        self.thresh = torch.tensor(1e-3, device=device, dtype=dtype)

        # Precomputed small-k Taylor coefficients (constant and k^2 terms):
        #   hatf(0)    = 4*pi * delta * r0^3 * A,  A = 1/3 + 1/(2*k0^2)
        #   hatf''(0)/2 = -4*pi * delta * r0^5 * B, B = 1/30 + 1/(6*k0^2) + 1/(8*k0^4)
        A = 1.0/3.0 + 1.0/(2.0 * k0**2)
        B = 1.0/30.0 + 1.0/(6.0 * k0**2) + 1.0/(8.0 * k0**4)
        self.small_c0 = 4.0 * self.pi * self.delta * self.r0sq * self.r0 * A
        self.small_c2 = 4.0 * self.pi * self.delta * self.r0sq * self.r0sq * self.r0 * B

        # Self-contribution: the value of f_AA(r) as r -> 0.
        # f_AA(0) = (delta_AA / 2) * erfc(-k0); for k0=7.5, erfc(-k0) ≈ 2 to machine precision.
        delta_diag = torch.diagonal(delta)
        self.selfcont = (delta_diag / 2.0 * torch.erfc(-k0t)).view(-1)

        self._cached_mesh_shape = None
        self._cached_sc = None
        self._cached_weights = None

    def lr_from_k_sq(self, _k: torch.Tensor) -> torch.Tensor:
        """Evaluate the Fourier transform of the D4 CN pair potential at each k.

        Returns hat_f_{AB}(k) for all species pairs (A, B) and all k-points.
        Used as the Green's function for computing D4 coordination numbers via
        Ewald/SPME (see `compute_cn_d4` in core.py).

        Two branches are used for numerical stability:
          - Large k (k*r0 >= thresh): full closed-form expression.
          - Small k (k*r0 < thresh):  Taylor expansion in k^2.

        For SPME, applies the same P3M dealiasing correction as D3Potential.
        """
        k = _k.view(1, 1, *_k.shape)  # broadcast over (n_species_A, n_species_B, ...)
        ksq = k * k

        kr0 = k * self.r0
        small = kr0 < self.thresh

        k_safe = torch.where(small, 1.0, k)
        kr0_safe = k_safe * self.r0

        # Gaussian envelope: exp(-(k*r0)^2 / (4*k0^2))
        gauss = torch.exp(-kr0_safe * kr0_safe / (4.0 * self.k0sq))

        # Trigonometric terms for the large-k formula
        s = torch.sin(kr0_safe)
        c = torch.cos(kr0_safe)

        # hat_f = 4*pi * delta * gauss * [(sin - k*r0*cos)/k^3 + r0^2*sin/(2*k0^2*k)]
        term1 = (s - kr0_safe * c) / (k_safe * k_safe * k_safe)
        term2 = self.r0sq * s / (2.0 * self.k0sq * k_safe)
        ft_large = 4.0 * self.pi * self.delta * gauss * (term1 + term2)

        # Taylor-expanded small-k formula: hatf ≈ small_c0 - small_c2 * k^2
        ft_small = self.small_c0 - self.small_c2 * ksq

        kfilter = torch.where(small, ft_small, ft_large)

        if self.method == 'spme':
            # P3M dealiasing: same correction as in D3Potential.lr_from_k_sq
            mesh_shape = kfilter.shape[-3:]

            if self._cached_mesh_shape != mesh_shape:
                mesh_nx, mesh_ny, mesh_nz_raw = mesh_shape
                mesh_nz = (mesh_nz_raw - 1) * 2

                miller_x = torch.fft.fftfreq(mesh_nx, d=1.0 / mesh_nx, device=self.device)
                miller_y = torch.fft.fftfreq(mesh_ny, d=1.0 / mesh_ny, device=self.device)
                miller_z = torch.fft.rfftfreq(mesh_nz, d=1.0 / mesh_nz, device=self.device)

                sinc_x = torch.sinc(miller_x / mesh_nx)
                sinc_y = torch.sinc(miller_y / mesh_ny)
                sinc_z = torch.sinc(miller_z / mesh_nz)

                sc = sinc_x[:, None, None] * sinc_y[None, :, None] * sinc_z[None, None, :]
                sc = torch.pow(sc, 2 * self.order)
                sc = torch.where(sc < 1e-10, 1e-10, sc)

                self._cached_sc = sc
                self._cached_mesh_shape = mesh_shape

            kfilter = kfilter / self._cached_sc

        if self.method in ('pme', 'spme'):
            # Flatten spatial dims and permute for use in KSpaceFilterCN.forward
            k_flat = kfilter.flatten(2)
            complex_dtype = torch.complex128 if self.dtype == torch.float64 else torch.complex64
            return k_flat.to(dtype=complex_dtype).permute(2, 0, 1).contiguous()

        return kfilter

    def self_contribution(self) -> torch.Tensor:
        """Return the diagonal self-contribution, shape (n_species,)."""
        return self.selfcont
