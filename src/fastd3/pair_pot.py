from typing import List

import torch
from torchpme.potentials import Potential

from fastd3.utils import load_sqrtQz, load_en, load_rcov_cn

class D3Potential(Potential):
    
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
        self.params = params.to(device)
        self.method = method
        self.sqrtQz = load_sqrtQz(species, device=device)
        self.QzQz = torch.outer(self.sqrtQz, self.sqrtQz)
        self.dtype = dtype
        
        if self.method in ('pme', 'spme'):
            self.view_shape = (len(species), len(species), 1, 1, 1)
        else:
            self.view_shape = (len(species), len(species), 1)
            
        self.Rab = (params[2]*torch.sqrt(3*self.QzQz) + params[3]).view(*self.view_shape)
        self.QzQz_view = self.QzQz.view(*self.view_shape)
        
        self.Rab2 = torch.pow(self.Rab, 2)
        self.Rab3 = torch.pow(self.Rab, 3)
        self.Rab4 = self.Rab * self.Rab3
        self.Rab5 = self.Rab * self.Rab4
        self.Rab6 = torch.pow(self.Rab3, 2)
        self.Rab8 = self.Rab3 * self.Rab5
        
        self.pi = torch.tensor(torch.pi, device=device)
        self.pi2 = self.pi**2
        self.sq2 = torch.sqrt(torch.tensor(2.0, device=device))
        self.sq3 = torch.sqrt(torch.tensor(3.0, device=device))
        self.sin8 = torch.sin(self.pi / 8)
        self.cos8 = torch.cos(self.pi / 8)
        self.thresh = torch.tensor(1e-15, device=device)
        
        diag6i = 1 / torch.diagonal(self.Rab6)
        diag8i = 1 / torch.diagonal(self.Rab8)
        diagQz = torch.diagonal(self.QzQz)
        self.selfcont = (self.params[0] * diag6i + 3 * self.params[1] * diagQz * diag8i).view(-1, 1)

        self.ft6_small_c1 = (2 * self.pi2) / (3 * self.Rab3)
        self.ft6_small_c2 = (2 * self.pi2 * self.Rab) / 9
        self.pre6_c = (2 * self.pi2) / (3 * self.Rab4)

        self.ft8_small_c1 = (self.pi2 * self.sq2 * self.sin8) / self.Rab5
        self.ft8_small_c2 = self.ft8_small_c1 * self.Rab2 / 6
        self.pre8_c = - (self.pi2) / self.Rab6
        
        self._cached_mesh_shape = None
        self._cached_sc = None
        self._cached_weights = None


    def lr_from_k_sq(self, _k: torch.tensor) -> torch.tensor:
        k = _k.view(1, 1, *_k.shape)
        ksq = torch.square(k)
        
        kRab = k * self.Rab
        small = kRab < self.thresh
        
        k_safe = torch.where(small, 1.0, k)
        kRab_safe = k_safe * self.Rab
        
        ft6_small = self.ft6_small_c1 - self.ft6_small_c2 * ksq
        num6 = torch.exp(-kRab_safe) - 2 * torch.exp(-kRab_safe / 2) * torch.cos(self.pi/3 + kRab_safe * self.sq3 / 2)
        
        ft6_large = (self.pre6_c / k_safe) * num6
        ft6 = self.params[0] * torch.where(small, ft6_small, ft6_large)
        
        ft8_small = self.ft8_small_c1 - self.ft8_small_c2 * ksq
        
        exp1 = torch.exp(-kRab_safe * self.sin8)
        arg1 = (self.pi / 4) + (kRab_safe * self.cos8)
        
        exp2 = torch.exp(-kRab_safe * self.cos8)
        arg2 = (3 * self.pi / 4) + (kRab_safe * self.sin8)
        
        num8 = exp1 * torch.cos(arg1) + exp2 * torch.cos(arg2)
        
        ft8_large = (self.pre8_c / k_safe) * num8
        ft8 = 3 * self.params[1] * torch.where(small, ft8_small, ft8_large)
        
        kfilter = ft6 + self.QzQz_view * ft8
        
        if self.method == 'spme':
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
                sc = torch.pow(sc, 2*self.order)
                sc = torch.where(sc < 1e-10, 1e-10, sc)
                
                self._cached_sc = sc
                self._cached_mesh_shape = mesh_shape
                
            kfilter = kfilter / self._cached_sc
            
        if self.method in ('pme', 'spme'):
            last_dim = kfilter.shape[-1]
            if self._cached_weights is None or self._cached_weights.shape[0] != last_dim:
                weights = torch.full((last_dim,), 2.0, device=kfilter.device, dtype=kfilter.dtype)
                weights[0] = 1.0
                weights[-1] = 1.0
                self._cached_weights = weights
                
            k_weighted = kfilter * self._cached_weights
            k_flat = k_weighted.flatten(2)
            
            complex_dtype = torch.complex128 if self.dtype == torch.float64 else torch.complex64
            return k_flat.to(dtype=complex_dtype).permute(2, 0, 1).contiguous()
        
        return kfilter

    def self_contribution(self) -> torch.tensor:
        return self.selfcont
    
    
class CNPotential(Potential):
    """
    D4 coordination number pair potential for use in periodic Ewald/PME summation.

    The D4 EN-weighted CN pair function for species pair (A, B):

        f_AB(r) = (delta_AB / 2) * erfc(k0 * (r - r0_AB) / r0_AB)

    where r0_AB = r_cov_A + r_cov_B and the electronegativity weight is:

        delta_AB = k1 * exp(-(|EN_A - EN_B| + k2)^2/k3) 

    Spherically symmetric 3D Fourier transform (analytically derived):

        hatf_AB(k) = 4*pi * delta_AB * exp(-(k*r0)^2 / (4*k0^2))
                     * [ (sin(k*r0) - k*r0*cos(k*r0)) / k^3
                         + r0^2 * sin(k*r0) / (2*k0^2 * k) ]

    Derivation:  integration by parts on I = int_0^inf erfc(alpha*r - beta)*sin(k*r)*r dr,
    substituting t = alpha*r - beta and extending the lower limit -beta -> -inf
    (error ~ exp(-k0^2) ~ 10^-24 for k0=7.5), then evaluating the resulting
    full-axis Gaussian integrals in closed form.

    Small-k limit (Taylor expansion, sin(x)-x*cos(x) = x^3/3 - x^5/30 + ...):

        hatf_AB(k -> 0) = 4*pi * delta_AB * r0^3 * (1/3 + 1/(2*k0^2))
    """

    def __init__(
        self,
        species: List,
        device,
        method: str,
        order: int = None,
        k0: float = 7.5,
        k1: float = 4.10451,
        k2: float = 19.08857,
        k3: float = 2 * 11.28174**2,
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

        # r0_AB = r_cov_A + r_cov_B  (indexed by unique species atomic number)
        rcov_full = load_rcov_cn()
        r_cov = rcov_full[list(species)].to(device=device, dtype=dtype)  # (n,)
        r0 = (r_cov.unsqueeze(0) + r_cov.unsqueeze(1)).view(*self.view_shape)  # (n, n, ...)
        self.r0 = r0
        self.r0sq = r0 * r0

        # delta_AB = k1 * exp((|EN_A - EN_B| + k2)^2) / k3
        en_full = load_en()
        en = en_full[list(species)].to(device=device, dtype=dtype)  # (n,)
        en_diff = torch.abs(en.unsqueeze(0) - en.unsqueeze(1))  # (n, n)
        delta = k1 * torch.exp(-torch.square(en_diff + k2) / k3)  # (n, n)
        self.delta = delta.view(*self.view_shape)

        # erfc width parameter
        k0t = torch.tensor(k0, device=device, dtype=dtype)
        self.k0 = k0t
        self.k0sq = k0t * k0t

        self.pi = torch.tensor(torch.pi, device=device, dtype=dtype)
        self.thresh = torch.tensor(1e-3, device=device, dtype=dtype)

        # Small-k (k->0) precomputed constants per species pair:
        #   hatf(0)   = 4*pi * delta * r0^3 * (1/3 + 1/(2*k0^2))
        #   hatf''(0) = -4*pi * delta * r0^5 * (1/30 + 1/(6*k0^2) + 1/(8*k0^4))
        A = 1.0/3.0 + 1.0/(2.0 * k0**2)
        B = 1.0/30.0 + 1.0/(6.0 * k0**2) + 1.0/(8.0 * k0**4)
        self.small_c0 = 4.0 * self.pi * self.delta * self.r0sq * self.r0 * A
        self.small_c2 = 4.0 * self.pi * self.delta * self.r0sq * self.r0sq * self.r0 * B

        # Self-contribution: f_AA(0) = (delta_AA / 2) * erfc(-k0)
        # For k0=7.5, erfc(-k0) = 1 + erf(k0) ≈ 2 to machine precision.
        delta_diag = torch.diagonal(delta)  # (n,)
        self.selfcont = (delta_diag / 2.0 * torch.erfc(-k0t)).view(-1)

        self._cached_mesh_shape = None
        self._cached_sc = None
        self._cached_weights = None

    def lr_from_k_sq(self, _k: torch.Tensor) -> torch.Tensor:
        k = _k.view(1, 1, *_k.shape)  # (1, 1, ...)
        ksq = k * k

        kr0 = k * self.r0  # (n, n, ...)
        small = kr0 < self.thresh

        # Avoid division by zero: replace k=0 with k=1 in the safe branch
        k_safe = torch.where(small, 1.0, k)
        kr0_safe = k_safe * self.r0

        # Gaussian envelope: exp(-(k*r0)^2 / (4*k0^2))
        gauss = torch.exp(-kr0_safe * kr0_safe / (4.0 * self.k0sq))

        # Trigonometric terms for large-k formula
        s = torch.sin(kr0_safe)
        c = torch.cos(kr0_safe)

        # hatf = 4*pi * delta * gauss * [(sin - kr0*cos)/k^3 + r0^2*sin/(2*k0^2*k)]
        term1 = (s - kr0_safe * c) / (k_safe * k_safe * k_safe)
        term2 = self.r0sq * s / (2.0 * self.k0sq * k_safe)
        ft_large = 4.0 * self.pi * self.delta * gauss * (term1 + term2)

        # Taylor-expanded small-k formula: hatf ≈ small_c0 - small_c2 * k^2
        ft_small = self.small_c0 - self.small_c2 * ksq

        kfilter = torch.where(small, ft_small, ft_large)

        if self.method == 'spme':
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
            k_flat = kfilter.flatten(2)

            complex_dtype = torch.complex128 if self.dtype == torch.float64 else torch.complex64
            return k_flat.to(dtype=complex_dtype).permute(2, 0, 1).contiguous()

        return kfilter

    def self_contribution(self) -> torch.Tensor:
        return self.selfcont