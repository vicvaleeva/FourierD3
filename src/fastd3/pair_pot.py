from typing import List

import torch
from torchpme.potentials import Potential

from fastd3.utils import load_sqrtQz

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