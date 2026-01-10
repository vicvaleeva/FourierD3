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
        method: str
    ):
        super().__init__()
        self.species = species
        self.params = params.to(device)
        self.method = method
        self.sqrtQz = load_sqrtQz(species, device=device)
        self.QzQz = torch.outer(self.sqrtQz, self.sqrtQz)
        if self.method == 'pme':
            self.Rab = (params[2]*torch.sqrt(3*self.QzQz) + params[3]).view(len(species), len(species), 1, 1, 1)
        elif self.method == 'ewald':
            self.Rab = (params[2]*torch.sqrt(3*self.QzQz) + params[3]).view(len(species), len(species), 1)
        self.Rab3 = torch.pow(self.Rab, 3)
        self.Rab4 = self.Rab * self.Rab3
        self.Rab5 = self.Rab * self.Rab4
        self.Rab6 = torch.pow(self.Rab3, 2)
        self.Rab8 = self.Rab3 * self.Rab5
        self.pi = torch.tensor(torch.pi, device=device)
        self.pi2 = self.pi**2
        self.sq2 = torch.sqrt(torch.tensor(2.0))
        self.sq3 = torch.sqrt(torch.tensor(3.0))
        self.sin8 = torch.sin(self.pi / 8)
        self.cos8 = torch.cos(self.pi / 8)
        self.thresh = torch.tensor(1e-15, device=device)
        
        diag6i = 1 / torch.diagonal(self.Rab6)
        diag8i = 1 / torch.diagonal(self.Rab8)
        diagQz = torch.diagonal(self.QzQz)
        self.selfcont = (self.params[0] * diag6i + 3 * self.params[1] * diagQz * diag8i).view(-1, 1)
        
        
    def lr_from_k_sq(self, _k: torch.tensor) -> torch.tensor:
        k = _k.view(1, 1, *_k.shape)
        ksq = torch.square(k)
        kRab = k * self.Rab
        small = kRab < self.thresh
        k_safe = torch.where(small, 1.0, k)
        kRab_safe = k_safe * self.Rab
        
        ft6_small = (2*self.pi2) / (3*self.Rab3) - (self.pi2 / 18) * ksq
        
        num6 = (torch.exp(-kRab_safe) - 
                   2 * torch.exp(-kRab_safe / 2) * torch.cos(self.pi/3 + kRab_safe * self.sq3 / 2))
        pre6 = (2*self.pi2) / (3*k_safe*self.Rab4)
        ft6_large = pre6 * num6
        
        ft6 = self.params[0] * torch.where(small, ft6_small, ft6_large)
        
        val0 = (self.pi2 * self.sq2 * self.sin8) / self.Rab5
        coeff_k2 = (self.pi2 * (self.sin8 + self.cos8)) / (16 * self.sq2 * self.Rab3)
        ft8_small = val0 - coeff_k2 * ksq
        
        exp1 = torch.exp(-kRab_safe * self.sin8)
        arg1 = (self.pi / 4) + (kRab_safe * self.cos8)
        
        exp2 = torch.exp(-kRab_safe * self.cos8)
        arg2 = (3 * self.pi / 4) + (kRab_safe * self.sin8)
        
        num8 = exp1 * torch.cos(arg1) + exp2 * torch.cos(arg2)
        pre8 = - (self.pi2) / (k_safe * self.Rab6)
        ft8_large = pre8 * num8
        
        ft8 = 3 * self.params[1] * torch.where(small, ft8_small, ft8_large)
        n_species = ft8.shape[0]
        # returns (n_species, n_species, nx, ny, nz) for pme, (n_species, n_species, nk) for ewald
        
        kfilter = (ft6 + self.QzQz.view(n_species, n_species, 1, 1, 1) * ft8)
        last_dim = kfilter.shape[-1]
        weights = torch.full((last_dim,), 2.0, device=kfilter.device, dtype=kfilter.dtype)
        weights[0] = 1.0
        weights[-1] = 1.0
            
        k_weighted = kfilter * weights
        k_flat = k_weighted.flatten(2)
        
        if self.method == 'pme':
            return k_flat.to(dtype=torch.complex128).permute(2, 0, 1).contiguous()
        
        return (ft6 + self.QzQz.view(n_species, n_species, 1) * ft8)
        
    def self_contribution(self) -> torch.tensor:
        return self.selfcont