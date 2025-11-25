from torchpme.potentials import Potential
from utils import load_sqrtQz

from typing import List

import torch
from numpy import pi

class D3Potential(Potential):
    
    def __init__(
        self,
        types: List,
        params: torch.Tensor,
        device
    ):
        super().__init__()
        self.types = types
        self.params = params.to(device)
        
        self.sqrtQz = load_sqrtQz(types, device)
        self.QzQz = torch.outer(self.sqrtQz, self.sqrtQz)
        self.Rab = (params[2]*torch.sqrt(3*self.QzQz) + params[3]).view(*self.Rab.shape, 1, 1, 1)
        self.Rab3 = torch.pow(self.Rab, 3)
        self.Rab4 = self.Rab * self.Rab3
        self.Rab5 = self.Rab * self.Rab4
        self.Rab6 = torch.pow(self.Rab3, 2)
        self.Rab8 = self.Rab3 * self.Rab5
        self.pi = torch.Tensor(pi, device=device)
        self.pi2 = self.pi**2
        self.sq2 = torch.sqrt(torch.Tensor(2.0), device=device)
        self.sq3 = torch.sqrt(torch.Tensor(3.0), device=device)
        self.sin8 = torch.sin(self.pi / 8)
        self.cos8 = torch.cos(self.pi/8)
        self.thresh = torch.Tensor(1e-15, device=device)
        
        diag6i = 1 / torch.diagonal(self.Rab6)
        diag8i = 1 / torch.diagonal(self.Rab8)
        diagQz = torch.diagonal(self.QzQz)
        self.selfcont = self.params[0] * diag6i + 3 * self.params[1] * diagQz * diag8i
        
        
    def lr_from_k_sq(self, k_sq: torch.Tensor) -> torch.Tensor:
        ksq = k_sq.view(1, 1, *k_sq.shape)
        k = torch.sqrt(ksq)
        kRab = k * self.Rab
        small = kRab < self.thresh
        k_safe = torch.where(small, 1.0, k)
        kRab_safe = k_safe * self.Rab
        
        ft6_small = (2*self.pi2) / (3*self.Rab3) - (self.pi2 / 18) * ksq
        
        num6 = (torch.exp(-kRab_safe) - 
                   2 * torch.exp(-kRab_safe /2) * torch.cos(self.pi/3 + kRab_safe * self.sq3 / 2))
        pre6 = (2*self.pi2) / (3*k_safe*self.Rab4)
        ft6_large = pre6 * num6
        
        ft6 = self.params[0] * torch.where(small, ft6_small, ft6_large)
        
        val0 = (self.pi2 * self.sq2 * self.sin8) / self.Rab5
        coeff_k2 = (self.pi2 * (self.sin8 + self.cos8)) / (16 * self.sq2 * self.Rab3)
        ft8_small = val0 - coeff_k2 * ksq
        
        exp1 = torch.exp(-kRab_safe * self.sin8)
        arg1 = (self.pi / 4) + (kRab_safe * self.cos8)
        
        exp2 = torch.exp(-kRab_safe * self.cos8)
        arg2 = (3 * pi / 4) + (kRab_safe * self.sin8)
        
        num8 = exp1 * torch.cos(arg1) + exp2 * torch.cos(arg2)
        pre8 = - (self.pi2) / (kRab_safe * self.Rab6)
        ft8_large = pre8 * num8
        
        ft8 = 3 * self.params[1] * torch.where(small, ft8_small, ft8_large)
        
        # returns (n_species, n_species, nx, ny, nz)
        
        return ft6 + torch.outer(self.sqrtQz, self.sqrtQz).unsqueeze(-1) * ft8
        
    def self_contribution(self) -> torch.Tensor:
        return self.selfcont