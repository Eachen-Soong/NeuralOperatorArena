import torch
import numpy as np
from enum import Enum
import math

device = torch.device('cuda')

class Force(str, Enum):
    li = 'li'
    random = 'random'
    none = 'none'
    kolmogorov = 'kolmogorov'

def non_random_force(force:Force, s):
    assert force in Force and force != Force.random, "Force type error!"
    if force == Force.li:
        # Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
        ft = torch.linspace(0, 1, s+1)
        ft = ft[0:-1]
        X, Y = torch.meshgrid(ft, ft, indexing='ij')
        f = 0.1*(torch.sin(2 * math.pi * (X + Y)) +
                 torch.cos(2 * math.pi * (X + Y)))
        return f
    if force == Force.kolmogorov:
        ft = torch.linspace(0, 2 * np.pi, s + 1)
        ft = ft[0:-1]
        X, Y = torch.meshgrid(ft, ft, indexing='ij')
        f = -4 * torch.cos(4 * Y)
        return f
    if force == Force.none:
        return None

class Random_Force_Generator:
    """
    In the F-FNO data generator implementation, the force is determined by 6*cycles random values,
    which are sampled on uniform distribution on interval [0, 1).
    They're the sin , cos of k[?]+wt, [?] is x, y, x+y; k is in [1, cycles]* 2 pi.
    Say, the coeffs are a_ij, i is [?], j is sin/cos, then:
    f_i = a_ij 
    """
    def __init__(self, b, s, device, cycles, scaling=1., t_scaling=0., seed=114514, coeff=None):
        self.device = device
        self.cycles = cycles
        self.scaling = scaling
        self.t_scaling = t_scaling
        self.seed = seed
        self.gen = torch.Generator(device)
        self.gen.manual_seed(seed)
        if coeff is not None:
            self.coeff = coeff
            assert coeff.shape == [b, cycles, 3, 2], "coeff shape error!"
        else: self.coeff = self.get_coeff(b, seed)
        self.base_waves = self.get_base_waves(s, cycles)

    def get_base_waves(self, s, cycles):
        base_waves = torch.zeros([3, 2, cycles, s, s])
        ft = torch.linspace(0, 1, s+1)
        ft = ft[0:-1]
        X, Y = torch.meshgrid(ft, ft, indexing='ij')
        p = torch.arange(1, cycles+1)
        k = 2 * torch.pi * p
        base_waves[0,0,...] = torch.sin(torch.einsum('k, x y -> k x y',k, X))
        base_waves[0,1,...] = torch.cos(torch.einsum('k, x y -> k x y',k, X))
        base_waves[1,0,...] = torch.sin(torch.einsum('k, x y -> k x y',k, Y))
        base_waves[1,1,...] = torch.cos(torch.einsum('k, x y -> k x y',k, Y))
        base_waves[2,0,...] = torch.sin(torch.einsum('k, x y -> k x y',k, X+Y))
        base_waves[2,1,...] = torch.cos(torch.einsum('k, x y -> k x y',k, X+Y))
        return base_waves

    def get_coeff(self, b, seed=None):
        if seed is not None:
            self.gen.manual_seed(seed)
        return torch.rand(b, self.cycles, 3, 2, generator=self.gen, device=self.device)

    def get_force(self, t=0.):
        if t == 0. or self.t_scaling == 0.:
            time_mat = torch.eye(2).to(self.device)
        else:
            phi = torch.scalar_tensor(t*self.t_scaling)
            time_mat = torch.stack([torch.stack([torch.cos(phi), -torch.sin(phi)]), 
                   torch.stack([torch.sin(phi), torch.cos(phi)])]).to(self.device)
        base_waves = self.base_waves.to(self.device)
        raw_basis = torch.einsum('a p k s t, p q -> a q k s t', base_waves, time_mat)
        force = self.scaling * torch.einsum('a p k s t, b k a p -> b s t', raw_basis, self.coeff)
        return force