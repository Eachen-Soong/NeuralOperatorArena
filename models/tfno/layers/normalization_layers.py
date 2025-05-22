import torch
import torch.nn as nn
from einops import repeat

class AdaIN(nn.Module):
    def __init__(self, embed_dim, in_channels, mlp=None, eps=1e-5):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.eps = eps

        if mlp is None:
            mlp = nn.Sequential(
                nn.Linear(embed_dim, 512),
                nn.GELU(),
                nn.Linear(512, 2*in_channels)
            )
        self.mlp = mlp

        self.embedding = None
    
    def set_embedding(self, x):
        self.embedding = x.reshape(self.embed_dim,)

    def forward(self, x):
        assert self.embedding is not None, "AdaIN: update embeddding before running forward"

        weight, bias = torch.split(self.mlp(self.embedding), self.in_channels, dim=0)

        return nn.functional.group_norm(x, self.in_channels, weight, bias, eps=self.eps)


class InstanceNorm(nn.Module):
    def __init__(self, num_features, n_dim):
        super().__init__()
        if n_dim == 1:
            self.norm = nn.InstanceNorm1d(num_features)
        elif n_dim == 2:
            self.norm = nn.InstanceNorm2d(num_features)
        elif n_dim == 3:
            self.norm = nn.InstanceNorm3d(num_features)

    def forward(self, input, *args, **kwargs):
        return self.norm(input)
    
    def set_coeffs(self, *args, **kwargs):
        return


class DimNorm(nn.Module):
    """
    Grouped Instance Norm, would scale the input constants
    """
    def __init__(self, num_consts, in_channels, n_dim=2, eps=1e-5):
        super().__init__()
        self.in_channels = in_channels
        self.num_consts = num_consts
        self.eps = eps
        group_N = in_channels//num_consts
        num_group1 = in_channels%num_consts
        if num_group1:
            def extend(consts:torch.Tensor):
                return torch.cat([consts[:, :num_group1].repeat(1, group_N+1), consts[:, num_group1:].repeat(1, group_N)], dim=1)
        else:
            def extend(consts:torch.Tensor):
                return consts.repeat(1, group_N)
        self.extend = extend
        if n_dim == 2:
            def transform(weight, bias, x):
                return torch.einsum('bc, bcxy -> bcxy', weight, x, ) + repeat(bias, 'b c -> b c x y', x=x.shape[-2], y=x.shape[-1])
        elif n_dim == 1:
            def transform(weight, bias, x):
                return torch.einsum('bc, bcx -> bcx', weight, x,) + repeat(bias, 'b c -> b c x', x=x.shape[-1])
        elif n_dim == 3:
            def transform(weight, bias, x):
                return torch.einsum('bc, bcxyz -> bcxyz', weight, x, ) + repeat(bias, 'b c -> b c x y z', x=x.shape[-3], y=x.shape[-2], z=x.shape[-1])
        else:
            transform = lambda weight, bias, x: x
        self.transform = transform
        self.weight_coeff = nn.Parameter(torch.ones(num_consts), requires_grad=True)
        self.bias_coeff = nn.Parameter(torch.ones(num_consts), requires_grad=True)
    
    def set_coeffs(self, weight_coeff:torch.Tensor, bias_coeff:torch.Tensor):
        device = self.weight_coeff.data.device
        self.weight_coeff.data = weight_coeff.to(device)
        self.bias_coeff.data = bias_coeff.to(device)

    def forward(self, x, std, mu):
        weight = self.extend(self.weight_coeff*std)
        bias = self.extend(self.bias_coeff*mu)
        x = nn.functional.instance_norm(x, eps=self.eps)
        return self.transform(weight, bias, x)
