import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

class ProductLayer(nn.Module):
    def __init__(self, in_dim, num_prod, out_dim):
        super(ProductLayer, self).__init__()
        self.in_dim = in_dim
        self.num_prods = num_prod
        self.out_dim = out_dim
        assert in_dim >= 2*num_prod, "Error: in_dim < 2*num_prods!"
        self.in_dim_linear = in_dim + num_prod
        self.linear = nn.Linear(self.in_dim_linear, self.out_dim, bias=False)

    def get_prods(self, x):
        return torch.stack([x[:, 2 *i] * x[:, 2 *i + 1] for i in torch.arange(0, self.num_prods-1, dtype=torch.int8)], dim=1)

    def forward(self, x):
        x = torch.cat((x, self.get_prods(x)), dim=-1)
        x = self.linear(x)
        return x
    

class ProductLayer2D(ProductLayer):
    def __init__(self, in_dim, num_prods, out_dim):
        super().__init__(in_dim, num_prods, out_dim)
        self.linear = nn.Conv2d(self.in_dim_linear, self.out_dim, 1)
        self.range_prods = torch.tensor(np.array(range(self.num_prods)), dtype=int)
    
    def get_prods(self, x):
        return torch.stack([x[:, 2 *i, ...] * x[:, 2 *i + 1, ...] for i in self.range_prods], dim=1)
    
    def forward(self, x):
        x = torch.cat((x, self.get_prods(x)), dim=1)
        x = self.linear(x)
        return x

class ProductLayer1D(ProductLayer):
    def __init__(self, in_dim, num_prods, out_dim):
        super().__init__(in_dim, num_prods, out_dim)
        self.linear = nn.Conv1d(self.in_dim_linear, self.out_dim, 1)
        self.range_prods = torch.tensor(np.array(range(self.num_prods)), dtype=int)
    
    def get_prods(self, x):
        return torch.stack([x[:, 2 *i, ...] * x[:, 2 *i + 1, ...] for i in self.range_prods], dim=1)
    
    def forward(self, x):
        x = torch.cat((x, self.get_prods(x)), dim=1)
        x = self.linear(x)
        return x