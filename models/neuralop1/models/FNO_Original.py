"""
@author: Zongyi Li
modified by Eachen Soong to adapt to this code base
"""
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from .modules import SpectralConv2d, SpectralConvProd2d
from ..layers.mlp import MLP
from functools import reduce, partial
import operator

################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1] = torch.einsum("bix,iox->box", x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft) # , s=(x.size(-1), )
        return x

class FNO_1D(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, modes=21, width=32, use_position=True):
        super(FNO_1D, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_position = use_position
        self.modes1 = modes
        self.width = width
        # self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)
        appended_dim = 1 if use_position else 0
        
        self.fc0 = nn.Conv1d(self.in_dim+appended_dim, self.width, 1)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Conv1d(self.width, 128, 1)
        self.fc2 = nn.Conv1d(128, out_dim, 1)
        # self.fc1 = nn.Linear(self.width, 128)
        # self.fc2 = nn.Linear(128, 1)

    def forward(self, x, **kwargs):
        if self.use_position:
            grid = self.get_grid(x.shape, x.device)
            x = torch.cat((x, grid), dim=1)

        x = self.fc0(x)
        # x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[-1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x).repeat([batchsize, 1, 1])
        return gridx.to(device)


class ProdFNO_1D(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, modes=21, width=32, use_position=True, num_prod=2, skip_connection=True, ):
        super(ProdFNO_1D, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_position = use_position
        self.modes1 = modes
        self.width = width
        self.num_prod = num_prod
        self.skip_connection = skip_connection
        # self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)
        appended_dim = 1 if use_position else 0
        
        self.fc0 = nn.Conv1d(self.in_dim+appended_dim, self.width, 1)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.prod0 = ProductLayer1D(2 * self.width, num_prod, self.width)
        self.prod1 = ProductLayer1D(2 * self.width, num_prod, self.width)

        self.fc1 = nn.Conv1d(self.width, 128, 1)
        self.fc2 = nn.Conv1d(128, out_dim, 1)
        # self.fc1 = nn.Linear(self.width, 128)
        # self.fc2 = nn.Linear(128, 1)

    def forward(self, x, **kwargs):
        if self.use_position:
            grid = self.get_grid(x.shape, x.device)
            x = torch.cat((x, grid), dim=1)

        x = self.fc0(x)
        # x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        xo = torch.cat((x1, x2), dim=1) # [batch, width, x, y] -> [batch, 2*width, x, y]
        xo = self.prod0(xo)
        if self.skip_connection:
            x = x + xo
        else: x = xo
        # x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        xo = torch.cat((x1, x2), dim=1)
        xo = self.prod1(xo)
        if self.skip_connection:
            x = x + xo
        else: x = xo
        # x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[-1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x).repeat([batchsize, 1, 1])
        return gridx.to(device)
    
################################################################
# fourier layer
################################################################
class FNO_2D(nn.Module):
    def __init__(self, in_dim, out_dim, modes1, modes2, width, use_position=True):
        super(FNO_2D, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        # self.padding = 9 # pad the domain if input is non-periodic
        self.use_position = use_position
        appended_dim = 2 if use_position else 0

        # self.p = nn.Linear(in_dim + appended_dim, self.width) # input channel is 3: (a(x, y), x, y)
        self.p = nn.Conv2d(in_channels=in_dim + appended_dim, out_channels=self.width, kernel_size=1)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.q = MLP(self.width, out_dim, self.width * 4) # output channel is 1: u(x, y)

    def forward(self, x, **kwargs):
        if self.use_position:
            grid = self.get_grid(x.shape, x.device)
            # x = torch.cat((x, grid), dim=-1)
            x = torch.cat((x, grid), dim=1)

        x = self.p(x)
        # x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding]
        x = self.q(x)
        # x = x.permute(0, 2, 3, 1)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[-2], shape[-1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)

    
class FNO_2D_test(nn.Module):
    def __init__(self, in_dim, appended_dim, out_dim, modes1, modes2, width, use_position=True, skip_connection=True):
        super(FNO_2D_test, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.in_dim = in_dim
        self.appended_dim = appended_dim
        self.out_dim = out_dim
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        # self.padding = 9 # pad the domain if input is non-periodic
        self.use_position = use_position
        self.skip_connection = skip_connection

        self.p = nn.Linear(in_dim + appended_dim, self.width) # input channel is 3: (a(x, y), x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        # self.w0 = nn.Conv2d(self.width, self.width, 1)
        # self.w1 = nn.Conv2d(self.width, self.width, 1)
        # self.w2 = nn.Conv2d(self.width, self.width, 1)
        # self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.w0 = nn.Conv2d(self.width, self.width, 3, padding=1, padding_mode='circular')
        self.w1 = nn.Conv2d(self.width, self.width, 3, padding=1, padding_mode='circular')
        self.w2 = nn.Conv2d(self.width, self.width, 3, padding=1, padding_mode='circular')
        self.w3 = nn.Conv2d(self.width, self.width, 3, padding=1, padding_mode='circular')
        self.q = MLP(self.width, out_dim, self.width * 4) # output channel is 1: u(x, y)

    def forward(self, x):
        if self.use_position:
            grid = self.get_grid(x.shape, x.device)
            x = torch.cat((x, grid), dim=-1)

        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        xo = x1 + x2
        if self.skip_connection:
            x = x + xo
        else: x = xo
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        xo = x1 + x2
        if self.skip_connection:
            x = x + xo
        else: x = xo
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        xo = x1 + x2
        if self.skip_connection:
            x = x + xo
        else: x = xo
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        xo = x1 + x2
        if self.skip_connection:
            x = x + xo
        else: x = xo

        x = x[..., :-self.padding, :-self.padding]
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
    

class FNO_2D_test1(nn.Module):
    def __init__(self, in_dim, out_dim, modes1, modes2, width, use_position=True, skip_connections=None):
        super(FNO_2D_test1, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic
        self.use_position = use_position
        self.skip_connections = [True, True, False, False]

        self.p = nn.Linear(in_dim + 2, self.width) # input channel is 3: (a(x, y), x, y)
        # self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        # self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        # self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        # self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.num_prod = 2
        self.conv0 = SpectralConvProd2d(self.width, self.width, self.modes1, self.modes2, self.num_prod)
        self.conv1 = SpectralConvProd2d(self.width, self.width, self.modes1, self.modes2, self.num_prod)
        self.conv2 = SpectralConvProd2d(self.width, self.width, self.modes1, self.modes2, self.num_prod)
        self.conv3 = SpectralConvProd2d(self.width, self.width, self.modes1, self.modes2, self.num_prod)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        # self.w0 = nn.Conv2d(self.width, self.width, 1)
        # self.w1 = nn.Conv2d(self.width, self.width, 1)
        # self.w2 = nn.Conv2d(self.width, self.width, 1)
        # self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.w0 = nn.Conv2d(self.width, self.width, 3, padding=1, padding_mode='circular')
        self.w1 = nn.Conv2d(self.width, self.width, 3, padding=1, padding_mode='circular')
        self.w2 = nn.Conv2d(self.width, self.width, 3, padding=1, padding_mode='circular')
        self.w3 = nn.Conv2d(self.width, self.width, 3, padding=1, padding_mode='circular')
        self.q = MLP(self.width, out_dim, self.width * 4) # output channel is 1: u(x, y)

    def forward(self, x):
        if self.use_position:
            grid = self.get_grid(x.shape, x.device)
            x = torch.cat((x, grid), dim=-1)

        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(x)
        # x1 = self.mlp0(x1)
        x2 = self.w0(x)
        xo = x1 + x2
        if self.skip_connections[0]:
            x = x + xo
        else: x = xo
        x = F.gelu(x)

        x1 = self.conv1(x)
        # x1 = self.mlp1(x1)
        x2 = self.w1(x)
        xo = x1 + x2
        if self.skip_connections[1]:
            x = x + xo
        else: x = xo
        x = F.gelu(x)

        x1 = self.conv2(x)
        # x1 = self.mlp2(x1)
        x2 = self.w2(x)
        xo = x1 + x2
        if self.skip_connections[2]:
            x = x + xo
        else: x = xo
        x = F.gelu(x)

        x1 = self.conv3(x)
        # x1 = self.mlp3(x1)
        x2 = self.w3(x)
        xo = x1 + x2
        if self.skip_connections[3]:
            x = x + xo
        else: x = xo

        x = x[..., :-self.padding, :-self.padding]
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


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

def get_mat_index(index, column):
    if (index+1)% column == 0:
        i = (index+1)//column - 1
        j = column - 1
    else:
        i = (index+1)//column
        j = ((index+1)%column) - 1
    if i>j:
        tmp = i;    i = j;    j = tmp
    return torch.tensor([i, j], dtype=torch.int16)


class QuadPath(nn.Module):
    def __init__(self, in_dim, out_dim, num_quad, num_prod):
        super(QuadPath, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_quad = num_quad
        self.num_prod = num_prod
        self.out_range = torch.tensor(np.array(range(out_dim)), dtype=torch.int16)
        
        self.prep_stage = True
        # Stage 1 params
        self.quadratic = nn.Linear(self.in_dim, self.num_quad, bias=False)  # First transformation for quadratic
        self.quadratic.weight.data = self.gram_schmidt(self.quadratic.weight.data)  # Initialize weights orthogonally
        self.quad_linear = nn.Linear(self.num_quad, self.out_dim, bias=False)  # Second transformation for quadratic
        # Stage 2 params
        self.prod_indices = torch.zeros([num_prod, 2], dtype=torch.int16)  # quad_indices[0] <= ~[1]
        self.prod_linear = nn.Linear(num_prod, self.out_dim, bias=False)
        nn.init.constant_(self.quad_linear.weight, 0)

    def forward(self, x):
        if self.prep_stage:
            y_quadratic = self.quadratic(x)
            out_quadratic = torch.square(y_quadratic)
            out_quadratic = self.quad_linear(out_quadratic)
        else:
            out_quadratic = self.calc_quads(x)
            out_quadratic = self.prod_linear(out_quadratic)

        return out_quadratic
    
    def calc_quads(self, x):
        # out_quadratic = torch.stack([func(x_T) for func in self.significant_quad_funcs]).T
        # x.shape = [B, C]
        out_quadratic = torch.stack([x[..., self.prod_indices[i][0]] * x[..., self.prod_indices[i][1]] for i in self.out_range], dim=-1)
        return out_quadratic
    
    def get_mat(self, linear):
        return linear.weight.data

    # replace previous representations if the proportion of new representation reaches threshold
    def replace_previous(self, threshold=0.0):
        Q = self.get_mat(self.quadratic)
        Lams = self.get_mat(self.quad_linear)
        n = Lams.shape[0]
        # A = torch.einsum('a b , n b c ->n a c', Q.T, Lams.diag_embed())
        # A = torch.einsum('n a b , b c ->n a c', A, Q)
        A = Q.T @ Lams.diag_embed() @ Q
        B = torch.stack([torch.triu(2. * A[i] - torch.diag(A[i].diag())) for i in range(n)])
        B1 = torch.zeros(B.shape)
        P = torch.zeros(B.shape)

        # Pick the top num_prod for each matrix        
        for i in range(n):
            B_view = B[i].view(-1)
            indices = torch.topk(abs(B_view), self.num_prod)
            sum_prods = torch.sum(B_view)
            for j in range(self.num_prod):
                idx = indices[1][j]
                index = get_mat_index(idx, A.shape[1])
                B1[i, index[0], index[1]] = B_view[idx]
                P[i, index[0], index[1]] = B_view[idx] / sum_prods

        # Pick top num_prod elems from P
        P_view = torch.sum(P, dim=0).view(-1)
        indices = torch.topk(P_view, self.num_prod)
        P1 = torch.zeros(B.shape)
        indices_2d = torch.zeros([self.num_prod, 2], dtype=torch.int16)
        for j in range(self.num_prod):
            idx = indices[1][j]
            index = get_mat_index(idx, A.shape[1])
            indices_2d[j] = index
            P1[:, index[0], index[1]] = P[:, index[0], index[1]]
        
        proportion = torch.sum(P1, dim=[1, 2])
        if_pass = torch.all(proportion >= threshold)
        
        if not if_pass:
            return proportion
        
        self.prep_stage = False

        self.prod_indices = indices_2d

        for (j, idx) in enumerate(indices_2d):
            self.prod_linear.weight.data[:, j] = B[:, idx[0], idx[1]]

        # Update the initial matrices
        # A_new = A.clone()
        # for idx in self.quad_indices:
        #     A_new[idx[0], idx[1]] = 0
        #     A_new[idx[1], idx[0]] = 0

        # L, V = torch.linalg.eig(A_new)

        # self.quadratic.weight.data = V.real.clone()[:self.num_quad, :]
        # self.quad_linear_prep.weight.data = L[:self.num_quad_prep].real.clone()
        return proportion
    
    def normalize(self):
        """
        Normalizes the rows of the quadratic layer and adjusts the quad_linear_prep accordingly.
        """
        with torch.no_grad():
            norms = torch.norm(self.quadratic.weight.data, dim=1, keepdim=True)
            self.quadratic.weight.data /= norms
            self.quad_linear.weight.data *= norms.squeeze()

    def gram_schmidt(self, v):
        """
        Applies the Gram-Schmidt method to the given matrix v for QR Factorization.
        The orthogonalization is performed along the rows, not the columns.
        """
        ortho_matrix = torch.zeros_like(v)
        for i in range(v.size(0)):
            # orthogonalization
            vec = v[i, :]
            space = ortho_matrix[:i, :]
            projection = torch.mm(space, vec.unsqueeze(1))
            vec = vec - torch.sum(projection, dim=0)
            # normalization
            norm = torch.norm(vec)
            vec = vec / norm
            ortho_matrix[i, :] = vec
        return ortho_matrix

    def reorder(self):
        """
        Reorders the rows of the quadratic layer based on the sum of the absolute values of the weights in each row of quad_linear_prep.
        """
        with torch.no_grad():
            # Get the indices that would sort the sum of the absolute values of quad_linear_prep's weights
            _, sorted_indices = torch.sort(self.quad_linear.weight.abs().sum(dim=0), descending=True)
            self.quadratic.weight.data = self.quadratic.weight.data[sorted_indices]
            self.quad_linear.weight.data = self.quad_linear.weight.data[:, sorted_indices]

    def orthonormalize_prep(self):
        self.normalize()
        self.reorder()
        self.gram_schmidt(self.quadratic.weight.data)

    def orthogonality_loss(self):
        """
        A loss function that encourages the weights of the quadratic layer to be orthogonal.
        """
        w = self.quadratic.weight
        gram = torch.mm(w, w.t())
        eye = torch.eye(gram.shape[0], device=gram.device)
        loss = torch.norm(gram - eye)
        return loss


class ProdFNO_2D_test(nn.Module):
    def __init__(self, in_dim, out_dim, modes1, modes2, width,
                 num_prod, skip_connection=True, use_position=True,):
        super(ProdFNO_2D_test, self).__init__()
        self.in_dim = in_dim
        self.appended_dim = 2 # 2 parts: from the dataset and from the model itself
        self.out_dim = out_dim
        self.modes1 = modes1
        self.modes2 = modes2
        self.num_prod = num_prod
        self.width = width
        self.padding = 9
        self.skip_connection = skip_connection
        self.use_position = use_position
        
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 3, padding=1, padding_mode='circular')
        self.w1 = nn.Conv2d(self.width, self.width, 3, padding=1, padding_mode='circular')
        self.w2 = nn.Conv2d(self.width, self.width, 3, padding=1, padding_mode='circular')
        self.w3 = nn.Conv2d(self.width, self.width, 3, padding=1, padding_mode='circular')

        self.prod0 = ProductLayer2D(2 * self.width, num_prod, self.width)
        self.prod1 = ProductLayer2D(2 * self.width, num_prod, self.width)
        self.prod2 = ProductLayer2D(2 * self.width, num_prod, self.width)
        self.prod3 = ProductLayer2D(2 * self.width, num_prod, self.width)

        # self.q0 = QuadraticLayer2D(self.in_dim + self.appended_dim, 3, 2, self.width)
        self.q0 = QuadPath(self.in_dim + self.appended_dim, self.num_prod, self.num_prod, self.num_prod)
        assert self.width - self.in_dim - self.appended_dim - self.num_prod > 0, "width <= in_dim + appended_dim + num_prod !"
        self.fc0 = nn.Linear(self.in_dim + self.appended_dim + self.num_prod, self.width - self.in_dim - self.appended_dim - self.num_prod)
        # self.fc1 = nn.Linear(self.width, self.out_dim)
        self.fc1 = nn.Conv2d(in_channels=self.width, out_channels=self.out_dim, kernel_size=1)


        self.quads = [self.q0]

    def forward(self, x, **kwargs):
        if self.use_position:
            grid = self.get_grid(x.shape, x.device)
            # x = torch.cat((x, grid), dim=-1)
            x = torch.cat((x, grid), dim=1)

        x = x.permute(0, 2, 3, 1)
        q = self.q0(x)
        x = torch.cat((x, q), dim=-1) # cat product terms
        x = torch.cat((x, self.fc0(x)), dim=-1)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        xo = torch.cat((x1, x2), dim=1) # [batch, width, x, y] -> [batch, 2*width, x, y]
        xo = self.prod0(xo)
        if self.skip_connection:
            x = x + xo
        else: x = xo
        # x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        xo = torch.cat((x1, x2), dim=1)
        xo = self.prod1(xo)
        if self.skip_connection:
            x = x + xo
        else: x = xo
        # x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        xo = torch.cat((x1, x2), dim=1)
        xo = self.prod2(xo)
        if self.skip_connection:
            x = x + xo
        else: x = xo
        # x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        xo = torch.cat((x1, x2), dim=1)
        xo = self.prod3(xo)
        if self.skip_connection:
            x = x + xo
        else: x = xo
        # x = F.gelu(x)

        # x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        # x = F.gelu(x)
        # x = self.fc2(x)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[-2], shape[-1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)

    def extract(self, threshold, epoch):
        for (idx, quad) in enumerate(self.quads):
            if quad.prep_stage == False:
                continue
            proportion = quad.replace_previous(threshold)
            if_pass = proportion >= threshold
            if torch.all(if_pass):
                print(f'Extracted quadratic term {idx} at epoch {epoch}')
                print(quad.prod_indices)
