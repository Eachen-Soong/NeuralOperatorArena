"""
@author: Zongyi Li
modified by Eachen Soong to adapt to this code base
"""
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from .modules import SpectralConv2d, MLP, SpectralConvProd2d

################################################################
# fourier layer
################################################################
class FNO_2D(nn.Module):
    def __init__(self, in_dim, appended_dim, out_dim, modes1, modes2, width, use_position=True):
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
        self.appended_dim = appended_dim
        self.out_dim = out_dim
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic
        self.use_position = use_position

        self.p = nn.Linear(in_dim + appended_dim, self.width) # input channel is 3: (a(x, y), x, y)
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
        self.padding = 9 # pad the domain if input is non-periodic
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
    def __init__(self, in_dim, appended_dim, out_dim, modes1, modes2, width, use_position=True, skip_connections=None):
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
        self.appended_dim = appended_dim
        self.out_dim = out_dim
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic
        self.use_position = use_position
        self.skip_connections = [True, True, False, False]

        self.p = nn.Linear(in_dim + appended_dim, self.width) # input channel is 3: (a(x, y), x, y)
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