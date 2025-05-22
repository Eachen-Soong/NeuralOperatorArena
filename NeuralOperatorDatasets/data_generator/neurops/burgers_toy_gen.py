from random_fields import GaussianRF
import torch
import numpy as np
from einops import repeat


def burg_system_batch(u_batch, mu, nu, n_modes, smooth=False):
    u_batch = torch.tensor(u_batch, dtype=torch.float32)
    batch_size, N_x = u_batch.shape
    k = repeat(2 * torch.pi * torch.fft.rfftfreq(N_x, d=dx), 'c -> b c', b=batch_size)

    u_hat = torch.fft.rfft(u_batch, dim=-1)
    # print(u_hat.shape)
    k_tensor = torch.tensor(k[:, :n_modes], dtype=torch.float32)

    u_hat_x = 1j * k_tensor * u_hat[:, :n_modes]
    u_hat_xx = -k_tensor**2 * u_hat[:, :n_modes]

    u_x = torch.fft.irfft(u_hat_x, n=N_x, dim=-1).real
    u_xx = torch.fft.irfft(u_hat_xx, n=N_x, dim=-1).real

    u_t = -mu * torch.einsum('b c, b c -> b c', u_batch , u_x) + nu * u_xx
    u_t = u_t.unsqueeze(-2)
    print(u_t.shape)
    if smooth:
        kernel_size = 5
        smoother = torch.nn.Conv1d(in_channels=1, out_channels=1 ,kernel_size=kernel_size, padding=kernel_size//2)
        smoother.weight = torch.nn.Parameter(torch.ones_like(smoother.weight)/kernel_size)
        u_t = smoother(u_t)
    return u_t.squeeze()

# def burg_system(u, mu, nu, n_modes):
#     # Ensure u is a 1D array
#         u = np.squeeze(np.array(u))
#         N_x = u.shape[-1]
#         k = 2 * np.pi * np.fft.rfftfreq(N_x, d=dx)
#         k = k[:n_modes]

#         # Spatial derivative in the Fourier domain
#         u_hat = np.fft.rfft(u)
#         u_hat_x = 1j * k * u_hat[:n_modes]
#         u_hat_xx = -k**2 * u_hat[:n_modes]

#         # Switching in the spatial domain
#         u_x = np.fft.irfft(u_hat_x, n=N_x).real
#         u_xx = np.fft.irfft(u_hat_xx, n=N_x).real

#         # ODE resolution
#         u_t = -mu * u * u_x + nu * u_xx
#         return u_t

mu = 1
nu = 0.01  # kinematic viscosity coefficient
dx = 0.01
N_x = 1024
n_modes = 768
n_data = 2048

random_generator = GaussianRF(1, N_x, alpha=2.5, tau=7)    

u = random_generator.sample(N=n_data)
print(u.shape)
# ut = burg_system(u, mu, nu, n_modes)
ut = burg_system_batch(u, mu, nu, n_modes, True)


data = {'x': u.detach().numpy(), 'y': ut.detach().numpy()}
# # Save the solutions to a local file
file_path = "./data/burgers_ut"
# torch.save(data, file_path)
np.save(file_path, data)

import matplotlib.pyplot as plt

plt.plot(np.linspace(0, 1, u[0].shape[0]), u[0].detach().numpy())
plt.plot(np.linspace(0, 1, ut[0].shape[0]), ut[0].detach().numpy())
# plt.plot(np.linspace(0, 1, u.shape[-1]), u[0])
# plt.plot(np.linspace(0, 1, ut.shape[-1]), ut*3)

plt.show()