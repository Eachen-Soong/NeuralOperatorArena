import torch
import numpy as np
from einops import rearrange, repeat
from tqdm import tqdm
import math


class NS_2D_Vorticity_Solver:
    """
    Defines 1 step of the updation of vorticity, with periodic BC, 
    through pseudo spetral method, crank nicolson rk2.
    """
    def __init__(self, resolution, device='cuda'):
        self.resolution = resolution
        self.k_max = math.floor(resolution / 2)
        # Negative Laplacian in Fourier space
        # Wavenumbers in y-direction
        self.k_y = torch.cat((
            torch.arange(start=0, end=self.k_max, step=1, device=device),
            torch.arange(start=-self.k_max, end=0, step=1, device=device)),
            0).repeat(resolution, 1)
        # Wavenumbers in x-direction
        self.k_x = self.k_y.transpose(0, 1)
        self.lap = 4 * (math.pi**2) * (self.k_x**2 + self.k_y**2)
        self.lap[0, 0] = 1.0
        self.dealias = torch.unsqueeze(
            torch.logical_and(
                torch.abs(self.k_y) <= (2.0 / 3.0) * self.k_max,
                torch.abs(self.k_x) <= (2.0 / 3.0) * self.k_max
            ).float(), 0)

    def step(self, w0, visc, delta_t, force=None):
        """calculates 1 step forward
        
        Parameters
        ----------
        w0 : torch.Tensor
            Initial vorticity field.
        visc : float
            Viscosity (1/Re).
        T : float
            Final time.
        delta_t : float
            Internal time-step for solve (descrease if blow-up).
        record_steps : int
            Number of in-time snapshots to record.
        """
        # Grid size - must be power of 2
        assert w0.shape[-1] == w0.shape[-2] and w0.shape[-1] == self.resolution, "w0 resolution doesn't match with the solver!"

        # Initial vorticity to Fourier space
        w_h = torch.fft.fftn(w0, dim=[1, 2], norm='backward')

        # Forcing to Fourier space
        if force == None:
            f_h = 0
        else:
            f_h = torch.fft.fftn(force, dim=[-2, -1], norm='backward')
            # If same forcing for the whole batch
            if len(f_h.shape) < len(w_h.shape):
                f_h = rearrange(f_h, '... -> 1 ...')

        if isinstance(visc, np.ndarray):
            visc = torch.from_numpy(visc).to(w0.device)
            visc = repeat(visc, 'b -> b m n', m=self.resolution, n=self.resolution)
            lap = repeat(self.lap, 'm n -> b m n', b=w0.shape[0])        

        # Stream function in Fourier space: solve Poisson equation
        psi_h = w_h / lap

        # Velocity field in x-direction = psi_y
        q = psi_h.clone()
        q_real_temp = q.real.clone()
        q.real = -2 * math.pi * self.k_y * q.imag
        q.imag = 2 * math.pi * self.k_y * q_real_temp
        q = torch.fft.ifftn(q, dim=[1, 2], norm='backward').real

        # Velocity field in y-direction = -psi_x
        v = psi_h.clone()
        v_real_temp = v.real.clone()
        v.real = 2 * math.pi * self.k_x * v.imag
        v.imag = -2 * math.pi * self.k_x * v_real_temp
        v = torch.fft.ifftn(v, dim=[1, 2], norm='backward').real

        # Partial x of vorticity
        w_x = w_h.clone()
        w_x_temp = w_x.real.clone()
        w_x.real = -2 * math.pi * self.k_x * w_x.imag
        w_x.imag = 2 * math.pi * self.k_x * w_x_temp
        w_x = torch.fft.ifftn(w_x, dim=[1, 2], norm='backward').real

        # Partial y of vorticity
        w_y = w_h.clone()
        w_y_temp = w_y.real.clone()
        w_y.real = -2 * math.pi * self.k_y * w_y.imag
        w_y.imag = 2 * math.pi * self.k_y * w_y_temp
        w_y = torch.fft.ifftn(w_y, dim=[1, 2], norm='backward').real

        # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
        F_h = torch.fft.fftn(q * w_x + v * w_y,
                            dim=[1, 2], norm='backward')

        # Dealias
        F_h *= self.dealias

        # Cranck-Nicholson update
        factor = 0.5 * delta_t * visc * lap
        num = -delta_t * F_h + delta_t * f_h + (1.0 - factor) * w_h
        w_h = num / (1.0 + factor)

        w = torch.fft.ifftn(w_h, dim=[1, 2], norm='backward').real

        return w.cpu().numpy()

    def solve_navier_stokes_2d(self, w0, visc, T, delta_t, record_steps, get_force):
        """
        Parameters
        ----------
        w0 : torch.Tensor
            Initial vorticity field.
        visc : float
            Viscosity (1/Re).
        T : float
            Final time.
        delta_t : float
            Internal time-step for solve (descrease if blow-up).
        record_steps : int
            Number of in-time snapshots to record.
        get_force : function float -> tensor [s, s]
            get force for each time step
        """
        # Number of steps to final time
        steps = math.ceil(T / delta_t)

        # Initial vorticity to Fourier space
        w_h = torch.fft.fftn(w0, dim=[1, 2], norm='backward')

        # Record solution every this number of steps
        record_time = math.floor(steps / record_steps)

        lap = self.lap
        if isinstance(visc, np.ndarray):
            visc = torch.from_numpy(visc).to(w0.device)
            visc = repeat(visc, 'b -> b m n', m=self.resolution, n=self.resolution)
            lap = repeat(self.lap, 'm n -> b m n', b=w0.shape[0])

        # Saving solution and time
        sol = torch.zeros(*w0.size(), record_steps, device=w0.device)
        fs = torch.zeros(*w0.size(), record_steps, device=w0.device)

        # Record counter
        c = 0
        # Physical time
        t = 0.0

        for j in tqdm(range(steps)):
            # Stream function in Fourier space: solve Poisson equation
            psi_h = w_h / lap

            # Velocity field in x-direction = psi_y
            q = psi_h.clone()
            q_real_temp = q.real.clone()
            q.real = -2 * math.pi * self.k_y * q.imag
            q.imag = 2 * math.pi * self.k_y * q_real_temp
            q = torch.fft.ifftn(q, dim=[1, 2], norm='backward').real

            # Velocity field in y-direction = -psi_x
            v = psi_h.clone()
            v_real_temp = v.real.clone()
            v.real = 2 * math.pi * self.k_x * v.imag
            v.imag = -2 * math.pi * self.k_x * v_real_temp
            v = torch.fft.ifftn(v, dim=[1, 2], norm='backward').real

            # Partial x of vorticity
            w_x = w_h.clone()
            w_x_temp = w_x.real.clone()
            w_x.real = -2 * math.pi * self.k_x * w_x.imag
            w_x.imag = 2 * math.pi * self.k_x * w_x_temp
            w_x = torch.fft.ifftn(w_x, dim=[1, 2], norm='backward').real

            # Partial y of vorticity
            w_y = w_h.clone()
            w_y_temp = w_y.real.clone()
            w_y.real = -2 * math.pi * self.k_y * w_y.imag
            w_y.imag = 2 * math.pi * self.k_y * w_y_temp
            w_y = torch.fft.ifftn(w_y, dim=[1, 2], norm='backward').real

            # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
            F_h = torch.fft.fftn(q * w_x + v * w_y,
                                dim=[1, 2], norm='backward')

            # Dealias
            F_h *= self.dealias

            # Forcing to Fourier space
            f = get_force(t)
            f_h = torch.fft.fftn(f, dim=[-2, -1], norm='backward')
            # If same forcing for the whole batch
            if len(f_h.shape) < len(w_h.shape):
                f_h = rearrange(f_h, '... -> 1 ...')

            # Cranck-Nicholson update
            factor = 0.5 * delta_t * visc * lap
            num = -delta_t * F_h + delta_t * f_h + (1.0 - factor) * w_h
            w_h = num / (1.0 + factor)

            # Update real time (used only for recording)
            t += delta_t

            if (j + 1) % record_time == 0:
                # Solution in physical space
                w = torch.fft.ifftn(w_h, dim=[1, 2], norm='backward').real
                if w.isnan().any().item():
                    raise ValueError('NaN values found.')

                # Record solution and time
                sol[..., c] = w
                fs[..., c] = f

                c += 1

        return sol.cpu().numpy(), fs.cpu().numpy()


    def solve_navier_stokes_2d_const_force(self, w0, visc, T, delta_t, record_steps, force=None):
        """Constant Force: we can calculate f_h just once to accelerate!
        
        Parameters
        ----------
        w0 : torch.Tensor
            Initial vorticity field.
        visc : float
            Viscosity (1/Re).
        T : float
            Final time.
        delta_t : float
            Internal time-step for solve (descrease if blow-up).
        record_steps : int
            Number of in-time snapshots to record.
        """
        # Number of steps to final time
        steps = math.ceil(T / delta_t)

        # Initial vorticity to Fourier space
        w_h = torch.fft.fftn(w0, dim=[1, 2], norm='backward')

        # Forcing to Fourier space
        if force == None:
            f_h = 0
        else:
            f_h = torch.fft.fftn(force, dim=[-2, -1], norm='backward')
            # If same forcing for the whole batch
            if len(f_h.shape) < len(w_h.shape):
                f_h = rearrange(f_h, '... -> 1 ...')

        # Record solution every this number of steps
        record_time = math.floor(steps / record_steps)
        lap = self.lap
        if isinstance(visc, np.ndarray):
            visc = torch.from_numpy(visc).to(w0.device)
            visc = repeat(visc, 'b -> b m n', m=self.resolution, n=self.resolution)
            lap = repeat(lap, 'm n -> b m n', b=w0.shape[0])

        # Saving solution and time
        sol = torch.zeros(*w0.size(), record_steps, device=w0.device)
        sol_t = torch.zeros(record_steps, device=w0.device)

        # Record counter
        c = 0
        # Physical time
        t = 0.0

        for j in tqdm(range(steps)):
            # Stream function in Fourier space: solve Poisson equation
            psi_h = w_h / lap

            # Velocity field in x-direction = psi_y
            q = psi_h.clone()
            q_real_temp = q.real.clone()
            q.real = -2 * math.pi * self.k_y * q.imag
            q.imag = 2 * math.pi * self.k_y * q_real_temp
            q = torch.fft.ifftn(q, dim=[1, 2], norm='backward').real

            # Velocity field in y-direction = -psi_x
            v = psi_h.clone()
            v_real_temp = v.real.clone()
            v.real = 2 * math.pi * self.k_x * v.imag
            v.imag = -2 * math.pi * self.k_x * v_real_temp
            v = torch.fft.ifftn(v, dim=[1, 2], norm='backward').real

            # Partial x of vorticity
            w_x = w_h.clone()
            w_x_temp = w_x.real.clone()
            w_x.real = -2 * math.pi * self.k_x * w_x.imag
            w_x.imag = 2 * math.pi * self.k_x * w_x_temp
            w_x = torch.fft.ifftn(w_x, dim=[1, 2], norm='backward').real

            # Partial y of vorticity
            w_y = w_h.clone()
            w_y_temp = w_y.real.clone()
            w_y.real = -2 * math.pi * self.k_y * w_y.imag
            w_y.imag = 2 * math.pi * self.k_y * w_y_temp
            w_y = torch.fft.ifftn(w_y, dim=[1, 2], norm='backward').real

            # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
            F_h = torch.fft.fftn(q * w_x + v * w_y,
                                dim=[1, 2], norm='backward')

            # Dealias
            F_h *= self.dealias

            # Cranck-Nicholson update
            factor = 0.5 * delta_t * visc * lap
            num = -delta_t * F_h + delta_t * f_h + (1.0 - factor) * w_h
            w_h = num / (1.0 + factor)

            # Update real time (used only for recording)
            t += delta_t

            if (j + 1) % record_time == 0:
                # Solution in physical space
                w = torch.fft.ifftn(w_h, dim=[1, 2], norm='backward').real
                if w.isnan().any().item():
                    raise ValueError('NaN values found.')

                # Record solution and time
                sol[..., c] = w
                sol_t[c] = t

                c += 1

        return sol.cpu().numpy(), force.cpu().numpy()


