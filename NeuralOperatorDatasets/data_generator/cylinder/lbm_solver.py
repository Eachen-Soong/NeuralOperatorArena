import h5py
import os
import numpy as np
import taichi as ti
import taichi.math as tm

from dataclasses import dataclass, field
from typing import List, Dict

from generate_mask import generate_mask


ti.init(arch=ti.gpu)

    
@ti.data_oriented
class lbm_solver:
    def __init__(self, config: dict):
        """
        Initialize the Lattice Boltzmann Method (LBM) solver.

        Args:
            config (dict): Simulation configuration, including:
                - `name` (str): Simulation name (default: "test").
                - `nx`, `ny` (int): Grid dimensions (default: 0).
                - `nu` (float): Kinematic viscosity (default: 0.0).
                - `time.delta_t` (float): Time step size (default: 1.0).
                - `boundary.types` (list[int]): Boundary conditions: [left, top, right, bottom]; 0 -> Dirichlet, 1 -> Neumann.
                - `boundary.values` (list[list[float]]): Velocity values for Dirichlet boundaries: [u, v] for each boundary
                - `mask.size` (int): Shape size in pixels (default: 10).
                - `mask.shape_counts` (dict): Number of shapes in the mask.Supported shape types: `'circle'`, `'triangle'`, `'square'`, `'pentagon'`. Example: `{'circle': 5, 'square': 3}` means 5 circles and 3 squares.
                - `time.interval` (int): Data output interval (default: 10).
                - `time.max_timestep` (int): Maximum simulation steps (default: 1000).

        Example:
            config = {
                "name": "flow_simulation",
                "nx": 100,
                "ny": 100,
                "nu": 0.1,
                "boundary": {
                    "types": [0, 0, 1, 0],  # Dirichlet on left and right, Neumann on top and bottom.
                    "values": [[0.1, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]  # Velocity values for Dirichlet boundaries.
                },
                "mask": {
                    "size": 15,
                    "shape_counts": {"circle": 3, "square": 2}  # 3 circles and 2 squares on the mask.
                },
                "time": {
                    "delta_t": 0.5,
                    "interval": 20,
                    "max_timestep": 5000
                }
            }
        """
        self.name = config.get("name", "test")
        self.nx = config.get("nx", 0)
        self.ny = config.get("ny", 0)
        self.nu = config.get("nu", 0.0)  
        self.dt = config.get("time", {}).get("delta_t", 1.0)
        self.tau = 3.0 * self.nu / self.dt + 0.5
        self.inv_tau = 1.0 / self.tau
        self.rho = ti.field(float, shape=(self.nx, self.ny))
        self.vel = ti.Vector.field(2, float, shape=(self.nx, self.ny))
        self.mask = ti.field(float, shape=(self.nx, self.ny))
        self.f_old = ti.Vector.field(9, float, shape=(self.nx, self.ny))
        self.f_new = ti.Vector.field(9, float, shape=(self.nx, self.ny))
        # Weights for the D2Q9 lattice model.
        self.w = ti.types.vector(9, float)(4, 1, 1, 1, 1, 1 / 4, 1 / 4, 1 / 4, 1 / 4) / 9.0  # Lattice weights.
        # Discrete velocity vectors for the D2Q9 lattice model.
        self.e = ti.types.matrix(9, 2, int)(
            [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
             [1, 1], [-1, 1], [-1, -1], [1, -1]]
        )  # Velocity vectors.
        # Boundary condition types and values.
        boundary_config = config.get("boundary", {})
        self.bc_type = ti.field(int, 4)  # Field to store boundary condition types.
        self.bc_type.from_numpy(np.array(boundary_config.get("types", [0, 0, 0, 0]), dtype=np.int32))  # Set boundary types.
        self.bc_value = ti.Vector.field(2, float, shape=4)  # Field to store boundary condition values.
        self.bc_value.from_numpy(np.array(boundary_config.get("values", [[0.0, 0.0]] * 4), dtype=np.float32))  # Set boundary values.
        # Mask configuration for obstacles.
        mask_config = config.get("mask", {})
        self.mask_shape = mask_config.get("shape_counts", {})  # Dictionary of shape counts for the mask.
        self.mask_size = mask_config.get("size", 10)  # Size of each shape in pixels.
        # Time-related configuration.
        time_config = config.get("time", {})
        self.time_interval = time_config.get("interval", 10)  # Interval between data outputs.
        self.max_timestep = time_config.get("max_timestep", 1000)  # Maximum number of time steps for the simulation.


    @ti.func  # compute equilibrium distribution function
    def f_eq(self, i, j):
        eu = self.e @ self.vel[i, j]
        uv = tm.dot(self.vel[i, j], self.vel[i, j])
        return self.w * self.rho[i, j] * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * uv)

    def init(self):
        self.vel.fill(0)
        self.rho.fill(1)
        self.mask.fill(0)
        self.mask.from_numpy(generate_mask(self.nx, self.ny, self.mask_size, self.mask_shape))
        

    @ti.kernel
    def collide_and_stream(self):  # lbm core equation
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            for k in ti.static(range(9)):
                ip = i - self.e[k, 0]
                jp = j - self.e[k, 1]
                feq = self.f_eq(ip, jp)
                self.f_new[i, j][k] = (1 - self.inv_tau) * self.f_old[ip, jp][k] + feq[k] * self.inv_tau

    @ti.kernel
    def update_macro_var(self):  # compute rho u v
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.rho[i, j] = 0
            self.vel[i, j] = 0, 0
            for k in ti.static(range(9)):
                self.f_old[i, j][k] = self.f_new[i, j][k]
                self.rho[i, j] += self.f_new[i, j][k]
                self.vel[i, j] += tm.vec2(self.e[k, 0], self.e[k, 1]) * self.f_new[i, j][k]

            self.vel[i, j] /= self.rho[i, j]

    @ti.kernel
    def apply_bc(self):  # impose boundary conditions
        # left and right
        for j in range(1, self.ny - 1):
            # left: dr = 0; ibc = 0; jbc = j; inb = 1; jnb = j
            self.apply_bc_core(1, 0, 0, j, 1, j)

            # right: dr = 2; ibc = nx-1; jbc = j; inb = nx-2; jnb = j
            self.apply_bc_core(1, 2, self.nx - 1, j, self.nx - 2, j)

        # top and bottom
        for i in range(self.nx):
            # top: dr = 1; ibc = i; jbc = ny-1; inb = i; jnb = ny-2
            self.apply_bc_core(1, 1, i, self.ny - 1, i, self.ny - 2)

            # bottom: dr = 3; ibc = i; jbc = 0; inb = i; jnb = 1
            self.apply_bc_core(1, 3, i, 0, i, 1)

        # cylindrical obstacle
        # Note: for cuda backend, putting 'if statement' inside loops can be much faster!
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.mask[i, j] == 1:
                self.vel[i, j] = 0, 0  # velocity is zero at solid boundary
                inb = i
                jnb = j
                if self.mask[i+1, j] == 0 or self.mask[i+1, j+1] == 0 or self.mask[i+1, j-1] == 0:
                    inb = i + 1
                else:
                    inb = i - 1
                if self.mask[i, j+1] == 0 or self.mask[i+1, j+1] == 0 or self.mask[i-1, j+1] == 0:
                    jnb = j + 1
                else:
                    jnb = j - 1
                self.apply_bc_core(0, 0, i, j, inb, jnb)

    @ti.func
    def apply_bc_core(self, outer, dr, ibc, jbc, inb, jnb):
        if outer == 1:  # handle outer boundary
            if self.bc_type[dr] == 0:
                self.vel[ibc, jbc] = self.bc_value[dr]

            elif self.bc_type[dr] == 1:
                self.vel[ibc, jbc] = self.vel[inb, jnb]

        self.rho[ibc, jbc] = self.rho[inb, jnb]
        self.f_old[ibc, jbc] = self.f_eq(ibc, jbc) - self.f_eq(inb, jnb) + self.f_old[inb, jnb]
        
    def solve(self):
        # Initialize simulation
        self.init()
        
        # Set up data storage
        timestep = 0
        vel_time = []  # Velocity fields
        times = []     # Time steps

        # Main loop
        threshold = self.max_timestep / 10  # Divergence check interval
        while timestep < self.max_timestep:  
            # Perform LBM steps
            for _ in range(self.time_interval):
                self.collide_and_stream()
                self.update_macro_var()
                self.apply_bc()

            # Calculate velocity field
            vel = self.vel.to_numpy()

            # Check for divergence
            if timestep > threshold:
                threshold += self.max_timestep / 10
                vel_magnitude = np.sqrt(vel[:, :, 0]**2 + vel[:, :, 1]**2)
                vel_max = np.max(vel_magnitude)
                if np.isnan(vel_max):
                    raise ValueError("Divergence detected! Simulation unstable.")

            # Save data
            vel_time.append(vel)
            timestep += self.time_interval
            times.append(timestep * self.dt)
            
        
        # Return results
        return np.array(vel_time), np.array(times), self.mask.to_numpy()


if __name__ == '__main__':
    # Define the LBM configuration
    import random
    random.seed(300)
    
    config = {
        "name": "flow_simulation",
        "nx": 801,
        "ny": 201,
        "nu": 0.05,
        "boundary": {
            "types": [0, 0, 1, 0],  # Dirichlet on left and right, Neumann on top and bottom.
            "values": [[0.1, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]  # Velocity values for Dirichlet boundaries.
        },
        "mask": {
            "size": 100,
            "shape_counts": {"circle": 1, "square": 1}  # 3 circles and 2 squares on the mask.
        },
        "time": {
            "delta_t": 1.0,
            "interval": 20,
            "max_timestep": 1000
        }
    }
    
    # Initialize the LBM solver
    solver = lbm_solver(config)
    
    Re = solver.bc_value[0][0] * solver.mask_size/ solver.nu
    print(f'Re:{Re}')

    # Run the simulation and get velocity fields and corresponding times
    vel_time, times, mask = solver.solve()
    
    from data_helper import write_data, animate_velocity_magnitude
    
    animate_velocity_magnitude(vel_time, times)
    
    write_data(f"velocity_field_{solver.name}.h5", vel_time, times, mask)
    
    
    