import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from lbm_solver import lbm_solver
import random

def write_data(filename, vel_time, times, mask):
    """
    Write velocity fields and time step data to an HDF5 file.
    """
    with h5py.File(filename, 'w') as f:
        f.create_dataset("velocities", data=vel_time)
        f.create_dataset("times", data=times)
        f.create_dataset("mask", data = mask)
        
def load_data(filename):
    """
    Load velocity fields and time step data from an HDF5 file.
    """
    with h5py.File(filename, 'r') as f:
        vel_time = f["velocities"][:]
        times = f["times"][:]
        mask = f["mask"][:]
    return vel_time, times, mask

def single_data_gen(config: dict, output_dic = 'output', seed = 10):
    """
    Generate a single simulation dataset and save it to an HDF5 file.

    Args:
        config (dict): Configuration for the LBM simulation.
        output_dic (str): Output directory where the HDF5 file will be saved. Default is 'output'.
        seed (int): Random seed for reproducibility. Default is 10.

    Returns:
        None
    """
    random.seed(seed)
    solver = lbm_solver(config)
    vel_time, times, mask = solver.solve()
    Re = int(solver.bc_value[0][0] * solver.mask_size/ solver.nu)
    shape_str = '_'.join([f'{shape}{count}' for shape, count in config['mask']['shape_counts'].items()])

    if not os.path.exists(output_dic):
        os.makedirs(output_dic)

    filename = os.path.join(output_dic, f'Re_{int(Re)}_shape_{shape_str}_seed_{seed}.h5')
    write_data(filename, vel_time, times, mask)
    print(f"Simulation data saved to: {filename}")


def animate_velocity_magnitude(vel_time, times):
    """
    Animate the velocity magnitude and add a simple progress indicator.
    """
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Initialize the image
    vel_mag = np.sqrt(vel_time[0, :, :, 0]**2 + vel_time[0, :, :, 1]**2)
    im = ax.imshow(vel_mag.T, cmap="plasma", origin="lower", extent=[0, vel_time.shape[1], 0, vel_time.shape[2]])
    plt.colorbar(im, label="Speed Magnitude", aspect=5, shrink=0.35)
    ax.set_title(f"Time Step: {times[0]:.2f}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Update function for animation
    def update(frame):
        # Calculate progress
        progress = (frame + 1) / len(vel_time) * 100
        print(f"Generating frame {frame + 1}/{len(vel_time)} ({progress:.1f}%)", end="\r")
        
        # Update image data
        vel_mag = np.sqrt(vel_time[frame, :, :, 0]**2 + vel_time[frame, :, :, 1]**2)
        im.set_data(vel_mag.T)
        ax.set_title(f"Time: {times[frame]:.2f}")
        return im

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(vel_time), interval=100, blit=False)

    # Save animation
    print("Saving animation...", end="")
    ani.save("velocity_magnitude.gif", writer='pillow')
    print("\nAnimation saved successfully!")

if __name__ == '__main__':
    # Load data
    vel_time, times = load_data("velocity_fields/velocity_field_test.h5")
    
    # Check data consistency
    if len(vel_time) != len(times):
        raise ValueError("The number of velocity fields and timesteps must be the same.")
    
    # Animate velocity magnitude
    animate_velocity_magnitude(vel_time, times)