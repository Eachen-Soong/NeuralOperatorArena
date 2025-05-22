from ns_2d_vorticity_spetral_solver import *
from random_force_generator import *
from random_fields import *
import os
import h5py
import argparse

"""
    Data Generator for the F-FNO datasets reimplemented 
    (including Torus-Li and Torus-Kolmogorov)
    with a more organized code structure, 
    where the force generator is decoupled with the NS spetral solver,
    and the random force generator function not reliant on pseudo-randomness
    (the original version passes the same seed to get_random_force(...) each time it's called).
"""
name_device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(name_device)
print('Generating on device:\t', name_device)

# configs
parser = argparse.ArgumentParser()
parser.add_argument('--s', type=int, default=256, help='resolution of the grid')
parser.add_argument('--N', type=int, default=20, help='number of solutions to generate')
parser.add_argument('--varying_force', type=bool, default=False, help='whether to vary the force')
parser.add_argument('--various_visc', type=bool, default=False, help='whether to vary the viscosity')
parser.add_argument('--record_steps', type=int, default=200, help='number of snapshots to record')
parser.add_argument('--bsize', type=int, default=20, help='batch size')
parser.add_argument('--total_time', type=float, default=50.0, help='total time')
parser.add_argument('--delta_t', type=float, default=1e-4, help='delta_t')
parser.add_argument('--force_type', type=Force, default=Force.li, help='type of force')
parser.add_argument('--visc', type=float, default=1e-4, help='viscosity')
parser.add_argument('--min_visc', type=float, default=1e-5, help='viscosity')
parser.add_argument('--max_visc', type=float, default=1e-4, help='viscosity')
parser.add_argument('--scaling', type=float, default=0.1, help='force scaling')
parser.add_argument('--t_scaling', type=float, default=0.2, help='changing rate of force with t')
parser.add_argument('--cycles', type=int, default=4, help='number of wave numbers of random forces')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--path', type=str, default='', help='save_path')
args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed + 1234)

s = args.s
N = args.N
varying_force = args.varying_force
various_visc = args.various_visc
record_steps = args.record_steps
bsize = args.bsize
total_time = args.total_time
delta_t = args.delta_t
scaling=args.scaling
t_scaling=args.t_scaling
cycles = args.cycles
path = args.path
if path == '':
    dirctory = './data/ns_exp1/'
    name = 'ns_'
    if varying_force:
        name += 'varying_'
    name += f'{args.force_type}_'
    if various_visc:
        name += f'visc_in_{args.min_visc}-{args.max_visc}_'
    else:
        name += f'visc={args.visc}_'
    name += f's={s}_steps={record_steps}_T={total_time}_dt={delta_t}_N={N}.h5'
    path = dirctory + name

if os.path.dirname(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

data_f = h5py.File(path, 'w')

# Solver, viscosity and force term Definition
solver = NS_2D_Vorticity_Solver(resolution=s, device=device)

if various_visc:
    visc = np.random.rand(bsize) * (args.max_visc - args.min_visc) + args.min_visc
else: visc = args.visc

force_generator=None
force_fetcher=None
force=None
if args.force_type == Force.random:
    force_generator = Random_Force_Generator(bsize, s, cycles=cycles, scaling=scaling, t_scaling=t_scaling, device=device)
    if varying_force:
        force_fetcher = force_generator.get_force
        print("force_fetcher: ", type(force_fetcher))
    else:
        force = force_generator.get_force()    
else:
    force = non_random_force(args.force_type, s=s)

if force is not None:
    force = force.to(device)

#Set up 2d GRF with covariance parameters
GRF = GaussianRF(2, s, alpha=2.5, tau=7, device=device)

t = torch.linspace(0, 1, s+1, device=device)
t = t[0:-1]

X,Y = torch.meshgrid(t, t, indexing='ij')

t0 =default_timer()

c = 0

data_f.create_dataset("a", (N, s, s), np.float32)
data_f.create_dataset("u", (N, s, s, record_steps), np.float32)
if varying_force:
    data_f.create_dataset('f', (N, s, s, record_steps), np.float32)
else:
    data_f.create_dataset('f', (N, s, s), np.float32)
data_f.create_dataset('mu', (N,), np.float32)
data_f.create_dataset('dt', (1,), np.float32)
data_f.create_dataset('total_time', (1,), np.float32)
data_f['dt'][0] = delta_t
data_f['total_time'][0] = total_time

for j in range(N//bsize):
    #Sample random feilds
    w0 = GRF.sample(bsize)

    # Solve NS
    if varying_force:
        print("force_fetcher: ", type(force_fetcher))
        sol, f = solver.solve_navier_stokes_2d(w0, visc, total_time, delta_t, record_steps, get_force=force_fetcher)
        data_f['a'][c:(c+bsize), ...] = w0.cpu().numpy()
        data_f['u'][c:(c+bsize), ...] = sol
        data_f['f'][c:(c+bsize), ...] = f
        data_f['mu'][c:(c+bsize)] = visc

        c += bsize
        t1 = default_timer()
        print(j, c, t1-t0)

        t0 = t1

    else:
        sol, _ = solver.solve_navier_stokes_2d_const_force(w0, visc, total_time, delta_t, record_steps, force=force)
        data_f['a'][c:(c+bsize), ...] = w0.cpu().numpy()
        data_f['u'][c:(c+bsize), ...] = sol
        data_f['f'][c:(c+bsize), ...] = force.cpu().numpy()
        data_f['mu'][c:(c+bsize)] = visc

        c += bsize
        t1 = default_timer()
        print(j, c, t1-t0)

        t0 = t1
