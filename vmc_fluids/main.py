import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import time

import global_defs
import var_state
import sampler
import net
import grid
import train
import evolutionEq
import tdvp
import stepper
import visualization
import mpi_wrapper
import util


def norm_fun(v, S):
    # norm_fun for the timesteps
    return v @ S @ v


# Initializing the net
initKey = 1
sampleKey = 1

mode_dict = {"fluidpaper": {"offset": jnp.ones(2) * 0.25, "dim": 2, "latent_space_name": "cos_dist", "mcmcbound": 0.25, "gridbound": 1., "symgrid": False, "evolution_type": "advection_paper"},
             "harmonicOsc": {"offset": jnp.ones(2) * 1, "dim": 2, "latent_space_name": "Gauss", "mcmcbound": 0.25, "gridbound": 8., "symgrid": True, "evolution_type": "advection_hamiltonian"},
             "harmonicOsc_diff": {"offset": jnp.array([1, 0, 1, 0, 1, 0]) * 1, "dim": 6, "latent_space_name": "Gauss", "mcmcbound": 0.25, "gridbound": 8., "symgrid": True, "evolution_type": "advection_hamiltonian_wDiss"},
             "diffusion": {"offset": jnp.zeros(8), "dim": 8, "latent_space_name": "Gauss", "mcmcbound": 0.25, "gridbound": 10., "symgrid": True, "evolution_type": "diffusion"},
             "diffusion_anisotropic": {"offset": jnp.zeros(12), "dim": 12, "latent_space_name": "Gauss", "mcmcbound": 0.25, "gridbound": 10., "symgrid": True, "evolution_type": "diffusion_anisotropic"}}
mode = "harmonicOsc_diff"
# mode = "diffusion"

"""
List of things that have to be set manually before starting a run:
- parameter nu of the student - t in BOTH (!!) sampler.py and net.py - starts with nu=2 atm.
- network specifications, whether to use both s and t, etc.
    - Diffusion: noAdd
    - harmonicOsc: DifferentAdd
- timestep:
    - Diffusion: dt = 1e-7, fixed, with increasing step size
    - harmonicOsc: dt=1e-4, fixed, with increasing step size
- blocks:
    - Diffusion:: 4, intmediate (dim//2)
    - harmonicOsc: 4, intmediate (,) <-- No extra layer

"""


dim = mode_dict[mode]["dim"]
offset = mode_dict[mode]["offset"]
mcmcbound = mode_dict[mode]["mcmcbound"]
gridbound = mode_dict[mode]["gridbound"]
symgrid = mode_dict[mode]["symgrid"]
latent_space_name = mode_dict[mode]["latent_space_name"]
evolution_type = mode_dict[mode]["evolution_type"]

# set up sampler
sampler = sampler.Sampler(dim=dim, numChains=30, name=latent_space_name, mcmc_info={"offset": offset, "bound": mcmcbound})

# set up variational state
print("Identifier -3")
vState = var_state.VarState(sampler, dim, initKey, 4, network_args={"intmediate": (dim // 2,) * 1, "offset": offset, "latentSpaceName": latent_space_name, "dim": dim})
print(f"Number of Model parameters: {vState.numParameters}")


# Some (old) sanity checks - can be removed
mynet = {"net": vState.net, "params": vState.params}
x_real = jnp.ones(dim)
print(mynet["params"])
z_latent, _ = mynet["net"].apply(mynet["params"], x_real, evaluate=False, inv=False)
x_real, _ = mynet["net"].apply(mynet["params"], z_latent, evaluate=False, inv=True)
print(z_latent)
print(x_real)

x_real = - jnp.ones(dim)
z_latent, jac = mynet["net"].apply(mynet["params"], x_real, evaluate=False, inv=False)
x_real, jac_inv = mynet["net"].apply(mynet["params"], z_latent, evaluate=False, inv=True)
print(z_latent)
print(x_real)

x_real = jnp.zeros(dim)
z_latent, jac = mynet["net"].apply(mynet["params"], x_real, evaluate=False, inv=False)
x_real, jac_inv = mynet["net"].apply(mynet["params"], z_latent, evaluate=False, inv=True)
print(z_latent)
print(x_real)


# Initializing the grid
if dim == 2:
    bounds = np.ones((dim,)) * gridbound
    n_gridpoints = 200
    grid = grid.Grid(bounds, n_gridpoints, sym=symgrid)
    integral = vState.integrate(grid)
    print("Integral value:", integral)

# time evolution
dt = 1e-7
tol = 1e-2
maxStep = 1e-2
myStepper = stepper.AdaptiveHeun(timeStep=dt, tol=tol, maxStep=maxStep)
# myStepper = stepper.FixedStepper(timeStep=dt, mode='Heun', maxStep=maxStep, increase_fac=1.3)
tdvpEq = tdvp.TDVP()
timings = util.Timings()
evolutionEq = evolutionEq.EvolutionEquation(dim=dim, name=evolution_type)
nSamplesTDVP = 10000
nSamplesObs = 10000

# data to learn a specific state
# std_dev = 1
# size = (1, 1000, dim)
# mode = "standard_normal"
# data, target_fun = train.gen_data(size, mode=mode, std=std_dev)
# net = train.train(vState, data, grid, lr=1e-3, batchsize=100, target_fun=target_fun, epoches=200)

wdir = "output/" + mode + f"/NsamplesTDVP{nSamplesTDVP}_NsamplesObs{nSamplesObs}_T10/"
wdir = "output/" + mode + f"/NsamplesTDVP{nSamplesTDVP}_NsamplesObs{nSamplesObs}/"
wdir = "output/" + "trash/" + mode + f"/NsamplesTDVP{nSamplesTDVP}_NsamplesObs{nSamplesObs}/"
if mpi_wrapper.rank == 0:
    try:
        os.makedirs(wdir)
    except OSError:
        print("Creation of the directory %s failed" % wdir)
    else:
        print("Successfully created the directory %s " % wdir)

t = 0
t_end = 5
plot_every = 1e2

if dim == 2:
    # visualization.plot_vectorfield(grid, evolutionEq)
    # plt.savefig(wdir + 'vectorfield.pdf')
    # plt.show()

    visualization.plot(vState, grid, proj=True)
    plt.savefig(wdir + f't_{t:.3f}.pdf')
    plt.show()

    # states = vState.sample(2000000)
    # visualization.plot_data(states, grid, title='Samples')
    # plt.show()


infos = {"times": [], "ev": [], "snr": [], "solver_res": [], "tdvp_error": [], "dist_params": []}
while t < t_end + dt:
    t1 = time.perf_counter()
    dp, dt, info = myStepper.step(0, tdvpEq, vState.get_parameters(), evolutionEq=evolutionEq, psi=vState, nSamplesTDVP=nSamplesTDVP, nSamplesObs=nSamplesObs, normFunction=norm_fun, timings=timings)
    vState.set_parameters(dp)
    infos["times"].append(t)

    print(f"t = {t:.3f}, dt = {dt:e}")
    print("\t Timings:")
    timings.print_timings()
    print(f"\t Total (in main.py): {time.perf_counter() - t1}")

    print("\t Data:")
    print(f"\t > Solver Residual = {tdvpEq.solverResidual}")
    print(f"\t > TDVP Error = {tdvpEq.tdvp_error}")
    print(f"\t > Integral 1sigma = {info['integral_1sigma']}")
    print(f"\t > Integral 0.5sigma = {info['integral_0.5sigma']}")
    print(f"\t > Integral 0.1sigma = {info['integral_0.1sigma']}")
    print(f"\t > Entropy = {info['entropy']}")
    print(f"\t > dist params = {vState.params['params']['dist_params']}")
    print(f"\t > Means = {info['x1']}")
    print(f"\t > Covar = {info['covar']}")

    for key in info.keys():
        if key not in infos.keys():
            infos[key] = []
        infos[key].append(info[key])
    infos["ev"].append(tdvpEq.ev)
    infos["snr"].append(tdvpEq.snr)
    infos["solver_res"].append(tdvpEq.solverResidual)
    infos["tdvp_error"].append(tdvpEq.tdvp_error)
    infos["dist_params"].append(vState.params['params']['dist_params'])

    if (t - dt) % plot_every >= t % plot_every or dt >= plot_every:
        if dim == 2:
            integral = vState.integrate(grid)
            print("Integral value:", integral)

            visualization.plot(vState, grid, proj=True)
            plt.savefig(wdir + f't_{t:.3f}.pdf')
            plt.show()

        print(vState.net.apply(vState.params, jnp.zeros(dim,), evaluate=False, inv=True)[0])

        # visualization.plot_line(vState, scale=10, fit=True, offset=offset)
        # plt.show()

    t = t + dt

util.store_infos(wdir, infos)
visualization.make_final_plots(wdir, infos)
plt.show()
