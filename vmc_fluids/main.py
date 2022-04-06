import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import matplotlib.pyplot as plt
import os

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


def norm_fun(v, S):
    # norm_fun for the timesteps
    return v @ S @ v


# Initializing the net
initKey = 1
sampleKey = 1

mode_dict = {"fluidpaper": {"offset": jnp.ones(2) * 0.25, "dim": 2, "latent_space_prob": sampler.cos_dist, "mcmcbound": 0.25, "gridbound": 1., "symgrid": False, "evolution_type": "advection_paper"},
             "harmonicOsc": {"offset": jnp.ones(2) * 1, "dim": 2, "latent_space_prob": sampler.unit_gauss, "mcmcbound": 0.25, "gridbound": 8., "symgrid": True, "evolution_type": "advection_hamiltonian"},
             "harmonicOsc_diff": {"offset": jnp.array([0, 1, 0.5, 0.5, 1, 0]) * 1, "dim": 6, "latent_space_prob": sampler.unit_gauss, "mcmcbound": 0.25, "gridbound": 8., "symgrid": True, "evolution_type": "advection_hamiltonian_wDiss"},
             "diffusion": {"offset": jnp.zeros(12), "dim": 12, "latent_space_prob": sampler.unit_gauss, "mcmcbound": 0.25, "gridbound": 10., "symgrid": True, "evolution_type": "diffusion"},
             "diffusion_anisotropic": {"offset": jnp.zeros(12), "dim": 12, "latent_space_prob": sampler.unit_gauss, "mcmcbound": 0.25, "gridbound": 10., "symgrid": True, "evolution_type": "diffusion_anisotropic"}}
mode = "diffusion"


wdir = "output/" + mode + "/"
if mpi_wrapper.rank == 0:
    try:
        os.makedirs(wdir)
    except OSError:
        print("Creation of the directory %s failed" % wdir)
    else:
        print("Successfully created the directory %s " % wdir)


dim = mode_dict[mode]["dim"]
offset = mode_dict[mode]["offset"]
mcmcbound = mode_dict[mode]["mcmcbound"]
gridbound = mode_dict[mode]["gridbound"]
symgrid = mode_dict[mode]["symgrid"]
latent_space_prob = mode_dict[mode]["latent_space_prob"]
evolution_type = mode_dict[mode]["evolution_type"]

# set up sampler
sampler = sampler.Sampler(dim=dim, numChains=30, latent_space_prob=latent_space_prob, mcmc_info={"offset": offset, "bound": mcmcbound})

# set up variational state
print("Identifier -3")
vState = var_state.VarState(sampler, dim, initKey, 3, pt_sym=False, intmediate=(2,) * 0, offset=offset)
print(f"Number of Model parameters: {vState.numParameters}")


# Some (old) sanity checks - can be removed
mynet = {"net": vState.net, "params": vState.params}
x_real = jnp.ones(dim)
z_latent, _ = mynet["net"].apply(mynet["params"], x_real)
x_real, _ = mynet["net"].apply(mynet["params"], z_latent, inv=True)
print(z_latent)
print(x_real)

x_real = - jnp.ones(dim)
z_latent, _ = mynet["net"].apply(mynet["params"], x_real)
x_real, _ = mynet["net"].apply(mynet["params"], z_latent, inv=True)
print(z_latent)
print(x_real)

x_real = jnp.zeros(dim)
z_latent, _ = mynet["net"].apply(mynet["params"], x_real)
x_real, _ = mynet["net"].apply(mynet["params"], z_latent, inv=True)
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
dt = 1e-2
tol = 1e-0
maxStep = 1e-2
myStepper = stepper.AdaptiveHeun(timeStep=dt, tol=tol, maxStep=maxStep)
# myStepper = stepper.FixedStepper(timeStep=dt, mode='Heun')
tdvpEq = tdvp.TDVP()
evolutionEq = evolutionEq.EvolutionEquation(dim=dim, name=evolution_type)
numSamples = 1500

# data to learn a specific state
# std_dev = 1
# size = (1, 1000, dim)
# mode = "standard_normal"
# data, target_fun = train.gen_data(size, mode=mode, std=std_dev)
# net = train.train(vState, data, grid, lr=1e-3, batchsize=100, target_fun=target_fun, epoches=200)

t = 0
t_end = 5
plot_every = 0.1

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


infos = {"times": [], "ev": [], "snr": [], "solver_res": [], "tdvp_error": []}
while t < t_end + dt:
    dp, dt, info = myStepper.step(0, tdvpEq, vState.get_parameters(), evolutionEq=evolutionEq, psi=vState, numSamples=numSamples, normFunction=norm_fun)
    vState.set_parameters(dp)
    infos["times"].append(t)
    print(f"t = {t:.3f}, dt = {dt:e}")
    print(f"\t > Solver Residual = {tdvpEq.solverResidual}")
    print(f"\t > TDVP Error = {tdvpEq.tdvp_error}")
    print(f"\t > Entropy = {info['entropy']}")
    print(f"\t > Means = {info['x1']}")
    print(f"\t > Covar (1st line) = {info['covar'][0, :]}")

    for key in info.keys():
        if key not in infos.keys():
            infos[key] = []
        infos[key].append(info[key])
    infos["ev"].append(tdvpEq.ev)
    infos["snr"].append(tdvpEq.snr)
    infos["solver_res"].append(tdvpEq.solverResidual)
    infos["tdvp_error"].append(tdvpEq.tdvp_error)

    if t % plot_every >= (t + dt) % plot_every or dt >= plot_every:
        if dim == 2:
            integral = vState.integrate(grid)
            print("Integral value:", integral)
            # visualization.plot(vState, grid)
            # plt.show()

            visualization.plot(vState, grid, proj=True)
            plt.savefig(wdir + f't_{t:.3f}.pdf')
            # plt.show()

        print(vState.net.apply(vState.params, jnp.zeros(dim,), inv=True))

        # visualization.plot_line(vState, scale=10, fit=True, offset=offset)
        # plt.show()

    t = t + dt

visualization.make_final_plots(wdir, infos)
