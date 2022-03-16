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


def latent_space_dist_paper(x, offset):
    r = jnp.min(jnp.array([1, 4 * jnp.sqrt(jnp.sum((x - offset)**2))]))
    return jnp.log(0.5 * (1 + jnp.cos(jnp.pi * r)))


# Initializing the net
initKey = 1
sampleKey = 1

mode_dict = {"fluidpaper": {"offset": jnp.ones(2) * 0.25, "dim": 2, "latent_space_prob": latent_space_dist_paper, "mcmcbound": 0.25, "gridbound": 1., "symgrid": False, "evolution_type": "advection_paper"},
             "harmonicOsc": {"offset": jnp.ones(2) * 1, "dim": 2, "latent_space_prob": sampler.unit_gauss, "mcmcbound": 0.25, "gridbound": 8., "symgrid": True, "evolution_type": "advection_hamiltonian"},
             "diffusion": {"offset": jnp.zeros(2), "dim": 2, "latent_space_prob": sampler.unit_gauss, "mcmcbound": 1, "gridbound": 10., "symgrid": True, "evolution_type": "diffusion"}}
mode = "fluidpaper"

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
print("Identifier 0")
vState = var_state.VarState(sampler, dim, initKey, 20, pt_sym=False, intmediate=(2,) * 1, offset=offset)
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
if dim < 4:
    bounds = np.ones((dim,)) * gridbound
    n_gridpoints = 200
    grid = grid.Grid(bounds, n_gridpoints, sym=symgrid)
    integral = vState.integrate(grid)
    print("Integral value:", integral)

# time evolution
dt = 3e-3
tol = 1e-2
maxStep = dt
myStepper = stepper.AdaptiveHeun(timeStep=dt, tol=tol, maxStep=maxStep)
# myStepper = stepper.FixedStepper(timeStep=dt, mode='Heun')
tdvpEq = tdvp.TDVP()
evolutionEq = evolutionEq.EvolutionEquation(name=evolution_type)
numSamples = 5000

# visualization.plot_vectorfield(grid, evolutionEq)
# plt.show()

# states = vState.sample(2000000)
# visualization.plot_data(states, grid, title='Samples')
# plt.show()

t = 0
t_end = 5.0
plot_every = 0.1

visualization.plot(vState, grid, proj=True)
plt.savefig(wdir + f't_{t:.3f}.pdf')
plt.show()

infos = {"times": [], "ev": [], "snr": [], "solver_res": [], "tdvp_error": []}
while t < t_end + dt:
    dp, dt, info = myStepper.step(0, tdvpEq, vState.get_parameters(), evolutionEq=evolutionEq, psi=vState, numSamples=numSamples, normFunction=norm_fun)
    vState.set_parameters(dp)
    infos["times"].append(t)
    print(f"t = {t:.3f}, dt = {dt:e}")
    print(f"\t > Solver Residual = {tdvpEq.solverResidual}")
    print(f"\t > TDVP Error = {tdvpEq.tdvp_error}")
    # print(f"\t > SNR = {tdvpEq.snr}")

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

        print(vState.net.apply(vState.params, jnp.zeros(2,), inv=True))

        # visualization.plot_line(vState, scale=10, fit=True, offset=offset)
        # plt.show()

    t = t + dt

plt.figure()
plt.plot(np.array(infos["times"]), np.array(infos["variance"]), label='INN')
plt.plot(np.array(infos["times"]), 1 + 2 * np.array(infos["times"]), label='Exact')
plt.grid()
plt.ylabel(r'$\sigma^2$')
plt.xlabel(r'$t$')
plt.legend()

plt.figure()
plt.plot(np.array(infos["times"]), np.array(infos["x2"]), label=r'$\langle x^2\rangle$')
plt.plot(np.array(infos["times"]), np.array(infos["x4"]), label=r'$\langle x^4\rangle$')
plt.plot(np.array(infos["times"]), np.array(infos["x6"]), label=r'$\langle x^6\rangle$')
plt.grid()
plt.ylabel(r'$\sigma^2$')
plt.xlabel(r'$t$')
plt.legend()

plt.figure()
plt.plot(np.array(infos["times"]), np.array(infos["solver_res"]))
plt.grid()
plt.ylabel('Residual')
plt.xlabel(r'$t$')
plt.yscale('log')

plt.figure()
plt.plot(np.array(infos["times"]), np.array(infos["tdvp_error"]))
plt.grid()
plt.ylabel('TDVP Error')
plt.xlabel(r'$t$')
plt.yscale('log')

plt.figure()
plt.plot(np.array(infos["times"])[:-1], np.diff(np.array(infos["times"])))
plt.grid()
plt.ylabel(r'$\Delta t$')
plt.xlabel(r'$t$')
plt.yscale('log')

plt.figure()
plt.plot(np.array(infos["times"]), np.array(infos["ev"]))
plt.grid()
plt.ylabel('EV')
plt.xlabel(r'$t$')
plt.yscale('log')

plt.figure()
plt.plot(np.array(infos["times"]), np.array(infos["snr"]))
plt.grid()
plt.ylabel('SNR')
plt.xlabel(r'$t$')
plt.yscale('log')
plt.show()

# data to learn a specific state
# std_dev = 1
# size = (1, 1000, dim)
# mode = "standard_normal"
# data, target_fun = train.gen_data(size, mode=mode, std=std_dev)
# net = train.train(vState, data, grid, lr=1e-3, batchsize=100, target_fun=target_fun, epoches=200)
