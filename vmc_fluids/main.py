import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import matplotlib.pyplot as plt

import global_defs
import var_state
import net
import grid
import train
import evolutionEq
import tdvp
import stepper

# Initializing the net
dim = 2
initKey = 0
sampleKey = 0
vState = var_state.VarState(sampleKey, dim, (dim, dim), initKey, pt_sym=False)

# Initializing the grid
# bound = 3
# bounds = np.ones((dim,)) * bound
# n_gridpoints = 20
# grid = grid.Grid(bounds, n_gridpoints)

# Some (old) sanity checks - can be removed
mynet = {"net": vState.net, "params": vState.params}
x_real = jnp.ones(dim)
z_latent = mynet["net"].apply(mynet["params"], x_real)
x_real = mynet["net"].apply(mynet["params"], z_latent, inv=True)
print(z_latent)
print(x_real)

x_real = - jnp.ones(dim)
z_latent = mynet["net"].apply(mynet["params"], x_real)
x_real = mynet["net"].apply(mynet["params"], z_latent, inv=True)
print(z_latent)
print(x_real)

# integral = vState.integrate(grid)
# print("Integral value:", integral)

# data to learn a specific state
std_dev = 4
size = (1, 1000, dim)
mode = "standard_normal"
data, target_fun = train.gen_data(size, mode=mode, std=std_dev)

# time evolution
myStepper = stepper.Euler(timeStep=1e-2)
tdvpEq = tdvp.TDVP()
evolutionEq = evolutionEq.EvolutionEquation(name="diffusion")
numSamples = 600

t = 0
infos = {"times": []}
while t < 2:
    dp, dt, info = myStepper.step(0, tdvpEq, vState.get_parameters(), evolutionEq=evolutionEq, psi=vState, numSamples=numSamples)
    infos["times"].append(t)
    t = t + dt
    vState.set_parameters(dp)

    for key in info.keys():
        if key not in infos.keys():
            infos[key] = []
        infos[key].append(info[key])

    print(f"t = {t}")

    if np.abs(t - int(t)) < 0.5 * dt and False:
        vState.plot(grid)
        plt.show()

plt.plot(np.array(infos["times"]), np.array(infos["variance"]), label='INN')
plt.plot(np.array(infos["times"]), 1 + 2 * np.array(infos["times"]), label='Exact')
plt.grid()
plt.ylabel(r'$\sigma^2$')
plt.xlabel(r'$t$')
plt.legend()
plt.show()

# net = train.train(vState, data, grid, lr=1e-3, batchsize=100, target_fun=target_fun, epoches=200)
