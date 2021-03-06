import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import matplotlib.pyplot as plt
from matplotlib import cm

from functools import partial
import os
import warnings

import sampler
import mpi_wrapper
import visualization
import util


def latent_space_dist_paper(x, offset):
    r = jnp.min(jnp.array([1, 4 * jnp.sqrt(jnp.sum((x - offset)**2))]))
    return jnp.log(0.5 * (1 + jnp.cos(jnp.pi * r)))


def plot_samples(coords, lim=6):
    plt.figure()
    plt.hist2d(coords[:, 0], coords[:, 1], bins=100, range=[[-lim, lim], [-lim, lim]], cmap=cm.coolwarm)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.show()


def _velocity_field_hamiltonian(coord, evolParams):
    """returns dx/dt = p, dp/dt = -x"""
    def H(x, coupled=False):
        if coupled:
            xs = x[0::2]
            ps = x[1::2]
            return jnp.pi * (evolParams["m"] * evolParams["omega"]**2 / 2 * jnp.sum((xs - jnp.roll(xs, 1))**2)
                             + jnp.sum(ps**2) / (2. * evolParams["m"])
                             + evolParams["lam"] * jnp.sum(xs**4))
        else:
            return (evolParams["m"] * evolParams["omega"]**2 / 2 * jnp.sum(x[0::2]**2)
                    + jnp.sum(x[1::2]**2) / (2. * evolParams["m"])
                    + evolParams["lam"] * jnp.sum(x[0::2]**4))
    grads = jax.grad(H)(coord)
    mat = jnp.kron(jnp.eye(coord.shape[0] // 2), jnp.array([[0, 1], [-1, 0]]))
    return mat @ grads


def _velocity_field_fluiddynpaper(coord, parameters):
    x, y = coord[0], coord[1]
    return jnp.array([-jnp.sin(jnp.pi * x)**2 * jnp.sin(2 * jnp.pi * y) * jnp.cos(jnp.pi * parameters["t"] / parameters["T"]),
                      jnp.sin(jnp.pi * y)**2 * jnp.sin(2 * jnp.pi * x) * jnp.cos(jnp.pi * parameters["t"] / parameters["T"])], dtype=jnp.float64)


def update_fun_phaseSpace(coord, parameters, vel_field, dt, key):
    mask = jnp.zeros_like(coord)
    mask = mask.at[1::2].set(1.)
    v_adv = vel_field(coord, parameters)
    v_diff = jnp.sqrt(2 * parameters["m"] * parameters["gamma"] * parameters["T"] / dt) * jax.random.normal(key, shape=coord.shape)
    v_damp = - parameters["gamma"] * coord
    return v_adv + v_diff * mask + v_damp * mask
    # return v_damp * mask


def update_fun_Diff(coord, parameters, vel_field, dt, key):
    v_diff = jnp.sqrt(2 / dt) * jax.random.normal(key, shape=coord.shape)
    return parameters["D"] * v_diff


def integrate_single_coord(coord, dt, parameters, vel_field, update_fun, key):
    keys = jax.random.split(key, 4)
    k1 = update_fun(coord, parameters, vel_field, dt / 6, keys[0])
    k2 = update_fun(coord + dt * 0.5 * k1, parameters, vel_field, dt / 3, keys[1])
    k3 = update_fun(coord + dt * 0.5 * k2, parameters, vel_field, dt / 3, keys[2])
    k4 = update_fun(coord + dt * k3, parameters, vel_field, dt / 6, keys[3])
    return coord + dt * (k1 + 2. * k2 + 2. * k3 + k4) / 6.


@partial(jax.jit, static_argnums=(3, 4))
def integrate(coords, dt, parameters, vel_field, update_fun, key):
    keys = jax.random.split(key, coords.shape[0])
    return jax.vmap(integrate_single_coord, in_axes=(0, None, None, None, None, 0))(coords, dt, parameters, vel_field, update_fun, keys)


if __name__ == "__main__":
    N_s = 10000
    case = "hamiltonian"
    sigma = 1e-0
    if case == "fluidpaper":
        dim = 2
        parameters = {"T": 10., "t": 0., "gamma": 1.0, "m": 1.0, "omega": 1.0, "lam": 0.0}
        vel_field = _velocity_field_fluiddynpaper
        mcmcbound = 0.25
        sampler = sampler.Sampler(dim=dim, numChains=30, latent_space_prob=latent_space_dist_paper, mcmc_info={"offset": offset, "bound": mcmcbound})
        coords = sampler(N_s)[0]
    elif case == "hamiltonian":
        dim = 6
        offset = jnp.array([0, 1, 0.5, 0.5, 1, 0])
        offset = jnp.array([1, 0, 1, 0, 1, 0])
        # offset = jnp.array([1, 0])
        parameters = {"T": 10, "t": 0., "gamma": 1, "m": 1.0, "omega": 1.0, "lam": 0.0}
        update_fun = update_fun_phaseSpace
        vel_field = _velocity_field_hamiltonian
        coords = sigma * jax.random.normal(jax.random.PRNGKey(0), (N_s, dim)) + offset
    elif case == "diffusion":
        dim = 6
        offset = jnp.zeros(dim)
        parameters = {"T": 10., "t": 0., "D": 1}
        update_fun = update_fun_Diff
        vel_field = None
        coords = sigma * jax.random.normal(jax.random.PRNGKey(0), (N_s, dim)) + offset

    t_end = 12
    dt = 1e-2
    t = 0
    plot_every = 1e2
    key = jax.random.PRNGKey(0)

    wdir = "output/exact_dyn/"
    wdir += case + f"/Nsamples{N_s}_T10.0/"
    if mpi_wrapper.rank == 0:
        try:
            os.makedirs(wdir)
        except OSError:
            print("Creation of the directory %s failed" % wdir)
        else:
            print("Successfully created the directory %s " % wdir)

    def mc_integral(coords, lim=1):
        return jnp.sum(jnp.linalg.norm(coords, axis=-1) < lim) / coords.shape[0]

    infos = {"x1": [], "times": [], "covar": [], "integral_1sigma": [], "integral_0.5sigma": [], "integral_0.1sigma": []}
    while t < t_end:
        key, key_to_use = jax.random.split(key)

        infos["times"].append(t)
        infos["x1"].append(jnp.mean(coords, axis=0))
        infos["covar"].append(jnp.cov(coords.T, ddof=0))
        infos["integral_1sigma"].append(mc_integral(coords, lim=jnp.sqrt(parameters["T"])))
        infos["integral_0.5sigma"].append(mc_integral(coords, lim=0.5 * jnp.sqrt(parameters["T"])))
        infos["integral_0.1sigma"].append(mc_integral(coords, lim=0.1 * jnp.sqrt(parameters["T"])))

        coords = integrate(coords, dt, parameters, vel_field, update_fun, key_to_use)
        print(f"\t covar = {infos['covar'][-1]}")
        print(f"\t integral_1sigma = {infos['integral_1sigma'][-1]}")
        print(f"\t integral_0.5sigma = {infos['integral_0.5sigma'][-1]}")
        print(f"\t integral_0.1sigma = {infos['integral_0.1sigma'][-1]}")

        t += dt
        parameters["t"] = t

        print(f"\t t = {t}")
        if t % plot_every > (t + dt) % plot_every:
            plot_samples(coords)

    visualization.make_final_plots(wdir, infos)
    util.store_infos(wdir, infos)
    plt.show()
