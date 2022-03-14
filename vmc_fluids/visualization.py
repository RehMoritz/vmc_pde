import matplotlib.pyplot as plt
from matplotlib import cm

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from dataclasses import dataclass
from functools import partial

import global_defs


# plotting of probabilities
def plot(vState, grid, z_lim=None, proj=False):
    real_space_probs = vState._evaluate_net_on_batch_jitd(grid.coords[None, ...], vState.params)
    fig = plt.figure(figsize=(6, 6))
    if proj:
        ax = plt.axes()
        real_space_probs = jnp.exp(real_space_probs).reshape((grid.n_gridpoints, grid.n_gridpoints))
        ax.pcolormesh(grid.meshgrid[0], grid.meshgrid[1], real_space_probs, cmap=cm.coolwarm)
    else:
        ax = plt.axes(projection='3d')
        real_space_probs = jnp.exp(real_space_probs).reshape((grid.n_gridpoints, grid.n_gridpoints))
        ax.plot_surface(grid.meshgrid[0], grid.meshgrid[1], real_space_probs, cmap=cm.coolwarm)
        ax.set_zlabel('z')
        if z_lim != None:
            ax.set_zlim([0, z_lim])

    ax.set_title('Model')
    ax.set_xlabel('x')
    ax.set_ylabel('y')


def plot_line(vState, scale=1, n_gridpoints=100, fit=False, offset=np.zeros(2)):
    gridpoints = np.zeros((n_gridpoints, vState.dim))
    gridpoints[:, 0] = np.arange(-scale, scale, 2 * scale / n_gridpoints)
    real_space_probs = vState(gridpoints[None, ...] + offset)

    plt.figure()
    plt.plot(gridpoints[:, 0], jnp.exp(real_space_probs)[0])
    plt.grid()
    plt.xlabel(r'Interpolation $\lambda$')
    plt.ylabel(r'Probability')
    plt.yscale('log')
    plt.title('1D-Interpolation of the Model probability')

    if fit:
        from scipy.optimize import curve_fit

        def gauss(x, a, x0, sigma):
            return a / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - x0)**2 / (2 * sigma**2))

        popt, pcov = curve_fit(gauss, gridpoints[:, 0], jnp.exp(real_space_probs)[0], p0=[1, 0, 1])
        plt.plot(gridpoints[:, 0], gauss(gridpoints[:, 0], *popt))


def plot_diff(vState, grid, target_fun):
    real_space_probs = vState._evaluate_net_on_batch_jitd(grid.coords[None, ...], vState.params)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    real_space_probs = jnp.exp(real_space_probs).reshape((grid.n_gridpoints, grid.n_gridpoints))

    target_vals = target_fun(grid.coords).reshape((grid.n_gridpoints, grid.n_gridpoints))
    diff = real_space_probs - target_vals
    print("integral over difference", grid.bin_area * jnp.sum(diff))

    ax.plot_surface(grid.meshgrid[0], grid.meshgrid[1], diff, cmap=cm.coolwarm)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Model - Target Function')


def plot_data(data, grid, title='Data'):
    bound = grid.bounds[0]
    hist, xedges, yedges = np.histogram2d(data[0, :, 0], data[0, :, 1], bins=grid.n_gridpoints, range=grid.range)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(grid.meshgrid[0], grid.meshgrid[1], hist, cmap=cm.coolwarm)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)


def plot_data_1D(data, grid, title='Data'):
    plt.figure()
    plt.hist(data[:, :, 0].reshape(-1), bins=200, range=grid.range[0], density=True)
    # x = grid.meshgrid[0, 0]
    # plt.plot(x, jnp.exp(jax.vmap(unit_gauss, in_axes=(0, None))(x, 1)))
    plt.show()


def plot_vectorfield(grid, evolutionEq, t=0, params={}):
    vecfield_fun = evolutionEq.eqParams[evolutionEq.name]["vel_field"]
    eqParams = evolutionEq.eqParams[evolutionEq.name]["params"]
    assert grid.dim == 2
    [x, y] = grid.meshgrid
    field = jax.vmap(vecfield_fun, in_axes=(None, 0, None))(params, grid.coords, t).reshape(-1, 2)
    u, v = field[:, 0], field[:, 1]
    plt.quiver(x, y, u, v)
