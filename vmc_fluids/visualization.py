import matplotlib.pyplot as plt
from matplotlib import cm

import jax
import jax.numpy as jnp
import numpy as np
import h5py
import flax.linen as nn
from dataclasses import dataclass
from functools import partial

import global_defs


# plotting of probabilities
def plot(vState, grid, z_lim=None, proj=False, fun=None):
    if fun != None:
        real_space_probs = jax.vmap(fun)(grid.coords)
    else:
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
        # ax.set_zlabel('z')
        ax.set_zlim(0, 0.15)
        if z_lim != None:
            ax.set_zlim([0, z_lim])

    ax.set_title('Model')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # plt.axis('off')
    plt.tight_layout()


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


def plot_diff(vState, grid, target_fun, fun=None):
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


def plot_data_diff(vState, grid, data, fun=None):
    if fun != None:
        real_space_probs = jax.vmap(fun)(grid.coords)
    else:
        real_space_probs = vState._evaluate_net_on_batch_jitd(grid.coords[None, ...], vState.params)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    real_space_probs = jnp.exp(real_space_probs).reshape((grid.n_gridpoints, grid.n_gridpoints))

    hist, _, _ = np.histogram2d(data[0, :, 0], data[0, :, 1], bins=grid.n_gridpoints, range=grid.range - grid.widths / 2, density=True)
    # The samples are confined to the range. If samples fall outside the range they are disregarded.
    # This overestimates the probability inside the range, since we set density to True.
    # Solution: Multiply with the integral of the function and divide by the integral of the histogram.
    # This leads to a new histogram which is not normalized on the grid.
    hist = hist * jnp.sum(real_space_probs) / jnp.sum(hist)
    diff = real_space_probs - hist

    print("integral over difference", grid.bin_area * jnp.sum(diff))
    print("integral over data", grid.bin_area * jnp.sum(hist))
    print("integral over model", grid.bin_area * jnp.sum(real_space_probs))

    ax.plot_surface(grid.meshgrid[0], grid.meshgrid[1], diff, cmap=cm.coolwarm)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Model - Target Function')

    return grid.bin_area * jnp.sum(jnp.abs(diff))


def plot_data(data, grid, title='Data'):
    hist, _, _ = np.histogram2d(data[0, :, 0], data[0, :, 1], bins=grid.n_gridpoints, range=grid.range - grid.widths / 2, density=True)
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


def plot_vectorfield(grid, evolutionEq, t=0):
    vecfield_fun = evolutionEq.eqParams[evolutionEq.name]["vel_field"]
    eqParams = evolutionEq.eqParams[evolutionEq.name]["params"]
    assert grid.dim == 2
    [x, y] = grid.meshgrid
    field = jax.vmap(vecfield_fun, in_axes=(None, 0, None))(eqParams, grid.coords, t).reshape(-1, 2)
    u, v = field[:, 0], field[:, 1]
    plt.quiver(x, y, u, v)


def make_final_plots(wdir, infos):
    if "x1" in infos.keys():
        plt.figure()
        for i, data in enumerate(np.array(infos["x1"]).T):
            idx = i // 2
            if i % 2 == 0:
                plt.plot(np.array(infos["times"]), data, label=rf'$\langle x_{idx} \rangle$')
            else:
                plt.plot(np.array(infos["times"]), data, label=rf'$\langle p_{idx} \rangle$')
        plt.grid()
        plt.ylabel(r'$\langle O \rangle$')
        plt.xlabel(r'$t$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(wdir + 'observables.pdf')

    if "covar" in infos.keys():
        plt.figure()
        covar = np.array(infos["covar"])
        for i in range(covar.shape[-1]):
            plt.plot(np.array(infos["times"]), covar[:, i, i], label=rf'$\sigma_{i}^2$ - INN')
        # plt.plot(np.array(infos["times"]), 1 + 2 * np.array(infos["times"]), label='Exact')
        plt.grid()
        plt.ylabel(r'$\sigma^2$')
        plt.xlabel(r'$t$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(wdir + 'variance.pdf')

    if "covar" in infos.keys():
        plt.figure()
        covar = np.array(infos["covar"])
        for i in range(covar.shape[-1]):
            for j in range(covar.shape[-1]):
                if i > j:
                    plt.plot(np.array(infos["times"]), covar[:, i, j], label=rf'$Cov(x_{i}, x_{j})$ - INN')
        plt.plot(np.array(infos["times"]), 0 * np.array(infos["times"]), label='Exact')
        plt.grid()
        plt.ylabel(r'$\sigma^2$')
        plt.xlabel(r'$t$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(wdir + 'covariance.pdf')

    if "entropy" in infos.keys():
        plt.figure()
        plt.plot(np.array(infos["times"]), np.array(infos["entropy"]), label=rf'INN')
        plt.plot(np.array(infos["times"]), 0.5 * np.array(infos["x1"]).shape[-1] * jnp.log(2 * jnp.pi * jnp.exp(1) * (1 + 2 * np.array(infos["times"]))), label='Exact')
        plt.grid()
        plt.ylabel(r'Entropy')
        plt.xlabel(r'$t$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(wdir + 'entropy.pdf')

    if "dist_params" in infos.keys():
        plt.figure()
        for i, dist_params in enumerate(np.array(infos["dist_params"]).T):
            plt.plot(np.array(infos["times"]), np.exp(dist_params) + 1, label=rf'Latent parameter {i}')
        plt.grid()
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\nu$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(wdir + 'latent_parameter.pdf')

    if "x3" in infos.keys():
        dim_of_interest = 0
        plt.figure()
        for key in infos.keys():
            if key[0] == "x":  # and int(key[1]) % 2 != 0:
                plt.plot(np.array(infos["times"]), np.array(infos[key])[:, dim_of_interest], label=fr'$\langle x^{key[1]}\rangle$')
        plt.grid()
        plt.ylabel(r'$\sigma^2$')
        plt.xlabel(r'$t$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(wdir + 'moments.pdf')

    if "solver_res" in infos.keys():
        plt.figure()
        plt.plot(np.array(infos["times"]), np.array(infos["solver_res"]))
        plt.grid()
        plt.ylabel('Residual')
        plt.xlabel(r'$t$')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(wdir + 'residual.pdf')

    if "tdvp_error" in infos.keys():
        plt.figure()
        plt.plot(np.array(infos["times"]), np.array(infos["tdvp_error"]))
        plt.grid()
        plt.ylabel('TDVP Error')
        plt.xlabel(r'$t$')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(wdir + 'tdvp_error.pdf')

    if "times" in infos.keys():
        plt.figure()
        plt.plot(np.array(infos["times"])[:-1], np.diff(np.array(infos["times"])))
        plt.grid()
        plt.ylabel(r'$\Delta t$')
        plt.xlabel(r'$t$')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(wdir + 'timesteps.pdf')

    if "ev" in infos.keys():
        plt.figure()
        plt.plot(np.array(infos["times"]), np.array(infos["ev"]))
        plt.grid()
        plt.ylabel('EV')
        plt.xlabel(r'$t$')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(wdir + 'eigenvalue.pdf')

    if "snr" in infos.keys():
        plt.figure()
        plt.plot(np.array(infos["times"]), np.array(infos["snr"]))
        plt.grid()
        plt.ylabel('SNR')
        plt.xlabel(r'$t$')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(wdir + 'snr.pdf')

    if any("integral" in key for key in infos.keys()) or False:
        plt.figure()
        plt.plot(np.array(infos["times"]), np.array(infos["integral_1sigma"]))
        plt.plot(np.array(infos["times"]), np.array(infos["integral_0.5sigma"]))
        plt.plot(np.array(infos["times"]), np.array(infos["integral_0.1sigma"]))
        plt.grid()
        plt.ylabel('Integral value')
        plt.yscale('log')
        plt.xlabel(r'$t$')
        plt.tight_layout()
        plt.savefig(wdir + 'integral.pdf')


if __name__ == "__main__":
    wdir = "/home/moritz/Insync/moritz.reh@gmail.com/Google Drive/10_Studium/02_MASTER/02_MasterArbeit/03_WorkResults/02_Results/200_GitHubRepos_Code/vmc_fluids/vmc_fluids/output/harmonicOsc_diff/NsamplesTDVP10000_NsamplesObs10000/"
    infos = h5py.File(wdir + "infos.hdf5", "r")
    make_final_plots(wdir, infos)
