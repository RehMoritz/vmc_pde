import matplotlib.pyplot as plt
from matplotlib import cm

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from dataclasses import dataclass

import global_defs

from jax.tree_util import tree_flatten, tree_unflatten

import net


class VarState:

    def __init__(self, sampleKey, dim, *args, **kwargs):
        self.sampleKey = jax.random.PRNGKey(sampleKey)
        self.dim = dim
        self.net, self.params = self.init_net(*args, **kwargs)

        self.paramShapes = [(p.size, p.shape) for p in tree_flatten(self.params)[0]]
        self.netTreeDef = jax.tree_util.tree_structure(self.params)
        self.numParameters = jnp.sum(jnp.array([p.size for p in tree_flatten(self.params)[0]]))

        self._evaluate_net_on_batch_jitd = global_defs.pmap_for_my_devices(jax.vmap(self.real_space_prob, in_axes=(0, None)), in_axes=(0, None))
        self._grads_params_jitd = global_defs.pmap_for_my_devices(jax.vmap(jax.value_and_grad(lambda coords, params: - self.real_space_prob(coords, params), argnums=1), in_axes=(0, None)), in_axes=(0, None))
        self._grads_coords_jitd = global_defs.pmap_for_my_devices(jax.vmap(jax.value_and_grad(lambda coords, params: self.real_space_prob(coords, params), argnums=(0, 1)), in_axes=(0, None)), in_axes=(0, None))
        self._hessian_coords_jitd = global_defs.pmap_for_my_devices(jax.vmap(jax.jacrev(jax.jacfwd(lambda coords, params: self.real_space_prob(coords, params), argnums=0), argnums=0), in_axes=(0, None)), in_axes=(0, None))
        self._latent_coords_jitd = global_defs.pmap_for_my_devices(jax.vmap(lambda coords, params: self.net.apply(params, coords, inv=True), in_axes=(0, None)), in_axes=(0, None))
        self._flatten_tree_jitd = global_defs.pmap_for_my_devices(jax.vmap(self.flatten_tree))

    def __call__(self, coords, mode="eval", avg=False):

        if mode == "eval":
            value = self._evaluate_net_on_batch_jitd(coords, self.params)
            if avg:
                return jnp.mean(value, axis=(0, 1))
            else:
                return value

        if mode == "costfun":
            """
            Used for training by example - Minimizes the cross entropy of samples <-> encoded function.
            """
            value, grad = self._grads_params_jitd(coords, self.params)
            if avg:
                return jnp.mean(value, axis=(0, 1)), self.average_tree(grad)
            else:
                return value, grad

        if mode == "eval_coordgrads":
            value, (coord_grads, param_grads) = self._grads_coords_jitd(coords, self.params)
            if avg:
                ValueError("Not implemented.")
                # return jnp.mean(value, axis=(0, 1)), self.average_tree(grad)
            else:
                return value, coord_grads, self._flatten_tree_jitd(param_grads)

    def hessian(self, coords):
        import warnings
        warnings.warn("Computing full Hessian of the coordinates.")
        return self._hessian_coords_jitd(coords, self.params)

    def real_space_prob(self, x, params):
        assert(x.shape[0] == self.dim)
        z = self.net.apply(params, x, inv=False)
        p_latent_log = self.latent_space_prob(z)
        jac = jax.jacfwd(self.net.apply, argnums=1)(params, x, inv=False)
        jac_det = jnp.linalg.det(jac)
        return p_latent_log + jnp.log(jnp.abs(jac_det))
        # return p_latent_log + jnp.log(jac_det)

    def latent_space_prob(self, x, sigma=1):
        return -0.5 * jnp.sum(x**2) / sigma**2 - jnp.log(jnp.sqrt(2 * jnp.pi * sigma**2)) * self.dim

    def sample(self, numSamples):
        self.sampleKey, key = jax.random.split(self.sampleKey)
        latent_space_samples = jax.random.normal(key, (1, numSamples, self.dim))
        return self._latent_coords_jitd(latent_space_samples, self.params)

    def average_tree(self, tree, axis=(0, 1)):
        flat, tree = jax.tree_util.tree_flatten(tree)
        avg_flat = []
        for leaf in flat:
            avg_flat.append(jnp.mean(leaf, axis=axis))
        return jax.tree_util.tree_unflatten(tree, avg_flat)

    def integrate(self, grid):
        real_space_probs = self._evaluate_net_on_batch_jitd(grid.coords[None, ...], self.params)
        integral = jnp.sum(grid.bin_area * jnp.exp(real_space_probs))
        return integral

    # Below are only parameter utilities such as setting & getting parameters and unrolling update vectors as trees
    def set_parameters(self, p_new):
        pTreeShape = []
        start = 0
        for s in self.paramShapes:
            pTreeShape.append(p_new[start:start + s[0]].reshape(s[1]))
            start += s[0]

        self.params = jax.tree_util.tree_unflatten(self.netTreeDef, pTreeShape)

    def get_parameters(self):
        return self.flatten_tree(self.params)

    def flatten_tree(self, tree):
        flat, _ = jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: x.ravel(), tree))
        return jnp.concatenate(flat).ravel()

    # plotting of probabilities
    def plot(self, grid):
        real_space_probs = self._evaluate_net_on_batch_jitd(grid.coords[None, ...], self.params)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        real_space_probs = jnp.exp(real_space_probs).reshape((grid.n_gridpoints, grid.n_gridpoints))
        ax.plot_surface(grid.meshgrid[0], grid.meshgrid[1], real_space_probs, cmap=cm.coolwarm)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Model')

    def plot_diff(self, grid, target_fun):
        real_space_probs = self._evaluate_net_on_batch_jitd(grid.coords[None, ...], self.params)
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

    def plot_data(self, data, grid):
        bound = grid.bounds[0]
        hist, xedges, yedges = np.histogram2d(data[0, :, 0], data[0, :, 1], bins=grid.n_gridpoints, range=[[-bound, bound], [-bound, bound]])
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(grid.meshgrid[0], grid.meshgrid[1], hist, cmap=cm.coolwarm)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Data')

    def init_net(self, widths, key, **kwargs):
        key = jax.random.PRNGKey(key)
        inds_up = []
        inds_down = []
        for width in widths:
            ind_up = jax.random.choice(key, width, shape=(int(width / 2),), replace=False)
            ind_down = jnp.setdiff1d(jnp.arange(width), ind_up)
            inds_up.append(ind_up)
            inds_down.append(ind_down)
        mynet = net.INN(inds_up, inds_down, **kwargs, widths=widths)
        # mynet = net.SanityINN(inds_up, inds_down, widths=widths)
        params = mynet.init(key, jnp.zeros(widths[0]))
        return mynet, params
