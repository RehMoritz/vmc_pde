import matplotlib.pyplot as plt
from matplotlib import cm

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from dataclasses import dataclass
from functools import partial

import global_defs

from jax.tree_util import tree_flatten, tree_unflatten

import net


class VarState:

    def __init__(self, sampler, dim, *args, **kwargs):
        self.sampler = sampler
        self.dim = dim
        self.net, self.params = self.init_net(*args, **kwargs)

        self.paramShapes = [(p.size, p.shape) for p in tree_flatten(self.params)[0]]
        self.netTreeDef = jax.tree_util.tree_structure(self.params)
        self.numParameters = jnp.sum(jnp.array([p.size for p in tree_flatten(self.params)[0]]))

        self._evaluate_net_on_batch_jitd = global_defs.pmap_for_my_devices(jax.vmap(self.real_space_prob, in_axes=(0, None)), in_axes=(0, None))
        self._grads_params_jitd = global_defs.pmap_for_my_devices(jax.vmap(jax.value_and_grad(lambda coords, params: - self.real_space_prob(coords, params), argnums=1), in_axes=(0, None)), in_axes=(0, None))
        self._grads_coords_jitd = global_defs.pmap_for_my_devices(jax.vmap(jax.value_and_grad(lambda coords, params: self.real_space_prob(coords, params), argnums=(0, 1)), in_axes=(0, None)), in_axes=(0, None))
        self._hessian_coords_jitd = global_defs.pmap_for_my_devices(jax.vmap(jax.jacrev(jax.jacfwd(lambda coords, params: self.real_space_prob(coords, params), argnums=0), argnums=0), in_axes=(0, None)), in_axes=(0, None))
        self._hessiandiag_coords_jitd = global_defs.pmap_for_my_devices(jax.vmap(self._hvp, in_axes=(0, None)), in_axes=(0, None))
        self._latent_coords_jitd = global_defs.pmap_for_my_devices(jax.vmap(lambda coords, params: self.net.apply(params, coords, inv=True)[0], in_axes=(0, None)), in_axes=(0, None))
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
            """
            return the function value, the gradients with respect to the input and the parameters
            """
            value, (coord_grads, param_grads) = self._grads_coords_jitd(coords, self.params)
            if avg:
                ValueError("Not implemented.")
                # return jnp.mean(value, axis=(0, 1)), self.average_tree(grad)
            else:
                return value, coord_grads, self._flatten_tree_jitd(param_grads)

    def hessian(self, coords):
        # import warnings
        # warnings.warn("Computing full Hessian of the coordinates.")
        return self._hessian_coords_jitd(coords, self.params)

    def hessian_diag(self, coords):
        import warnings
        warnings.warn("Hessian diag is not correctly implemented.")
        return self._hessiandiag_coords_jitd(coords, self.params)

    def _hvp(self, coords, params):
        f = jax.grad(partial(lambda x, y: self.real_space_prob(y, x), params))
        return jax.jvp(f, (coords,), (jnp.ones_like(coords),))[1]

    def real_space_prob(self, x, params):
        z, log_jac = self.net.apply(params, x, inv=False)
        p_latent_log = self.sampler.latent_space_prob(z, self.sampler.mcmc_info["offset"])
        return p_latent_log + log_jac

    def sample(self, numSamples):
        latent_space_samples = self.sampler(numSamples)
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

    def init_net(self, key, depth, **kwargs):
        key = jax.random.PRNGKey(key)
        inds_up = []
        inds_down = []
        for _ in range(depth):
            key, use_key = jax.random.split(key)
            ind_up = jax.random.choice(use_key, self.dim, shape=(int(self.dim / 2),), replace=False)
            ind_down = jnp.setdiff1d(jnp.arange(self.dim), ind_up)
            inds_up.append(ind_up)
            inds_down.append(ind_down)
        mynet = net.INN(inds_up, inds_down, **kwargs)
        # mynet = net.SanityINN(inds_up, inds_down)
        params = mynet.init(key, jnp.zeros(self.dim))
        return mynet, params
