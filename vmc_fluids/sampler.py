import matplotlib.pyplot as plt
from matplotlib import cm

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from dataclasses import dataclass
from functools import partial

import global_defs
import mpi_wrapper

from jax.tree_util import tree_flatten, tree_unflatten

import net


def unit_gauss(x, offset, sigma=1e0):
    return - jnp.log(jnp.sqrt(2 * jnp.pi * sigma**2)) * x.shape[0] - 0.5 * jnp.sum((x - offset)**2) / sigma**2


def radial_update_prop(key, numChains, mcmc_info):
    keys = jax.random.split(key)
    r = jax.random.uniform(keys[0], shape=(numChains, 1))
    r = jnp.sqrt(r) * mcmc_info["bound"]
    phi = 2 * jnp.pi * jax.random.uniform(keys[1], shape=(numChains, 1))
    x = r * jnp.cos(phi)
    y = r * jnp.sin(phi)

    return jnp.concatenate((x, y), axis=-1) + mcmc_info["offset"]


@dataclass
class Sampler:
    key: int = 0
    numChains: int = 1
    dim: int = 2
    name: str = "Gauss"
    latent_space_prob: callable = unit_gauss
    updateProposer: callable = radial_update_prop
    mcmc_info: any = None

    def __post_init__(self):
        self.key = jax.random.PRNGKey(self.key)
        self.key = jax.random.split(self.key, mpi_wrapper.commSize)[mpi_wrapper.rank]
        self.key = jax.random.split(self.key, global_defs.device_count())[0]
        self.exact_samples = self.latent_space_prob in [unit_gauss]
        self.exact_sample_generator_dict = {"Gauss": jax.random.normal}
        self.states = None

        self._get_samples_jitd = {}

    def mc_init(self):
        self.key, key_to_use = jax.random.split(self.key)
        self.states = self.updateProposer(key_to_use, self.numChains, self.mcmc_info)
        self.states = self.states[None, ...]

    def __call__(self, numSamples, multipleOf=1):
        self.key, key_to_use = jax.random.split(self.key, 2)

        if not self.exact_samples:
            # We don't have an expression from which we can sample exactly - resort to rejection sampling
            if self.states == None:
                self.mc_init()
            numSamples = mpi_wrapper.distribute_sampling(numSamples, localDevices=global_defs.device_count(), numChainsPerDevice=np.lcm(self.numChains, multipleOf))

            if numSamples not in self._get_samples_jitd.keys():
                self._get_samples_jitd[numSamples] = global_defs.pmap_for_my_devices(partial(self._get_samples, numSamples=numSamples), in_axes=(None, 0))
            return self._get_samples_jitd[numSamples](key_to_use, self.states)

        else:
            return self.exact_sample_generator_dict[self.name](key_to_use, (1, numSamples, self.dim)) + self.mcmc_info["offset"][None, None, :]

    def _get_samples(self, key, states, numSamples=1000):

        def perform_mc_update(carry, x):
            # Generate update proposals
            newKeys = jax.random.split(carry[1], 3)
            newStates = self.updateProposer(newKeys[0], states.shape[0], self.mcmc_info)

            P = jax.vmap(lambda x, y: jnp.exp(self.latent_space_prob(x, self.mcmc_info["offset"]) - self.latent_space_prob(y, self.mcmc_info["offset"])))(newStates, carry[0])
            accepted = jax.random.bernoulli(newKeys[1], P).reshape((-1,))

            # Bookkeeping
            numProposed = carry[2] + len(newStates)
            numAccepted = carry[3] + jnp.sum(accepted)

            def update(acc, old, new):
                return jax.lax.cond(acc, lambda x: x[1], lambda x: x[0], (old, new))
                # return jax.lax.cond(True, lambda x: x[1], lambda x: x[0], (old, new))
            # Perform accepted updates
            carryStates = jax.vmap(update, in_axes=(0, 0, 0))(accepted, carry[0], newStates)

            return ((carryStates, newKeys[2], numProposed, numAccepted), carryStates)

        _, states = jax.lax.scan(perform_mc_update, (states, key, 0, 0), None, length=numSamples)
        return states.reshape(states.shape[0] * states.shape[1], states.shape[2])


if __name__ == "__main__":
    sampler = Sampler(latent_space_prob=unit_gauss, mcmc_info={"bound": 10}, numChains=1)
    states = sampler(2000000)
    # states = states[:, 100000:, :]
    print(states.shape)
    print(states)
    # exit()

    import matplotlib.pyplot as plt
    plt.hist2d(states[:, :, 0].reshape(-1), states[:, :, 1].reshape(-1), bins=200, range=[[-4, 4], [-4, 4]])
    plt.show()

    plt.hist(states[:, :, 0].reshape(-1), bins=200, range=[-4, 4], density=True)
    x = jnp.arange(-4, 4, 0.01)
    plt.plot(x, jnp.exp(jax.vmap(unit_gauss, in_axes=(0, None))(x, 1)))
    plt.show()

    def latent_space_dist_paper(x):
        r = jnp.min(jnp.array([1, 4 * jnp.sqrt(jnp.sum((x - offset)**2))]))
        return jnp.log(0.5 * (1 + jnp.cos(jnp.pi * r)))
    dim = 2
    offset = 0.25 * jnp.ones(dim)
    mcmc_info = {"offset": offset, "bound": 0.25}
    sampler = Sampler(name="paper", mcmc_info=mcmc_info, numChains=1, latent_space_prob=latent_space_dist_paper)
    states = sampler(2000000)
    # states = states[:, 100000:, :]
    print(states.shape)
    print(states)
    # exit()

    import matplotlib.pyplot as plt
    plt.hist2d(states[:, :, 0].reshape(-1), states[:, :, 1].reshape(-1), bins=200, range=[[0, 1], [0, 1]])
    plt.show()
