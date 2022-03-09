import jax
import jax.numpy as jnp
from jax.experimental import optimizers
import flax.linen as nn
import matplotlib.pyplot as plt

import numpy as np
import global_defs


import net
import grid
import visualization


def gen_data(size, mode="standard_normal", key=0, std=1):
    if mode == "standard_normal":
        data = std * jax.random.normal(jax.random.PRNGKey(key), shape=size)

        def target_fun(x):
            return 1 / jnp.sqrt(2 * jnp.pi * std**2)**size[-1] * jnp.exp(-0.5 * jnp.sum(x**2, axis=-1) / std**2)

        return data, target_fun

    elif mode == "normal_superpos":
        shift = 4
        data = std * jax.random.normal(jax.random.PRNGKey(key), shape=size)
        data = data.at[0, ::2].add(shift)
        data = data.at[0, 1::2].add(-shift)

        def target_fun(x):
            return 0.5 * 1 / jnp.sqrt(2 * jnp.pi * std**2)**size[-1] * (jnp.exp(-0.5 * jnp.sum((x - shift)**2, axis=-1) / std**2) +
                                                                        jnp.exp(-0.5 * jnp.sum((x + shift)**2, axis=-1) / std**2))
        return data, target_fun


def train(vState, data, grid, batchsize=100, epoches=100, key=0, lr=1e-3, target_fun=None):
    key = jax.random.PRNGKey(key)
    opt_init, opt_update, get_params = optimizers.adam(lr)
    opt_state = opt_init(vState.params)

    losses = []
    for epoch, key in enumerate(jax.random.split(key, num=epoches)):
        data = jax.random.permutation(key, data)
        batches = data.reshape((-1, data.shape[0], batchsize, grid.dim))
        for batch_id, batch in enumerate(batches):
            loss, grad = vState(batch, mode="costfun", avg=True)
            opt_state = opt_update(epoch * batches.shape[0] + batch_id, grad, opt_state)
            vState.params = get_params(opt_state)

        print(f"Epoch {epoch}")
        if epoch % 50 == 0:
            print(vState.integrate(grid))
            visualization.plot(vState, grid)
            if target_fun != None:
                visualization.plot_diff(vState, grid, target_fun)
            visualization.plot_data(data, grid)
            plt.show()
        print(loss)
        losses.append(loss)

    plt.figure()
    plt.plot(losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    data = gen_data((100,))
    print(data)
