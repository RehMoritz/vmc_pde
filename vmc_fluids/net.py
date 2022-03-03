import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial

import global_defs


def uniform_init(rng, shape, scale=1.0):
    unif = jax.nn.initializers.uniform()
    return 2 * scale * (unif(rng, shape) / 0.01 - 0.5)


class SingleTrafo(nn.Module):
    intmediate: tuple = (10,)
    width: int = 4
    use_bias: bool = True
    kernel_init: callable = partial(uniform_init, scale=1e-4)
    bias_init: callable = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        for feat in self.intmediate:
            x = nn.Dense(features=feat, use_bias=self.use_bias,
                         kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
            x = 1 + nn.elu(x)
        return nn.tanh(nn.Dense(features=self.width, use_bias=self.use_bias,
                                kernel_init=self.kernel_init, bias_init=self.bias_init)(x))
        # return nn.Dense(features=self.width, use_bias=self.use_bias)(x)


class SingleBlock(nn.Module):
    ind_up: list
    ind_down: list
    intmediate: tuple = (10,)
    width: int = 4
    pt_sym: bool = True

    def setup(self):
        self.s1 = SingleTrafo(self.intmediate, len(self.ind_down))
        self.t1 = SingleTrafo(self.intmediate, len(self.ind_down))
        self.s2 = SingleTrafo(self.intmediate, len(self.ind_up))
        self.t2 = SingleTrafo(self.intmediate, len(self.ind_up))

    def __call__(self, x, inv=False):
        if not inv:
            u1 = x[self.ind_up]
            u2 = x[self.ind_down]

            if self.pt_sym:
                v1 = u1 * jnp.exp(self.s2(u2) + self.s2(-u2)) + (self.t2(u2) - self.t2(-u2))
                v2 = u2 * jnp.exp(self.s1(v1) + self.s1(-v1)) + (self.t1(v1) - self.t1(-v1))
            else:
                v1 = u1 * jnp.exp(self.s2(u2)) + self.t2(u2)
                v2 = u2 * jnp.exp(self.s1(v1)) + self.t1(v1)

            result = jnp.zeros_like(x)
            result = result.at[self.ind_up].set(v1)
            result = result.at[self.ind_down].set(v2)

            return result

        else:
            v1 = x[self.ind_up]
            v2 = x[self.ind_down]

            if self.pt_sym:
                u2 = (v2 - (self.t1(v1) - self.t1(-v1))) * jnp.exp(-(self.s1(v1) + self.s1(-v1)))
                u1 = (v1 - (self.t2(u2) - self.t2(-u2))) * jnp.exp(-(self.s2(u2) + self.s2(-u2)))
            else:
                u2 = (v2 - self.t1(v1)) * jnp.exp(-self.s1(v1))
                u1 = (v1 - self.t2(u2)) * jnp.exp(-self.s2(u2))

            result = jnp.zeros_like(x)
            result = result.at[self.ind_up].set(u1)
            result = result.at[self.ind_down].set(u2)

            return result


class INN(nn.Module):
    inds_up: list
    inds_down: list
    widths: tuple = (4,)
    pt_sym: bool = False

    def setup(self):
        self.blocks = [SingleBlock(ind_up, ind_down, intmediate=(2,), width=width, pt_sym=self.pt_sym) for width, ind_up, ind_down in zip(self.widths, self.inds_up, self.inds_down)]

    def __call__(self, x, inv=False):
        if not inv:
            for block in self.blocks:
                x = block(x, inv=inv)
        else:
            for block in self.blocks[::-1]:
                x = block(x, inv=inv)
        return x


class SanityINN(nn.Module):
    """
    Same interface, but rescaling with a single variational parameter of the input.
    """
    inds_up: list
    inds_down: list
    widths: tuple = (4,)
    pt_sym: bool = False

    @nn.compact
    def __call__(self, x, inv=False):
        if not inv:
            return self.param("scale", jax.nn.initializers.ones, 1) * x
        else:
            return 1 / self.param("scale", jax.nn.initializers.ones, 1) * x


if __name__ == "__main__":
    s = jnp.array([1, 0])
    sINN = SanityINN(inds_up=[], inds_down=[])
    param = sINN.init(jax.random.PRNGKey(1), s)
    print(param)
    param = param.unfreeze()
    param["params"]["scale"] = jnp.array([2.])
    print(param)

    z = sINN.apply(param, s, inv=False)
    x = sINN.apply(param, z, inv=True)
    print(z)
    print(x)
