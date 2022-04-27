import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial

import global_defs
import util


class Gauss(nn.Module):
    dim: int = 2

    @nn.compact
    def __call__(self, x, dist_params):
        S = dist_params["S"]
        mu = dist_params["mu"]

        x = x - mu
        arg = x @ jnp.linalg.inv(S) @ x
        return - 0.5 * (self.dim * jnp.log(2 * jnp.pi) + jnp.log(jnp.linalg.det(S)) + arg)


class Student_t(nn.Module):
    dim: int = 2

    @nn.compact
    def __call__(self, x, dist_params):
        p = x.shape[0]

        S = dist_params["S"]
        mu = dist_params["mu"]
        nu = jnp.exp(dist_params["dist_params"][0]) + 1e0

        x = x - mu
        return (jax.scipy.special.gammaln((nu + p) / 2) - jax.scipy.special.gammaln(nu / 2) - p / 2 * jnp.log(nu * jnp.pi) -
                (nu + p) / 2 * jnp.log(1 + x @ jnp.linalg.inv(S) @ x / nu))


def uniform_init(rng, shape, scale=1.0):
    unif = jax.nn.initializers.uniform()
    return 2 * scale * (unif(rng, shape) / 0.01 - 0.5)


class SingleTrafo(nn.Module):
    intmediate: tuple = (10,)
    width: int = 4
    use_bias: bool = True
    kernel_init: callable = partial(uniform_init, scale=1e-5)
    bias_init: callable = jax.nn.initializers.zeros
    alpha: float = 1e1

    @nn.compact
    def __call__(self, x):
        for feat in self.intmediate:
            x = nn.Dense(features=feat, use_bias=self.use_bias,
                         kernel_init=partial(uniform_init, scale=1e-0), bias_init=self.bias_init)(x)
            # x = 1 + nn.elu(x)
            # x = nn.relu(x)
            x = nn.tanh(x)
        return self.alpha * nn.tanh(nn.Dense(features=self.width, use_bias=self.use_bias,
                                             kernel_init=self.kernel_init, bias_init=self.bias_init)(x))
        # return nn.Dense(features=self.width, use_bias=self.use_bias)(x)


class SingleBlock(nn.Module):
    ind_up: list
    ind_down: list
    intmediate: tuple = (3,)
    jac_eq_1: bool = False
    different_add: bool = False
    no_add: bool = True
    global_change: bool = False

    def setup(self):
        self.s1 = SingleTrafo(self.intmediate, len(self.ind_down))
        self.s2 = SingleTrafo(self.intmediate, len(self.ind_up))
        if self.different_add:
            self.t1 = SingleTrafo(self.intmediate, len(self.ind_down))
            self.t2 = SingleTrafo(self.intmediate, len(self.ind_up))
        if self.global_change:
            self.global_scale = self.param("global_scale", jax.nn.initializers.ones, 1)
            self.global_offset = self.param("global_offset", jax.nn.initializers.zeros, len(self.ind_up) + len(self.ind_down))

    def __call__(self, x, inv=False):
        if not inv:
            u1 = x[self.ind_up]
            u2 = x[self.ind_down]

            s2_u2 = self.s2(u2)
            if self.jac_eq_1:
                v1 = u1 + s2_u2
                s2_u2 = jnp.zeros_like(s2_u2)
            elif self.different_add:
                v1 = u1 * jnp.exp(s2_u2) + self.t2(u2)
            elif self.no_add:
                v1 = u1 * jnp.exp(s2_u2)
            else:
                v1 = u1 * jnp.exp(s2_u2) + s2_u2

            s1_v1 = self.s1(v1)
            if self.jac_eq_1:
                v2 = u2 + s1_v1
                s1_v1 = jnp.zeros_like(s1_v1)
            elif self.different_add:
                v2 = u2 * jnp.exp(s1_v1) + self.t1(v1)
            elif self.no_add:
                v2 = u2 * jnp.exp(s1_v1)
            else:
                v2 = u2 * jnp.exp(s1_v1) + s1_v1

            result = jnp.zeros_like(x)
            result = result.at[self.ind_up].set(v1)
            result = result.at[self.ind_down].set(v2)

            if self.global_change:
                return self.global_scale * result + self.global_offset, (jnp.sum(s2_u2) + jnp.sum(s1_v1)) + jnp.log(self.global_scale[0]) * (len(self.ind_up) + len(self.ind_down))
            else:
                return result, (jnp.sum(s2_u2) + jnp.sum(s1_v1))

        else:
            v1 = x[self.ind_up]
            v2 = x[self.ind_down]

            s1_v1 = self.s1(v1)
            if self.jac_eq_1:
                u2 = v2 - s1_v1
                s1_v1 = jnp.zeros_like(s1_v1)
            elif self.different_add:
                u2 = (v2 - self.t1(v1)) * jnp.exp(-s1_v1)
            elif self.no_add:
                u2 = v2 * jnp.exp(-s1_v1)
            else:
                u2 = (v2 - s1_v1) * jnp.exp(-s1_v1)

            s2_u2 = self.s2(u2)
            if self.jac_eq_1:
                u1 = v1 - s2_u2
                s2_u2 = jnp.zeros_like(s2_u2)
            elif self.different_add:
                u1 = (v1 - self.t2(u2)) * jnp.exp(-s2_u2)
            elif self.no_add:
                u1 = v1 * jnp.exp(-s2_u2)
            else:
                u1 = (v1 - s2_u2) * jnp.exp(-s2_u2)

            result = jnp.zeros_like(x)
            result = result.at[self.ind_up].set(u1)
            result = result.at[self.ind_down].set(u2)

            if self.global_change:
                return (result - self.global_offset) / self.global_scale, - (jnp.sum(s1_v1) + jnp.sum(s2_u2)) - jnp.log(self.global_scale[0]) * (len(self.ind_up) + len(self.ind_down))
            else:
                return result, - (jnp.sum(s1_v1) + jnp.sum(s2_u2))


class INN(nn.Module):
    """
    An INN that solely does an invertible coordinate transform.
    It returns the new coordinates and the jacobian determinant associated with the transform.
    """
    inds_up: list
    inds_down: list
    intmediate: tuple = (3,)

    def setup(self):
        self.blocks = [SingleBlock(ind_up, ind_down, intmediate=self.intmediate) for ind_up, ind_down in zip(self.inds_up, self.inds_down)]

    def __call__(self, x, inv=False):
        log_jac = 0

        if not inv:
            # from Real Space -> Latent Space
            for block in self.blocks:
                x, log_jac_block = block(x, inv=inv)
                log_jac += log_jac_block
        else:
            # from Latent Space -> Real Space
            for block in self.blocks[::-1]:
                x, log_jac_block = block(x, inv=inv)
                log_jac += log_jac_block

        return x, log_jac


class INNwProb(nn.Module):
    """
    Class that evaluates the probability at a certain point in real space or generates samples in latent space.
    """
    inds_up: list
    inds_down: list
    intmediate: tuple = (3,)
    offset: any = jnp.zeros(2)
    latentSpaceName: str = "Gauss"
    dim: int = 2

    def setup(self):
        self.latent_space_dist_dict = {"Gauss": {"pdf": Gauss, "N_params": 0},
                                       "Student_t": {"pdf": Student_t, "N_params": 1}}
        self.latent_space_dist = self.latent_space_dist_dict[self.latentSpaceName]["pdf"](dim=self.dim)

        self.A = self.param("A", jax.nn.initializers.zeros, (self.dim, self.dim))
        self.mu = self.param("mu", jax.nn.initializers.zeros, self.dim)
        self.dist_params = self.param("dist_params", jax.nn.initializers.zeros, (self.latent_space_dist_dict[self.latentSpaceName]["N_params"]))
        # self.dist_params = None

        self.myINN = INN(inds_up=self.inds_up, inds_down=self.inds_down, intmediate=self.intmediate)

    def __call__(self, x, evaluate=True, inv=False):
        if evaluate:  # used for evaluation
            z, log_jac = self.myINN(x, inv=inv)
            p_latent_log = self.latent_space_dist(z - self.offset, {"S": util.build_cov_matrix(self.A), "mu": self.mu, "dist_params": self.dist_params})
            return p_latent_log + log_jac
        else:  # used for sampling
            p_latent_log = self.latent_space_dist(x - self.offset, {"S": util.build_cov_matrix(self.A), "mu": self.mu, "dist_params": self.dist_params})
            x, log_jac = self.myINN(x, inv=inv)
            return x, p_latent_log - log_jac


class SanityINN(nn.Module):
    """
    Same interface, but rescaling with a single variational parameter of the input.
    """
    inds_up: list
    inds_down: list
    widths: tuple = (4,)
    intmediate: bool = False

    @nn.compact
    def __call__(self, x, inv=False):
        scale = self.param("scale", jax.nn.initializers.ones, 1)[0]
        if not inv:
            return scale * x, x.shape[0] * jnp.log(scale)
        else:
            return 1 / scale * x, -x.shape[0] * jnp.log(scale)


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
