import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

import global_defs
import var_state
import net
import grid
import train
import mpi_wrapper

from functools import partial
from dataclasses import dataclass
import time


def _getRandomD_matrix(dim):
    A = jax.random.normal(jax.random.PRNGKey(0), shape=(dim, dim))
    return A.T @ A


def _velocity_field_MLPaper(evolParams, coord, t):
    x, y = coord[0], coord[1]
    print(evolParams)
    return jnp.array([-jnp.sin(jnp.pi * x)**2 * jnp.sin(2 * jnp.pi * y) * jnp.cos(jnp.pi * t / evolParams["T"]),
                      jnp.sin(jnp.pi * y)**2 * jnp.sin(2 * jnp.pi * x) * jnp.cos(jnp.pi * t / evolParams["T"])], dtype=global_defs.tReal)


def _velocity_field_hamiltonian(evolParams, coord, t):
    """returns dx/dt = p, dp/dt = -x"""
    def H(x, coupled=True):
        if coupled:
            xs = x[0::2]
            ps = x[1::2]
            return jnp.pi * (evolParams["m"] * evolParams["omega"]**2 / 2 * jnp.sum((xs - jnp.roll(xs, 1))**2)
                             + jnp.sum(ps**2) / (2. * evolParams["m"])
                             + evolParams["lam"] * jnp.sum(xs**4))
        else:
            return jnp.pi * (evolParams["m"] * evolParams["omega"]**2 / 2 * jnp.sum(x[0::2]**2)
                             + jnp.sum(x[1::2]**2) / (2. * evolParams["m"])
                             + evolParams["lam"] * jnp.sum(x[0::2]**4))
    grads = jax.grad(H)(coord)
    mat = jnp.kron(jnp.eye(coord.shape[0] // 2), jnp.array([[0, 1], [-1, 0]]))
    return mat @ grads


@dataclass
class EvolutionEquation:
    dim: int = 2
    name: str = "diffusion"

    def __post_init__(self, eqParams={"D": 1.}):
        self.function_dict = {"diffusion": self._diffusion_eq,
                              "diffusion_drift": self._diffusion_eq_wDrift,
                              "diffusion_anisotropic": self._diffusion_eq_anisotropic,
                              "advection_paper": self._advection,
                              "advection_hamiltonian": self._advection,
                              "advection_hamiltonian_wDiss": self._advection_wDiss,
                              }
        self.eqParams = {"diffusion":
                         {"D": 1},
                         "diffusion_anisotropic":
                         {"D": _getRandomD_matrix(self.dim)},
                         "diffusion_drift":
                         {"D": 1, "mu": 4},
                         "advection_paper":
                         {"params": {"T": 5},
                          "vel_field": _velocity_field_MLPaper
                          },
                         "advection_hamiltonian":
                         {"params": {"m": 1.0, "omega": 1.0, "lam": 0.0},
                          "vel_field": _velocity_field_hamiltonian},
                         "advection_hamiltonian_wDiss":
                         {"params": {"m": 1.0, "omega": 1.0, "T": 1.0, "gamma": 1.0, "lam": 0.2},
                          "vel_field": _velocity_field_hamiltonian}
                         }

        self._get_advection_velfield_jitd = global_defs.pmap_for_my_devices(jax.vmap(lambda coord_grad, coord, t, vel_field: - coord_grad @ vel_field(coord, t), in_axes=(0, 0, None, None)), in_axes=(0, 0, None, None), static_broadcasted_argnums=(3,))

    def __call__(self, *args):
        return self.function_dict[self.name](*args)

    def _diffusion_eq(self, vState, configs, t):
        logProbs, coord_grads, param_grads = vState(configs, mode="eval_coordgrads")
        # time_grads = self.eqParams[self.name]["D"] * (jnp.sum(coord_grads**2, axis=-1) + jnp.sum(vState.hessian_diag(configs), axis=(-1,)))
        time_grads = self.eqParams[self.name]["D"] * (jnp.sum(coord_grads**2, axis=-1) + jnp.einsum('abii -> ab', vState.hessian(configs)))
        return time_grads, param_grads, logProbs

    def _diffusion_eq_wDrift(self, vState, configs, t):
        logProbs, coord_grads, param_grads = vState(configs, mode="eval_coordgrads")
        time_grads = (self.eqParams[self.name]["D"] * (jnp.sum(coord_grads**2, axis=-1) + jnp.einsum('abii -> ab', vState.hessian(configs)))
                      + self.eqParams[self.name]["mu"] * jnp.sum(coord_grads, axis=-1))

        return time_grads, param_grads, logProbs

    def _diffusion_eq_anisotropic(self, vState, configs, t):
        logProbs, coord_grads, param_grads = vState(configs, mode="eval_coordgrads")
        # time_grads = self.eqParams[self.name]["D"] * (jnp.sum(coord_grads**2, axis=-1) + jnp.einsum('abii -> ab', vState.hessian(configs)))
        time_grads = (jnp.einsum('abi, ij, abj -> ab', coord_grads, self.eqParams[self.name]["D"], coord_grads) +
                      jnp.einsum('abij, ji -> ab', vState.hessian(configs), self.eqParams[self.name]["D"]))
        return time_grads, param_grads, logProbs

    def _advection(self, vState, configs, t):
        logProbs, coord_grads, param_grads = vState(configs, mode="eval_coordgrads")
        time_grads = self._get_advection_velfield_jitd(coord_grads, configs, t, partial(self.eqParams[self.name]["vel_field"], self.eqParams[self.name]["params"]))

        # arg = jnp.argmax(time_grads)
        # print("New Configuration")
        # print(arg)
        # print(configs[0, arg])
        # print(time_grads[0, arg])
        # print(jnp.exp(logProbs[0, arg]) * time_grads[0, arg])
        # print(jnp.exp(logProbs[0, arg]) * coord_grads[0, arg])
        # print(jnp.exp(logProbs[0, arg]))
        # exit()
        return time_grads, param_grads, logProbs

    def _advection_wDiss(self, vState, configs, t):
        """implements Eq. 2.14 from https://arxiv.org/pdf/quant-ph/9709002.pdf"""

        logProbs, coord_grads, param_grads = vState(configs, mode="eval_coordgrads")
        time_grads_adv = self._get_advection_velfield_jitd(coord_grads, configs, t, partial(self.eqParams[self.name]["vel_field"], self.eqParams[self.name]["params"]))
        time_grads_diff = (self.eqParams[self.name]["params"]["m"] * self.eqParams[self.name]["params"]["gamma"] * self.eqParams[self.name]["params"]["T"] *
                           (jnp.sum(coord_grads[:, :, 1::2]**2, axis=-1) + jnp.einsum('abii -> ab', vState.hessian(configs)[:, :, 1::2, 1::2])))
        time_grads_damping = self.eqParams[self.name]["params"]["gamma"] * (logProbs + jnp.sum(configs[:, :, 1::2] * coord_grads[:, :, 1::2], axis=-1))
        time_grads = time_grads_adv + time_grads_diff + time_grads_damping

        return time_grads, param_grads, logProbs


if __name__ == "__main__":
    # evolEq = EvolutionEquation(name="advection_divFree")
    v = _velocity_field_hamiltonian(None, jnp.array([3., 3.]), 0)
    print(v)
