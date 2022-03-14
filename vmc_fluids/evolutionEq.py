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


def _velocity_field_MLPaper(evolParams, coord, t):
    x, y = coord[0], coord[1]
    return jnp.array([-jnp.sin(jnp.pi * x)**2 * jnp.sin(2 * jnp.pi * y) * jnp.cos(jnp.pi * t / evolParams["T"]),
                      jnp.sin(jnp.pi * y)**2 * jnp.sin(2 * jnp.pi * x) * jnp.cos(jnp.pi * t / evolParams["T"])], dtype=global_defs.tReal)


def _velocity_field_hamiltonian(evolParams, coord, t):
    """returns dx/dt = p, dp/dt = -x"""
    def H(x):
        lam = 1e-1 * 1
        return jnp.pi * (jnp.sum(x**2) + lam * x[0]**4)
    grads = jax.grad(H)(coord)
    return jnp.array([grads[1], -grads[0]])


@dataclass
class EvolutionEquation:
    name: str = "diffusion"

    def __post_init__(self, eqParams={"D": 1.}):
        self.function_dict = {"diffusion": self._diffusion_eq,
                              "diffusion_drift": self._diffusion_eq_wDrift,
                              "advection_paper": self._advection,
                              "advection_hamiltonian": self._advection,
                              }
        self.eqParams = {"diffusion":
                         {"D": 1},
                         "diffusion_drift":
                         {"D": 1, "mu": 4},
                         "advection_paper":
                         {"params": {"T": 5},
                          "vel_field": _velocity_field_MLPaper
                          },
                         "advection_hamiltonian":
                         {"params": {},
                          "vel_field": _velocity_field_hamiltonian}
                         }

        self._get_advection_difFreeVelocity_jitd = global_defs.pmap_for_my_devices(jax.vmap(lambda coord_grad, coord, t, vel_field: - coord_grad @ vel_field(coord, t), in_axes=(0, 0, None, None)), in_axes=(0, 0, None, None), static_broadcasted_argnums=(3,))

    def __call__(self, *args):
        return self.function_dict[self.name](*args)

    def _diffusion_eq(self, vState, configs, t):
        logProbs, coord_grads, param_grads = vState(configs, mode="eval_coordgrads")
        # time_grads = self.eqParams[self.name]["D"] * (jnp.sum(coord_grads**2, axis=-1) + jnp.sum(vState.hessian_diag(configs), axis=(-1,)))
        time_grads = self.eqParams[self.name]["D"] * (jnp.sum(coord_grads**2, axis=-1) + jnp.einsum('abii -> ab', vState.hessian(configs)))
        return time_grads, param_grads

    def _diffusion_eq_wDrift(self, vState, configs, t):
        logProbs, coord_grads, param_grads = vState(configs, mode="eval_coordgrads")
        # time_grads = (self.eqParams[self.name]["D"] * (jnp.sum(coord_grads**2, axis=-1) + jnp.sum(vState.hessian_diag(configs), axis=(-1,)))
        #               + self.eqParams[self.name]["mu"] * jnp.sum(coord_grads, axis=-1))
        time_grads = (self.eqParams[self.name]["D"] * (jnp.sum(coord_grads**2, axis=-1) + jnp.einsum('abii -> ab', vState.hessian(configs)))
                      + self.eqParams[self.name]["mu"] * jnp.sum(coord_grads, axis=-1))

        return time_grads, param_grads

    def _advection(self, vState, configs, t):
        logProbs, coord_grads, param_grads = vState(configs, mode="eval_coordgrads")
        time_grads = self._get_advection_difFreeVelocity_jitd(coord_grads, configs, t, partial(self.eqParams[self.name]["vel_field"], self.eqParams[self.name]["params"]))

        arg = jnp.argmax(time_grads)
        print("New Configuration")
        # print(arg)
        print(configs[0, arg])
        # print(time_grads[0, arg])
        # print(jnp.exp(logProbs[0, arg]) * time_grads[0, arg])
        # print(jnp.exp(logProbs[0, arg]) * coord_grads[0, arg])
        print(jnp.exp(logProbs[0, arg]))
        # exit()
        return time_grads, param_grads


if __name__ == "__main__":
    # evolEq = EvolutionEquation(name="advection_divFree")
    v = _velocity_field_hamiltonian(None, jnp.array([3., 3.]), 0)
    print(v)
