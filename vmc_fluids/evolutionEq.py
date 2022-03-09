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


@dataclass
class EvolutionEquation:
    name: str = "diffusion"

    def __post_init__(self, eqParams={"D": 1.}):
        self.function_dict = {"diffusion": self._diffusion_eq,
                              "diffusion_drift": self._diffusion_eq_wDrift,
                              "advection_divFree": self._advection_divFreeVelocity,
                              }
        self.eqParams = {"diffusion":
                         {"D": 1},
                         "diffusion_drift":
                         {"D": 1, "mu": 4},
                         "advection_divFree":
                         {"T": 5},
                         }
        self._get_advection_difFreeVelocity_jitd = global_defs.pmap_for_my_devices(jax.vmap(lambda coord_grad, coord, t: - coord_grad @ self._velocity_field(coord, t), in_axes=(0, 0, None)), in_axes=(0, 0, None))

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

    def _advection_divFreeVelocity(self, vState, configs, t):
        logProbs, coord_grads, param_grads = vState(configs, mode="eval_coordgrads")
        time_grads = self._get_advection_difFreeVelocity_jitd(coord_grads, configs, t)
        return time_grads, param_grads

    def _velocity_field(self, coord, t):
        x, y = coord[0], coord[1]
        return jnp.array([-jnp.sin(jnp.pi * x)**2 * jnp.sin(2 * jnp.pi * y) * jnp.cos(jnp.pi * t / self.eqParams[self.name]["T"]),
                          jnp.sin(jnp.pi * y)**2 * jnp.sin(2 * jnp.pi * x) * jnp.cos(jnp.pi * t / self.eqParams[self.name]["T"])])


if __name__ == "__main__":
    evolEq = EvolutionEquation(name="advection_divFree")
    v = evolEq._velocity_field(jnp.array([0.1, 0.5]), 0)
    print(v)
