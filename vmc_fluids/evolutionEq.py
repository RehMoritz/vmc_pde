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


@dataclass
class EvolutionEquation:
    name: str = "diffusion"

    def __post_init__(self, eqParams={"D": 1.}):
        self.eqParams = eqParams
        self.function_dict = {"diffusion": self._diffusion_eq}

    def __call__(self, vState, configs):
        return self.function_dict[self.name](vState, configs)

    def _diffusion_eq(self, vState, configs):
        logProbs, coord_grads, param_grads = vState(configs, mode="eval_coordgrads")
        hessian = vState.hessian(configs)
        time_grads = self.eqParams["D"] * (jnp.sum(coord_grads**2, axis=-1) + jnp.einsum('abii->ab', hessian))
        return time_grads, param_grads
