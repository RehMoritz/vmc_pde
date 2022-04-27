import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

import time
from dataclasses import dataclass
import h5py

import global_defs
import var_state
import net
import grid
import train
import mpi_wrapper as mpi

from functools import partial
import time


def build_cov_matrix(L_para, L_diag, dim):
    L = jnp.zeros((dim, dim), dtype=global_defs.tReal)
    idx = jnp.triu_indices(dim, k=1)
    L = L.at[idx].set(L_para)
    L = L + jnp.diag(jnp.exp(L_diag))
    return L @ L.T


def store_infos(wdir, infos, name="infos.hdf5"):
    with h5py.File(wdir + name, "w") as f:
        for key, value in infos.items():
            f.create_dataset(key, data=value)


class Timings():
    def __init__(self):
        self.timing_dict = {}

    def start_timing(self, key):
        if key not in self.timing_dict.keys():
            self.timing_dict[key] = []
        self.timing_dict[key].append(- time.perf_counter())

    def stop_timing(self, key):
        self.timing_dict[key][-1] += time.perf_counter()

    def print_timings(self):
        total = 0
        for key, value in self.timing_dict.items():
            print(f"\t > {key}: {value[-1]}")
            total += value[-1]
        print(f"\t > TOTAL: {total}")


if __name__ == "__main__":
    dim = 4
    L = 0 * jax.random.normal(jax.random.PRNGKey(1), shape=((dim**2 - dim) // 2,))
    L_diag = jnp.zeros(dim)
    S = build_cov_matrix(L, L_diag, dim)
    eigvals, eigvecs = jnp.linalg.eigh(S)

    print(S)
    print(eigvals)
    print(eigvecs)
