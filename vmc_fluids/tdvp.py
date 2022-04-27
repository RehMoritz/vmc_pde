import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

import time
from dataclasses import dataclass
import scipy.special

import global_defs
import var_state
import net
import grid
import train
import mpi_wrapper as mpi

from functools import partial


@dataclass
class TDVP:
    useSNR: bool = False
    snrTol: float = 2e0
    svdTol: float = 1e-11
    diagonalShift: float = 0
    diagonalizeOnDevice: bool = False

    def __post_init__(self):
        # pmap'd member functions
        self.subtract_helper_Eloc = global_defs.pmap_for_my_devices(lambda x, y: x - y, in_axes=(0, None))
        self.subtract_helper_grad = global_defs.pmap_for_my_devices(lambda x, y: x - y, in_axes=(0, None))
        self.get_EO = global_defs.pmap_for_my_devices(lambda Eloc, grad: Eloc[:, None] * grad, in_axes=(0, 0))
        self.get_expGrad = global_defs.pmap_for_my_devices(lambda logProb, grad: logProb[:, None] * grad, in_axes=(0, 0))
        self.transform_EO = global_defs.pmap_for_my_devices(lambda eo, v: jnp.matmul(eo, jnp.conj(v)), in_axes=(0, None))

    def get_tdvp_equation(self, Eloc, gradients, logProbs):
        self.ElocMean = mpi.global_mean(Eloc)
        self.ElocMeanAbs = mpi.global_mean(jnp.abs(Eloc))
        self.ElocVar = jnp.real(mpi.global_variance(Eloc))
        Eloc = self.subtract_helper_Eloc(Eloc, self.ElocMean)
        gradientsMean = mpi.global_mean(gradients)
        gradients = self.subtract_helper_grad(gradients, gradientsMean)

        EOdata = self.get_EO(Eloc, gradients)
        self.F0 = mpi.global_mean(EOdata)
        self.S0 = mpi.global_covariance(gradients)
        self.SExp = mpi.global_covariance(self.get_expGrad(logProbs, gradients))
        S, F = self.S0, self.F0

        if self.diagonalShift > 1e-10:
            S = S + jnp.diag(self.diagonalShift * jnp.diag(S))
        return S, F, EOdata

    def get_sr_equation(self, Eloc, gradients):
        return get_tdvp_equation(Eloc, gradients, rhsPrefactor=1.)

    def transform_to_eigenbasis(self, S, F, EOdata):
        if self.diagonalizeOnDevice:
            self.ev, self.V = jnp.linalg.eigh(S)
        else:
            tmpS = np.array(S)
            tmpEv, tmpV = np.linalg.eigh(tmpS)
            self.ev = jnp.array(tmpEv)
            self.V = jnp.array(tmpV)

        self.VtF = jnp.dot(jnp.transpose(jnp.conj(self.V)), F)

        EOdata = self.transform_EO(EOdata, self.V)
        EOdata.block_until_ready()
        self.rhoVar = mpi.global_variance(EOdata)
        self.snr = jnp.sqrt(jnp.abs(mpi.globNumSamples * (jnp.conj(self.VtF) * self.VtF) / self.rhoVar))

    def solve(self, Eloc, gradients, logProbs):
        # Get TDVP equation from MC data
        self.S, F, Fdata = self.get_tdvp_equation(Eloc, gradients, logProbs)
        F.block_until_ready()

        # Transform TDVP equation to eigenbasis
        self.transform_to_eigenbasis(self.S, F, Fdata)

        # Discard eigenvalues below numerical precision
        self.invEv = jnp.where(jnp.abs(self.ev / self.ev[-1]) > 1e-14, 1. / self.ev, 0.)

        # Set regularizer for singular value cutoff
        regularizer = 1. / (1. + (self.svdTol / jnp.abs(self.ev / self.ev[-1]))**6)

        if self.useSNR:
            # Construct a soft cutoff based on the SNR
            regularizer *= 1. / (1. + (self.snrTol / self.snr)**6)

        update = jnp.real(jnp.dot(self.V, (self.invEv * regularizer * self.VtF)))

        tdvp_error = 1 + (update @ self.S0 @ update - 2 * self.F0 @ update) / jnp.mean(Eloc**2)
        return update, jnp.linalg.norm(self.S.dot(update) - F) / jnp.linalg.norm(F), tdvp_error

    def __call__(self, netParameters, t, psi, evolutionEq, **rhsArgs):
        nSamplesTDVP = rhsArgs["nSamplesTDVP"]
        nSamplesObs = rhsArgs["nSamplesObs"]
        mpi.globNumSamples = nSamplesTDVP
        tmpParameters = psi.get_parameters()
        psi.set_parameters(netParameters)

        timings = rhsArgs["timings"]

        def start_timing(timings, name):
            if timings is not None:
                timings.start_timing(name)

        def stop_timing(timings, name, waitFor=None):
            if waitFor is not None:
                waitFor.block_until_ready()
            if timings is not None:
                timings.stop_timing(name)

        # Get sample
        start_timing(timings, "sampling")
        sampleConfigs, logProbs = psi.sample(numSamples=nSamplesTDVP)
        stop_timing(timings, "sampling", waitFor=sampleConfigs)

        # Evaluate local energy
        start_timing(timings, "compute Eloc")
        Eloc, sampleGradients, logProbs = evolutionEq(psi, sampleConfigs, t)
        # sampleGradients = jnp.clip(sampleGradients, a_min=-100, a_max=100)
        stop_timing(timings, "compute Eloc", waitFor=Eloc)

        start_timing(timings, "solve TDVP eqn.")
        update, self.solverResidual, self.tdvp_error = self.solve(Eloc, sampleGradients, logProbs)
        stop_timing(timings, "solve TDVP eqn.")

        if nSamplesObs > nSamplesTDVP:
            # Get sample
            start_timing(timings, "sampling observables")
            sampleConfigs, logProbs = psi.sample(numSamples=nSamplesObs)
            stop_timing(timings, "sampling observables", waitFor=sampleConfigs)

        if jnp.any(jnp.isnan(update)):
            print(sampleGradients)
            print(self.S0)
            print(self.F0)
            print("nan encountered. Exitting.")
            exit()

        info = {}
        mean = jnp.mean(sampleConfigs, axis=(0, 1), keepdims=True)
        info["x1"] = mean[0, 0]
        info["covar"] = jnp.cov(sampleConfigs[0, ...].T, ddof=0)
        info["entropy"] = -jnp.mean(logProbs)
        for m in [3, 4, 5, 6]:
            info[f"x{m}"] = jnp.mean((sampleConfigs - mean)**m, axis=(0, 1))
        info["max_grad"] = jnp.max(Eloc)

        # integrate on small cell
        dim = sampleConfigs.shape[-1]
        samples = jax.random.normal(psi.sampler.key, shape=(1, nSamplesObs, sampleConfigs.shape[-1]))
        samples = samples / jnp.linalg.norm(samples, axis=-1, keepdims=True) * jax.random.uniform(psi.sampler.key, shape=(1, nSamplesObs,))[..., None]**(1 / dim)

        for lim in [1, 0.5, 0.1]:  # the sigma here is in reference to a standard normal distribution - we need to scale it accordingly
            lim_normal = lim
            T = 10
            lim = lim * jnp.sqrt(T)  # the variance is T - the stddev is sqrt(T)
            sphere_volume = jnp.pi**(dim / 2) / scipy.special.gamma(dim / 2 + 1) * lim**dim
            info[f"integral_{lim_normal}sigma"] = jnp.mean(jnp.exp(psi(lim * samples))) * sphere_volume

        return update, info
