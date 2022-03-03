import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

import time
from dataclasses import dataclass

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
    snrTol: float = 2.
    svdTol: float = 1e-7
    diagonalShift: float = 1e-9
    diagonalizeOnDevice: bool = True

    def __post_init__(self):
        # pmap'd member functions
        self.subtract_helper_Eloc = global_defs.pmap_for_my_devices(lambda x, y: x - y, in_axes=(0, None))
        self.subtract_helper_grad = global_defs.pmap_for_my_devices(lambda x, y: x - y, in_axes=(0, None))
        self.get_EO = global_defs.pmap_for_my_devices(lambda Eloc, grad: jnp.multiply(Eloc[:, None], jnp.conj(grad)),
                                                      in_axes=(0, 0), static_broadcasted_argnums=())
        self.transform_EO = global_defs.pmap_for_my_devices(lambda eo, v: jnp.matmul(eo, jnp.conj(v)), in_axes=(0, None))
        self.compute_variance = global_defs.pmap_for_my_devices(lambda samples: jnp.mean(jax.vmap(lambda sample: sample**2)(samples)))

    def get_tdvp_equation(self, Eloc, gradients):
        self.ElocMean = mpi.global_mean(Eloc)
        self.ElocMeanAbs = mpi.global_mean(jnp.abs(Eloc))
        self.ElocVar = jnp.real(mpi.global_variance(Eloc))
        Eloc = self.subtract_helper_Eloc(Eloc, self.ElocMean)
        gradientsMean = mpi.global_mean(gradients)
        gradients = self.subtract_helper_grad(gradients, gradientsMean)

        EOdata = self.get_EO(Eloc, gradients)
        self.F0 = mpi.global_mean(EOdata)
        self.S0 = mpi.global_covariance(gradients)
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
        self.snr = jnp.sqrt(jnp.abs(mpi.globNumSamples / (self.rhoVar / (jnp.conj(self.VtF) * self.VtF) - 1.)))

    def solve(self, Eloc, gradients):
        # Get TDVP equation from MC data
        self.S, F, Fdata = self.get_tdvp_equation(Eloc, gradients)
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
        return update, jnp.linalg.norm(self.S.dot(update) - F) / jnp.linalg.norm(F)

    def __call__(self, netParameters, t, psi, evolutionEq, **rhsArgs):
        numSamples = rhsArgs["numSamples"]
        mpi.globNumSamples = numSamples
        tmpParameters = psi.get_parameters()
        psi.set_parameters(netParameters)

        outp = None
        if "outp" in rhsArgs:
            outp = rhsArgs["outp"]
        self.outp = outp

        def start_timing(outp, name):
            if outp is not None:
                outp.start_timing(name)

        def stop_timing(outp, name, waitFor=None):
            if waitFor is not None:
                waitFor.block_until_ready()
            if outp is not None:
                outp.stop_timing(name)

        # Get sample
        start_timing(outp, "sampling")
        sampleConfigs = psi.sample(numSamples=numSamples)
        stop_timing(outp, "sampling", waitFor=sampleConfigs)

        # Evaluate local energy
        start_timing(outp, "compute Eloc")
        Eloc, sampleGradients = evolutionEq(psi, sampleConfigs)
        stop_timing(outp, "compute Eloc", waitFor=Eloc)

        start_timing(outp, "solve TDVP eqn.")
        update, solverResidual = self.solve(Eloc, sampleGradients)
        stop_timing(outp, "solve TDVP eqn.")

        if outp is not None:
            outp.add_timing("MPI communication", mpi.get_communication_time())

        info = {}
        info["variance"] = self.compute_variance(sampleConfigs - jnp.mean(sampleConfigs, axis=(0, 1), keepdims=True))

        return update, info
