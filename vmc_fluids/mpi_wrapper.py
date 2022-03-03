from mpi4py import MPI
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

import global_defs

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
commSize = comm.Get_size()

globNumSamples = 0
myNumSamples = 0

from functools import partial
import time
communicationTime = 0.


def _cov_helper_without_p(data):
    return jnp.expand_dims(
        jnp.matmul(jnp.conj(jnp.transpose(data)), data),
        axis=0
    )


_sum_up_pmapd = None
_sum_sq_pmapd = None
_sum_sq_withp_pmapd = None
mean_helper = None
cov_helper_with_p = None
cov_helper_without_p = None

pmapDevices = None

import collections


def pmap_devices_updated():

    if collections.Counter(pmapDevices) == collections.Counter(global_defs.myPmapDevices):
        return False

    return True


def jit_my_stuff():
    # This is a helper function to make sure that pmap'd functions work with the actual choice of devices
    # at all times.

    global _sum_up_pmapd
    global _sum_sq_pmapd
    global _sum_sq_withp_pmapd
    global mean_helper
    global cov_helper_with_p
    global cov_helper_without_p
    global pmapDevices

    if pmap_devices_updated():
        _sum_up_pmapd = global_defs.pmap_for_my_devices(lambda x: jax.lax.psum(jnp.sum(x, axis=0), 'i'), axis_name='i')
        _sum_sq_pmapd = global_defs.pmap_for_my_devices(lambda data, mean: jax.lax.psum(jnp.sum(jnp.conj(data - mean) * (data - mean), axis=0), 'i'), axis_name='i', in_axes=(0, None))
        cov_helper_without_p = global_defs.pmap_for_my_devices(_cov_helper_without_p)

        pmapDevices = global_defs.myPmapDevices


def distribute_sampling(numSamples, localDevices=None, numChainsPerDevice=1):
    """Distribute sampling tasks across processes and devices.

    For a desired total number of samples this function determines how many samples
    should be generated by each Monte Carlo chain.

    It is assumed that a given number of MC chains is running in parallel on each device,
    and that each MPI process can potentially utilize multiple devices. Since the numbers
    of samples per chain have to be identical accross the devices of one MPI process, the
    resulting total number of samples can slightly exceed the requested number of samples.

    Arguments:
        * ``numSamples``: Total number of samples.
        * ``localDevices``: Number of devices per MPI process.
        * ``numChainsPerDevice``: Number of chains run in parallel on each device.

    Returns:
        Number of samples to be generated per device to reach the desired total number of samples.
    """

    global globNumSamples

    # Determine number of samples per process
    samplesPerProcess = numSamples // commSize

    if rank < numSamples % commSize:
        samplesPerProcess += 1

    if localDevices is None:

        globNumSamples = numSamples

        return samplesPerProcess

    numChainsPerProcess = localDevices * numChainsPerDevice

    def spc(spp):
        return (spp + numChainsPerProcess - 1) // numChainsPerProcess

    a = numSamples % commSize
    globNumSamples = (a * spc(1 + numSamples // commSize) + (commSize - a) * spc(numSamples // commSize)) * numChainsPerProcess

    return spc(samplesPerProcess)


def first_sample_id():

    global globNumSamples

    mySamples = globNumSamples // commSize

    firstSampleId = rank * mySamples

    if rank < globNumSamples % commSize:
        firstSampleId += rank
    else:
        firstSampleId += globNumSamples % commSize

    return firstSampleId


def global_sum(data):
    """ Computes the sum of input data across MPI processes and device/batch dimensions.

    On each MPI process the input data is assumed to be a ``jax.numpy.array`` with a leading
    device dimension followed by a batch dimension. The data is reduced by summing up along
    device and batch dimensions as well as accross MPI processes. Hence, the result is an
    array of shape ``data.shape[2:]``.

    Arguments:
        * ``data``: Array of input data.

    Returns:
        Sum of data across MPI processes and device/batch dimensions.
    """

    jit_my_stuff()

    data.block_until_ready()
    t0 = time.perf_counter()

    # Compute sum locally
    localSum = np.array(_sum_up_pmapd(data)[0])

    # Allocate memory for result
    res = np.empty_like(localSum, dtype=localSum.dtype)

    # Global sum
    comm.Allreduce(localSum, res, op=MPI.SUM)

    global communicationTime
    communicationTime += time.perf_counter() - t0

    # return jnp.array(res)
    result = jax.device_put(res, global_defs.myDevice)
    return result


def global_mean(data):
    """ Computes the mean of input data across MPI processes and device/batch dimensions.

    On each MPI process the input data is assumed to be a ``jax.numpy.array`` with a leading
    device dimension followed by a batch dimension. The data is reduced by computing the mean
    along device and batch dimensions as well as accross MPI processes. Hence, the result is
    an array of shape ``data.shape[2:]``.

    If no probabilities ``p`` are given, the empirical mean is computed, i.e.,

        :math:`\\langle X\\rangle=\\frac{1}{N_S}\sum_{j=1}^{N_S} X_j`

    Otherwise, the mean is computed using the given probabilities, i.e.,

        :math:`\\langle X\\rangle=\sum_{j=1}^{N_S} p_jX_j`

    Arguments:
        * ``data``: Array of input data.
        * ``p``: Probabilities associated with the given data.

    Returns:
        Mean of data across MPI processes and device/batch dimensions.
    """

    jit_my_stuff()
    global globNumSamples

    return global_sum(data) / globNumSamples


def global_variance(data):
    """ Computes the variance of input data across MPI processes and device/batch dimensions.

    On each MPI process the input data is assumed to be a ``jax.numpy.array`` with a leading
    device dimension followed by a batch dimension. The data is reduced by computing the variance
    along device and batch dimensions as well as accross MPI processes. Hence, the result is
    an array of shape ``data.shape[2:]``.

    If no probabilities ``p`` are given, the empirical element-wise variance is computed, i.e.,

        :math:`\\text{Var}(X)=\\frac{1}{N_S}\sum_{j=1}^{N_S} |X_j-\\langle X\\rangle|^2`

    Otherwise, the mean is computed using the given probabilities, i.e.,

        :math:`\\text{Var}(X)=\sum_{j=1}^{N_S} p_j |X_j-\\langle X\\rangle|^2`

    Arguments:
        * ``data``: Array of input data.
        * ``p``: Probabilities associated with the given data.

    Returns:
        Variance of data across MPI processes and device/batch dimensions.
    """

    jit_my_stuff()

    data.block_until_ready()

    mean = global_mean(data)

    # Compute sum locally
    localSum = None

    res = _sum_sq_pmapd(data, mean)[0]
    res.block_until_ready()
    localSum = np.array(res)

    # Allocate memory for result
    res = np.empty_like(localSum, dtype=localSum.dtype)

    t0 = time.perf_counter()

    # Global sum
    global globNumSamples
    comm.Allreduce(localSum, res, op=MPI.SUM)

    global communicationTime
    communicationTime += time.perf_counter() - t0

    return jax.device_put(res / globNumSamples, global_defs.myDevice)


def global_covariance(data):
    """ Computes the covariance matrix of input data across MPI processes and device/batch dimensions.

    On each MPI process the input data is assumed to be a ``jax.numpy.array`` with a leading
    device dimension followed by a batch dimension and one data dimension.
    The data is reduced by computing the covariance
    matrix along device and batch dimensions as well as accross MPI processes. Hence, the result is
    an array of shape ``data.shape[2]`` :math:`\\times` ``data.shape[2]``.

    If no probabilities ``p`` are given, the empirical covariance is computed, i.e.,

        :math:`\\text{Cov}(X)=\\frac{1}{N_S}\sum_{j=1}^{N_S} X_j\\cdot X_j^\\dagger - \\bigg(\\frac{1}{N_S}\sum_{j=1}^{N_S} X_j\\bigg)\\cdot\\bigg(\\frac{1}{N_S}\sum_{j=1}^{N_S}X_j^\\dagger\\bigg)`

    Otherwise, the mean is computed using the given probabilities, i.e.,

        :math:`\\text{Cov}(X)=\sum_{j=1}^{N_S} p_jX_j\\cdot X_j^\\dagger - \\bigg(\sum_{j=1}^{N_S} p_jX_j\\bigg)\\cdot\\bigg(\sum_{j=1}^{N_S}p_jX_j^\\dagger\\bigg)`

    Arguments:
        * ``data``: Array of input data.
        * ``p``: Probabilities associated with the given data.

    Returns:
        Covariance matrix of data across MPI processes and device/batch dimensions.
    """

    jit_my_stuff()
    return global_mean(cov_helper_without_p(data))


def bcast_unknown_size(data, root=0):
    """ Broadcast a one-dimensional array.

    This function broadcasts the input data array to all MPI processes.

    Arguments:
        * ``data``: One dimensional array of datatype ``np.float64``.
        * ``root``: Rank of root process.

    Returns:
        On each MPI process the data received from the root process.
    """

    if rank == root:
        if data.dtype != np.float64:
            raise TypeError("Datatype has to be float64.")

    dim = None
    buf = None
    if rank == root:
        dim = len(data)
        comm.bcast(dim, root=root)
        buf = np.array(data)
    else:
        dim = comm.bcast(None, root=root)
        buf = np.empty(dim, dtype=np.float64)

    comm.Bcast([buf, dim, MPI.DOUBLE], root=root)

    return buf


def get_communication_time():
    global communicationTime
    t = communicationTime
    communicationTime = 0.
    return t
