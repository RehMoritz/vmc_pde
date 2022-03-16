import jax
import jax.numpy as jnp
import numpy as np


class AdaptiveHeun:
    """ This class implements an adaptive second order consistent integration scheme.

    Initializer arguments:
        * ``timeStep``: Initial time step (will be adapted automatically)
        * ``tol``: Tolerance for integration errors.
        * ``maxStep``: Maximal allowed time step.
    """

    def __init__(self, timeStep=1e-3, tol=1e-8, maxStep=1):
        self.dt = timeStep
        self.tolerance = tol
        self.maxStep = maxStep

    def step(self, t, f, y, normFunction=jnp.linalg.norm, **rhsArgs):
        """ This function performs an integration time step.

        For a first order ordinary differential equation (ODE) of the form

        :math:`\\frac{dy}{dt} = f(y, t, p)`

        where :math:`t` denotes the time and :math:`p` denotes further external parameters
        this function computes a second-order consistent integration step for :math:`y`.
        The time step :math:`\\Delta t` is chosen such that the integration error (quantified
        by a given norm) is smaller than the given tolerance.

        Arguments:
            * ``t``: Initial time.
            * ``f``: Right hand side of the ODE. This callable will be called as ``f(y, t, **rhsArgs, intStep=k)``, \
                    where k is an integer indicating the step number of the underlying Runge-Kutta integration scheme.
            * ``y``: Initial value of :math:`y`.
            * ``normFunction``: Norm function to be used to quantify the magnitude of errors.
            * ``**rhsArgs``: Further static arguments :math:`p` that will be passed to the right hand side function \
                    ``f(y, t, **rhsArgs, intStep=k)``.

        Returns:
            New value of :math:`y` and time step used :math:`\\Delta t`.
        """

        fe = 0.5

        dt = self.dt

        yInitial = y.copy()

        while fe < 1.:

            y = yInitial.copy()
            k0, _ = f(y, t, **rhsArgs, intStep=0)
            y += dt * k0
            k1, _ = f(y, t + dt, **rhsArgs, intStep=1)
            dy0 = 0.5 * dt * (k0 + k1)

            # now with half step size
            y -= 0.5 * dt * k0
            k10, _ = f(y, t + 0.5 * dt, **rhsArgs, intStep=2)
            dy1 = 0.25 * dt * (k0 + k10)
            y = yInitial + dy1
            k01, _ = f(y, t + 0.5 * dt, **rhsArgs, intStep=3)
            y += 0.5 * dt * k01
            k11, info = f(y, t + dt, **rhsArgs, intStep=4)
            dy1 += 0.25 * dt * (k01 + k11)

            # compute deviation
            # updateDiff = normFunction(dy1 - dy0, f.S0)
            updateDiff = normFunction(dy1 - dy0, f.SExp)
            fe = self.tolerance / updateDiff

            if 0.2 > 0.9 * fe**0.33333:
                tmp = 0.2
            else:
                tmp = 0.9 * fe**0.33333
            if tmp > 2.:
                tmp = 2.

            realDt = dt
            dt *= tmp

            if dt > self.maxStep:
                dt = self.maxStep

        # end while

        self.dt = dt

        return yInitial + dy1, realDt, info


class FixedStepper:
    """ This class implements an two integration schemes with fixed step sizes.

    Initializer arguments:
        * ``timeStep``: Initial time step (will be adapted automatically)
        * ``tol``: Tolerance for integration errors.
        * ``maxStep``: Maximal allowed time step.
    """

    def __init__(self, timeStep=1e-3, mode='Heun'):
        self.dt = timeStep
        self.mode = mode

    def step(self, t, f, y, normFunction=jnp.linalg.norm, **rhsArgs):
        """ This function performs an integration time step.

        For a first order ordinary differential equation (ODE) of the form

        :math:`\\frac{dy}{dt} = f(y, t, p)`

        Arguments:
            * ``t``: Initial time.
            * ``f``: Right hand side of the ODE. This callable will be called as ``f(y, t, **rhsArgs, intStep=k)``, \
                    where k is an integer indicating the step number of the underlying Runge-Kutta integration scheme.
            * ``y``: Initial value of :math:`y`.
            * ``normFunction``: Norm function to be used to quantify the magnitude of errors.
            * ``**rhsArgs``: Further static arguments :math:`p` that will be passed to the right hand side function \
                    ``f(y, t, **rhsArgs, intStep=k)``.

        Returns:
            New value of :math:`y` and time step used :math:`\\Delta t`.
        """

        yInitial = y.copy()

        if self.mode == 'Heun':
            y = yInitial.copy()
            k0, _ = f(y, t, **rhsArgs, intStep=0)
            y += self.dt * k0
            k1, info = f(y, t + self.dt, **rhsArgs, intStep=1)
            dy = 0.5 * self.dt * (k0 + k1)
            return yInitial + dy, self.dt, info

        if self.mode == 'Euler':
            y = yInitial.copy()
            k0, info = f(y, t, **rhsArgs, intStep=0)
            dy = self.dt * k0
            return yInitial + dy, self.dt, info
