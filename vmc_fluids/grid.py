import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn


class Grid():

    def __init__(self, bounds, n_gridpoints):
        self.dim = bounds.shape[0]
        self.bounds = bounds
        self.n_gridpoints = n_gridpoints
        self.widths = 2 * self.bounds / self.n_gridpoints
        self.bin_area = np.prod(self.widths)

        self.vals = [np.arange(-bound, bound, width) for bound, width in zip(self.bounds, self.widths)]

        self.meshgrid = np.meshgrid(*self.vals)

        self.coords = np.array(self.meshgrid)
        self.coords = np.moveaxis(np.array(self.meshgrid), 0, -1).reshape(self.n_gridpoints**self.dim, self.dim)


if __name__ == "__main__":
    bounds = np.array([2., 2.])
    n_gridpoints = 5
    Grid(bounds, n_gridpoints)

    x, y = np.meshgrid(np.arange(3), np.arange(3))
