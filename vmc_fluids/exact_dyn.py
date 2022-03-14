import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_samples(coords, lim=12):
    plt.figure()
    plt.hist2d(coords[:, 0], coords[:, 1], bins=100, range=[[-lim, lim], [-lim, lim]])
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.show()


def _velocity_field_hamiltonian(coord):
    """returns dx/dt = p, dp/dt = -x"""
    def H(x):
        lam = 1e-1
        return jnp.pi * (jnp.sum(x**2) + lam * x[0]**4)
    grads = jax.grad(H)(coord)
    return jnp.array([grads[1], -grads[0]])


def integrate_single_coord(coord, dt):
    k1 = _velocity_field_hamiltonian(coord)
    k2 = _velocity_field_hamiltonian(coord + dt * 0.5 * k1)
    k3 = _velocity_field_hamiltonian(coord + dt * 0.5 * k2)
    k4 = _velocity_field_hamiltonian(coord + dt * k3)

    return coord + dt * (k1 + 2. * k2 + 2. * k3 + k4) / 6.


@jax.jit
def integrate(coords, dt):
    return jax.vmap(integrate_single_coord, in_axes=(0, None))(coords, dt)


dim = 2
N_s = 100000
offset = jnp.ones(dim) * 1

coords = jax.random.normal(jax.random.PRNGKey(0), (N_s, dim)) + offset

t = 0
t_end = 5
dt = 1e-2
plot_every = 1e-2

while t < t_end:
    coords = integrate(coords, dt)
    t += dt

    print(f"\t t = {t}")
    if jnp.abs(jnp.round(t, 3) % plot_every) < 1e-6:
        plot_samples(coords)
