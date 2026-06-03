from typing import Tuple

import jax
import jax.numpy as jnp

State = jax.Array


def kepler_dynamics(state: State, gm: float = 1.0) -> State:
    x, y, vx, vy = state
    r2 = x * x + y * y
    r3 = r2 * jnp.sqrt(r2)
    inv_r3 = jnp.where(r3 > 1e-12, 1.0 / r3, 0.0)
    ax = -gm * x * inv_r3
    ay = -gm * y * inv_r3
    return jnp.stack([vx, vy, ax, ay])


def rk4_step(state: State, dt: float = 0.01, gm: float = 1.0) -> Tuple[State, State]:
    k1 = kepler_dynamics(state, gm)
    k2 = kepler_dynamics(state + 0.5 * dt * k1, gm)
    k3 = kepler_dynamics(state + 0.5 * dt * k2, gm)
    k4 = kepler_dynamics(state + dt * k3, gm)

    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return next_state, k1


def energy(state: State, gm: float = 1.0) -> jax.Array:
    x, y, vx, vy = state
    kinetic = 0.5 * (vx * vx + vy * vy)
    r = jnp.sqrt(x * x + y * y)
    potential = -gm / jnp.where(r > 1e-12, r, 1e12)
    return kinetic + potential