from typing import Tuple

import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict
from flax.training.train_state import TrainState

from physics_engine import energy, rk4_step

vmap_rk4 = jax.vmap(rk4_step, in_axes=(0, None, None))
vmap_energy = jax.vmap(energy, in_axes=(0, None))


def generate_trajectories(
    rng: jax.random.PRNGKey,
    num_trajs: int = 256,
    num_steps: int = 200,
    dt: float = 0.05,
    gm: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    rng_pos, rng_vel = jax.random.split(rng)
    angles = jax.random.uniform(rng_pos, (num_trajs,), minval=0.0, maxval=2.0 * jnp.pi)
    radii = jax.random.uniform(rng_pos, (num_trajs,), minval=0.8, maxval=2.0)
    x0 = radii * jnp.cos(angles)
    y0 = radii * jnp.sin(angles)

    v_mags = jax.random.uniform(rng_vel, (num_trajs,), minval=0.4, maxval=1.2)
    v_angles = angles + jnp.pi / 2.0
    vx0 = v_mags * jnp.cos(v_angles)
    vy0 = v_mags * jnp.sin(v_angles)

    init_states = jnp.stack([x0, y0, vx0, vy0], axis=-1)

    def scan_step(state, _):
        next_state, _ = vmap_rk4(state, dt, gm)
        return next_state, state

    _, trajectory = jax.lax.scan(scan_step, init_states, length=num_steps)
    trajectory = jnp.swapaxes(trajectory, 0, 1)
    init_states_expanded = init_states[:, None, :]
    states = jnp.concatenate([init_states_expanded, trajectory], axis=1)

    targets = states[:, 1:, :]
    return states, targets


def angular_momentum(state: jnp.ndarray) -> jnp.ndarray:
    x, y, vx, vy = state[..., 0], state[..., 1], state[..., 2], state[..., 3]
    return x * vy - y * vx


def loss_fn(
    params: FrozenDict,
    state_in: jnp.ndarray,
    state_target: jnp.ndarray,
    apply_fn: callable,
    lambda_energy: float = 0.1,
    gm: float = 1.0,
    lambda_angular: float = 0.1,
) -> Tuple[jnp.ndarray, dict]:
    state_pred = apply_fn({"params": params}, state_in)

    mse_loss = jnp.mean((state_pred - state_target) ** 2)

    e_pred = vmap_energy(state_pred.reshape(-1, 4), gm)
    e_target = vmap_energy(state_target.reshape(-1, 4), gm)
    energy_loss = jnp.mean(jnp.abs(e_pred - e_target))

    L_pred = angular_momentum(state_pred)
    angular_loss = jnp.var(L_pred)

    total_loss = mse_loss + lambda_energy * energy_loss + lambda_angular * angular_loss

    metrics = {
        "loss": total_loss,
        "mse": mse_loss,
        "energy_loss": energy_loss,
        "angular_loss": angular_loss,
    }
    return total_loss, metrics


def create_train_state(
    rng: jax.random.PRNGKey,
    model: "OrbitMLP",
    learning_rate: float = 1e-3,
) -> TrainState:
    dummy_input = jnp.zeros((1, 4), dtype=jnp.float32)
    variables = model.init(rng, dummy_input)
    params = variables["params"]

    schedule = optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=2000, alpha=1e-4)
    tx = optax.adamw(schedule)

    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def make_train_step(
    model: "OrbitMLP",
    lambda_energy: float = 0.1,
    gm: float = 1.0,
    lambda_angular: float = 0.1,
):
    @jax.jit
    def train_step(
        state: TrainState,
        batch_in: jnp.ndarray,
        batch_target: jnp.ndarray,
    ) -> Tuple[TrainState, dict]:
        def _loss(params):
            return loss_fn(params, batch_in, batch_target, model.apply, lambda_energy, gm, lambda_angular)

        (_, metrics), grads = jax.value_and_grad(_loss, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, metrics

    return train_step


def make_predict_trajectory(model: "OrbitMLP"):
    def predict_trajectory(
        params: FrozenDict,
        init_state: jnp.ndarray,
        num_steps: int,
    ) -> jnp.ndarray:
        def scan_predict(s, _):
            next_s = model.apply({"params": params}, s[None, :])[0]
            return next_s, next_s

        _, trajectory = jax.lax.scan(scan_predict, init_state, length=num_steps)
        return jnp.concatenate([init_state[None, :], trajectory], axis=0)

    return predict_trajectory