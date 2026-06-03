import glob
import os
import pickle

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flax import serialization

from model import OrbitMLP
from physics_engine import energy as compute_energy, rk4_step
from train import angular_momentum, make_predict_trajectory


def load_latest_model(models_dir: str = "models"):
    flax_files = glob.glob(os.path.join(models_dir, "*.flax"))
    if not flax_files:
        raise FileNotFoundError(f"No .flax files found in {models_dir}/")
    latest = max(flax_files, key=os.path.getmtime)
    print(f"Loading model: {latest}")
    with open(latest, "rb") as f:
        bytes_params = f.read()
    return serialization.from_bytes(jax.random.PRNGKey(0), bytes_params)


def generate_random_initial_state(rng, gm: float = 1.0):
    rng_pos, rng_vel = jax.random.split(rng)
    angle = jax.random.uniform(rng_pos, (), minval=0.0, maxval=2.0 * jnp.pi)
    radius = jax.random.uniform(rng_pos, (), minval=1.0, maxval=2.0)
    x0 = radius * jnp.cos(angle)
    y0 = radius * jnp.sin(angle)

    v_mag = jax.random.uniform(rng_vel, (), minval=0.5, maxval=1.1)
    v_angle = angle + jnp.pi / 2.0
    vx0 = v_mag * jnp.cos(v_angle)
    vy0 = v_mag * jnp.sin(v_angle)

    return jnp.array([x0, y0, vx0, vy0], dtype=jnp.float32)


def main():
    os.makedirs("results", exist_ok=True)

    gm = 1.0
    dt = 0.05
    num_steps = 500

    params = load_latest_model()
    model = OrbitMLP()

    rng = jax.random.PRNGKey(0)
    init_state = generate_random_initial_state(rng, gm=gm)
    print(f"Initial state: {init_state}")

    predict_trajectory = make_predict_trajectory(model)
    nn_traj = np.array(predict_trajectory(params, init_state, num_steps))

    rk4_traj = np.zeros((num_steps + 1, 4), dtype=np.float32)
    rk4_traj[0] = np.array(init_state)
    s = init_state
    for i in range(num_steps):
        s, _ = rk4_step(s, dt, gm)
        rk4_traj[i + 1] = np.array(s)

    rk4_energy = np.array([compute_energy(rk4_traj[i], gm) for i in range(num_steps + 1)])
    nn_energy = np.array([compute_energy(nn_traj[i], gm) for i in range(num_steps + 1)])

    rk4_L = np.array([angular_momentum(rk4_traj[i]) for i in range(num_steps + 1)])
    nn_L = np.array([angular_momentum(nn_traj[i]) for i in range(num_steps + 1)])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(rk4_traj[:, 0], rk4_traj[:, 1], "b-", label="RK4 (truth)", alpha=0.7)
    ax.plot(nn_traj[:, 0], nn_traj[:, 1], "r--", label="OrbitMLP", alpha=0.7)
    ax.scatter(0, 0, c="k", marker="o", s=80, label="Central mass")
    ax.scatter(rk4_traj[0, 0], rk4_traj[0, 1], c="g", marker="x", s=100, label="Start")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Orbit Comparison (500 steps)")
    ax.legend()
    ax.set_aspect("equal")

    ax = axes[1]
    ax.plot(range(num_steps + 1), rk4_energy, "b-", label="RK4 energy", alpha=0.7)
    ax.plot(range(num_steps + 1), nn_energy, "r--", label="NN energy", alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Total Energy")
    ax.set_title("Energy Conservation")
    ax.legend()

    ax = axes[2]
    ax.plot(range(num_steps + 1), rk4_L, "b-", label="RK4 angular momentum", alpha=0.7)
    ax.plot(range(num_steps + 1), nn_L, "r--", label="NN angular momentum", alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Angular Momentum L")
    ax.set_title("Angular Momentum Conservation")
    ax.legend()

    plt.tight_layout()
    out_path = "results/inference_orbit.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Inference plot saved to {out_path}")

    mse_pos = np.mean((nn_traj - rk4_traj) ** 2)
    print(f"Position MSE vs RK4: {mse_pos:.6e}")
    print(f"NN energy drift (final - initial): {nn_energy[-1] - nn_energy[0]:.6e}")
    print(f"RK4 energy drift (final - initial): {rk4_energy[-1] - rk4_energy[0]:.6e}")
    print(f"NN L variance: {np.var(nn_L):.6e}")
    print(f"RK4 L variance: {np.var(rk4_L):.6e}")


if __name__ == "__main__":
    main()