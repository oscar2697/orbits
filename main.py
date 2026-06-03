import os
from datetime import datetime

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from flax import serialization

from checks import run_checks
from model import OrbitMLP
from physics_engine import energy as compute_energy, rk4_step
from train import (
    create_train_state,
    generate_trajectories,
    make_predict_trajectory,
    make_train_step,
)


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_checks()

    config = load_config("config.yaml")

    seed = config["seed"]
    num_trajs = config["data"]["num_trajs"]
    num_steps = config["data"]["num_steps"]
    dt = config["data"]["dt"]
    gm = config["physics"]["gm"]
    lambda_energy = config["physics"].get("lambda_energy", 0.1)
    lambda_angular = config["physics"].get("lambda_angular", 0.1)
    num_epochs = config["training"]["num_epochs"]
    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]
    hidden_dim = config["model"]["hidden_dim"]
    num_blocks = config["model"]["num_blocks"]

    rng = jax.random.PRNGKey(seed)

    rng_data, rng_init, rng_shuffle_st = jax.random.split(rng, 3)
    states, targets = generate_trajectories(
        rng_data, num_trajs=num_trajs, num_steps=num_steps, dt=dt, gm=gm
    )

    B, T, D = states.shape
    flat_inputs = states[:, :-1, :].reshape(-1, 4)
    flat_targets = targets.reshape(-1, 4)
    num_samples = flat_inputs.shape[0]

    model = OrbitMLP(hidden_dim=hidden_dim, num_blocks=num_blocks)
    train_state = create_train_state(rng_init, model, learning_rate=learning_rate)

    train_step = make_train_step(model, lambda_energy=lambda_energy, gm=gm, lambda_angular=lambda_angular)

    print(f"Training on {num_samples} samples for {num_epochs} epochs...")
    print(f"{'Epoch':>8s} {'Loss':>10s} {'MSE':>10s} {'Energy':>10s}")

    rng_shuffle = rng_shuffle_st
    for epoch in range(num_epochs):
        rng_shuffle, rng_epoch = jax.random.split(rng_shuffle)
        perm = jax.random.permutation(rng_epoch, num_samples)
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_energy_loss = 0.0
        num_batches = 0

        for start in range(0, num_samples, batch_size):
            indices = perm[start : start + batch_size]
            batch_in = flat_inputs[indices]
            batch_target = flat_targets[indices]

            train_state, metrics = train_step(
                train_state, batch_in, batch_target
            )
            epoch_loss += float(metrics["loss"])
            epoch_mse += float(metrics["mse"])
            epoch_energy_loss += float(metrics["energy_loss"])
            num_batches += 1

        if (epoch + 1) % 30 == 0:
            print(
                f"{epoch + 1:>8d} "
                f"{epoch_loss / num_batches:>10.6f} "
                f"{epoch_mse / num_batches:>10.6f} "
                f"{epoch_energy_loss / num_batches:>10.6f}"
            )

    bytes_params = serialization.to_bytes(train_state.params)
    model_path = os.path.join("models", f"orbitmlp_{now}.flax")
    with open(model_path, "wb") as f:
        f.write(bytes_params)
    print(f"Model saved to {model_path}")

    rng_test = jax.random.PRNGKey(999)
    test_states, _ = generate_trajectories(
        rng_test, num_trajs=1, num_steps=num_steps, dt=dt, gm=gm
    )
    init_state_arr = test_states[0, 0]

    rk4_traj = np.zeros((num_steps + 1, 4), dtype=np.float32)
    rk4_traj[0] = np.array(init_state_arr)
    s = init_state_arr
    for i in range(num_steps):
        s, _ = rk4_step(s, dt, gm)
        rk4_traj[i + 1] = np.array(s)

    predict_trajectory = make_predict_trajectory(model)
    nn_traj = np.array(
        predict_trajectory(train_state.params, init_state_arr, num_steps)
    )

    rk4_energies = np.array(
        [compute_energy(rk4_traj[i], gm) for i in range(num_steps + 1)]
    )
    nn_energies = np.array(
        [compute_energy(nn_traj[i], gm) for i in range(num_steps + 1)]
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(rk4_traj[:, 0], rk4_traj[:, 1], "b-", label="RK4 (truth)", alpha=0.8)
    ax.plot(nn_traj[:, 0], nn_traj[:, 1], "r--", label="OrbitMLP", alpha=0.8)
    ax.scatter(0, 0, c="k", marker="o", s=80, label="Central mass")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Orbit Comparison")
    ax.legend()
    ax.set_aspect("equal")

    ax = axes[1]
    ax.plot(range(num_steps + 1), rk4_traj[:, 0], "b-", label="RK4 x", alpha=0.8)
    ax.plot(range(num_steps + 1), nn_traj[:, 0], "r--", label="NN x", alpha=0.8)
    ax.plot(range(num_steps + 1), rk4_traj[:, 1], "c-", label="RK4 y", alpha=0.8)
    ax.plot(range(num_steps + 1), nn_traj[:, 1], "m--", label="NN y", alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Position")
    ax.set_title("Position vs Time")
    ax.legend()

    ax = axes[2]
    ax.plot(range(num_steps + 1), rk4_energies, "b-", label="RK4 energy", alpha=0.8)
    ax.plot(range(num_steps + 1), nn_energies, "r--", label="NN energy", alpha=0.8)
    initial_e = rk4_energies[0]
    ax.axhline(y=initial_e, color="k", linestyle=":", alpha=0.5, label=f"E0 = {initial_e:.4f}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Total Energy")
    ax.set_title("Energy Conservation")
    ax.legend()

    plt.tight_layout()
    plot_filename = f"orbit_{now}.png"
    plot_path = os.path.join("results", plot_filename)
    plt.savefig(plot_path, dpi=150)
    plt.close()

    avg_loss = epoch_loss / num_batches
    avg_mse = epoch_mse / num_batches
    avg_e = epoch_energy_loss / num_batches
    print(f"Plot saved to {plot_path}")
    print(f"Final loss: {avg_loss:.6f} | MSE: {avg_mse:.6f} | Energy: {avg_e:.6f}")


if __name__ == "__main__":
    main()