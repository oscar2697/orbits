import jax
import jax.numpy as jnp
from physics_engine import kepler_dynamics, rk4_step, energy
from model import OrbitMLP
from train import (
    generate_trajectories,
    create_train_state,
    make_train_step,
    make_predict_trajectory,
)

print("Imports OK")

rng = jax.random.PRNGKey(0)
states, targets = generate_trajectories(rng, num_trajs=4, num_steps=10)
print(f"States: {states.shape}, Targets: {targets.shape}")

model = OrbitMLP()
ts = create_train_state(rng, model)
train_step = make_train_step(model)
s2, m = train_step(ts, jnp.zeros((2, 4)), jnp.zeros((2, 4)))
print(f"Train step OK. Loss={float(m['loss']):.4f}")

predict = make_predict_trajectory(model)
traj = predict(ts.params, jnp.ones(4), 10)
print(f"Predict trajectory shape: {traj.shape}")
print("All OK")