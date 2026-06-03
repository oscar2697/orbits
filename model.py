import flax.linen as nn
import jax.numpy as jnp
from flax.linen import initializers


class ResidualBlock(nn.Module):
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = nn.Dense(self.hidden_dim, kernel_init=initializers.he_normal())(x)
        h = nn.Dense(self.hidden_dim, kernel_init=initializers.he_normal())(x)
        h = nn.LayerNorm()(h)
        h = nn.gelu(h)
        h = nn.Dense(self.hidden_dim, kernel_init=initializers.he_normal())(h)
        h = nn.LayerNorm()(h)
        h = nn.gelu(h)
        return residual + h


class OrbitMLP(nn.Module):
    hidden_dim: int = 128
    num_blocks: int = 3

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, kernel_init=initializers.he_normal())(state)
        for _ in range(self.num_blocks):
            x = ResidualBlock(hidden_dim=self.hidden_dim)(x)
        out = nn.Dense(4, kernel_init=initializers.he_normal())(x)
        return out