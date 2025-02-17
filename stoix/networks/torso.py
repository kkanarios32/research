from typing import Sequence

import chex
import numpy as np
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import Initializer, orthogonal

from stoix.networks.layers import NoisyLinear
from stoix.networks.utils import parse_activation_fn


class ContrastiveTorso(nn.Module):
    """
    MLP with residual connections: residual blocks have $block_size layers. Uses swish activation, optionally uses layernorm.
    """
    output_size: int
    width: int = 1024
    num_blocks: int = 1
    block_size: int = 2
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, x):
        lecun_uniform = nn.initializers.variance_scaling(
            1/3, "fan_in", "uniform")
        normalize = nn.LayerNorm() if self.use_layer_norm else (lambda x: x)

        # Start of net
        residual_stream = jnp.zeros((x.shape[0], self.width))

        # Main body
        for i in range(self.num_blocks):
            for j in range(self.block_size):
                x = nn.swish(
                    normalize(nn.Dense(self.width, kernel_init=lecun_uniform)(x)))
            x += residual_stream
            residual_stream = x

        # Last layer mapping to representation dimension
        x = nn.Dense(self.output_size, kernel_init=lecun_uniform)(x)
        return x


class MLPTorso(nn.Module):
    """MLP torso."""

    layer_sizes: Sequence[int]
    activation: str = "relu"
    use_layer_norm: bool = False
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))
    activate_final: bool = True

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        """Forward pass."""
        x = observation
        for layer_size in self.layer_sizes:
            x = nn.Dense(
                layer_size, kernel_init=self.kernel_init, use_bias=not self.use_layer_norm
            )(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            if self.activate_final or layer_size != self.layer_sizes[-1]:
                x = parse_activation_fn(self.activation)(x)
        return x


class NoisyMLPTorso(nn.Module):
    """MLP torso using NoisyLinear layers instead of standard Dense layers."""

    layer_sizes: Sequence[int]
    activation: str = "relu"
    use_layer_norm: bool = False
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))
    activate_final: bool = True
    sigma_zero: float = 0.5

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        x = observation
        for layer_size in self.layer_sizes:
            x = NoisyLinear(
                layer_size, sigma_zero=self.sigma_zero, use_bias=not self.use_layer_norm
            )(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            if self.activate_final or layer_size != self.layer_sizes[-1]:
                x = parse_activation_fn(self.activation)(x)
        return x


class CNNTorso(nn.Module):
    """2D CNN torso. Expects input of shape (batch, height, width, channels).
    After this torso, the output is flattened and put through an MLP of
    hidden_sizes."""

    channel_sizes: Sequence[int]
    kernel_sizes: Sequence[int]
    strides: Sequence[int]
    activation: str = "relu"
    use_layer_norm: bool = False
    kernel_init: Initializer = orthogonal(np.sqrt(2.0))
    channel_first: bool = False
    hidden_sizes: Sequence[int] = (256,)

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        """Forward pass."""
        x = observation
        # Move channels to the last dimension if they are first
        if self.channel_first:
            x = x.transpose((0, 2, 3, 1))
        # Convolutional layers
        for channel, kernel, stride in zip(self.channel_sizes, self.kernel_sizes, self.strides):
            x = nn.Conv(
                channel, (kernel, kernel), (stride,
                                            stride), use_bias=not self.use_layer_norm
            )(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(reduction_axes=(-3, -2, -1))(x)
            x = parse_activation_fn(self.activation)(x)

        # Flatten
        x = x.reshape(*observation.shape[:-3], -1)

        # MLP layers
        x = MLPTorso(
            layer_sizes=self.hidden_sizes,
            activation=self.activation,
            use_layer_norm=self.use_layer_norm,
            kernel_init=self.kernel_init,
            activate_final=True,
        )(x)

        return x
