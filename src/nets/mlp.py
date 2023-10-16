from typing import List
import haiku as hk
import jax
import numpy as np


def generate_fc_layers(
        output_dim: int,
        hidden_dims: List[int],
        activation: str = 'relu',
    ) -> List[hk.Module]:
    '''Generate layers for MLP.'''
    layers = []
    for dim in hidden_dims:
        layers.append(hk.Linear(dim))
        if activation == 'relu':
            layers.append(jax.nn.relu)
        elif activation == 'leaky_relu':
            layers.append(jax.nn.leaky_relu)
        else:
            raise NotImplementedError
    layers.append(hk.Linear(output_dim))
    return layers

def generate_conv_layers(
        n_conv_layers: int = 2,
        activation: str = 'relu',
        kernel_shape: int = 3,
    ) -> List[hk.Module]:
    '''Generate layers for MLP.'''
    layers = []
    for i in range(n_conv_layers-1):
        layers.append(hk.Conv2D(
            output_channels=16, kernel_shape=kernel_shape, stride=2, padding=(1,1),
            w_init=hk.initializers.VarianceScaling(2.0, "fan_in",  "truncated_normal")
        ))
        if activation == 'relu':
            layers.append(jax.nn.relu)
        elif activation == 'leaky_relu':
            layers.append(jax.nn.leaky_relu)
        else:
            raise NotImplementedError
    layers.append(hk.Conv2D(
        output_channels=16, kernel_shape=kernel_shape, stride=2, padding=(1,1),
        w_init=hk.initializers.VarianceScaling(2.0, "fan_in",  "truncated_normal")
    ))
    layers.append(jax.nn.relu)
    return layers

class MLP(hk.Module):
    '''
        Standard multi-layer perceptron.
    '''
    def __init__(
            self,
            output_dim: int,
            hidden_dims: List[int],
            activation: str = 'relu',
            name: str = 'MLP',
        ) -> None:
        super().__init__(name=name)
        self.sequential = hk.Sequential(
            generate_fc_layers(output_dim, hidden_dims, activation))

    def __call__(self, x: np.ndarray) -> jax.Array:
        '''Forward pass through the layers.'''
        return self.sequential(x)
    
class ConvNet(hk.Module):
    '''
        Standard multi-layer perceptron.
    '''
    def __init__(
            self,
            output_dim: int,
            hidden_dims: List[int],
            activation: str = 'relu',
            name: str = 'ConvNet',
            n_conv_layers: int = 2,
            kernel_shape: int = 3,
        ) -> None:
        super().__init__(name=name)
        self.conv = hk.Sequential(
            generate_conv_layers(n_conv_layers, activation, kernel_shape))
        self.flatten = hk.Flatten()
        self.linear = hk.Sequential(
            generate_fc_layers(output_dim, hidden_dims, activation))

    def __call__(self, x: np.ndarray) -> jax.Array:
        '''Forward pass through the layers.'''
        x = self.conv(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
