from typing import List
import haiku as hk
import jax
import numpy as np


def generate_layers(
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
            generate_layers(output_dim, hidden_dims, activation))

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
        ) -> None:
        super().__init__(name=name)
        self.conv = hk.Sequential([
            hk.Conv2D(output_channels=16, kernel_shape=4, stride=2, padding=(1,1),
                      w_init=hk.initializers.VarianceScaling(2.0, "fan_in",  "truncated_normal")),   
            jax.nn.relu,
            hk.Conv2D(output_channels=16, kernel_shape=4, stride=2, padding=(1,1),
                      w_init=hk.initializers.VarianceScaling(2.0, "fan_in",  "truncated_normal")),   
            jax.nn.relu,
            hk.Conv2D(output_channels=16, kernel_shape=4, stride=1, padding=(0,0),
                      w_init=hk.initializers.VarianceScaling(2.0, "fan_in",  "truncated_normal")),   
        ])
        self.flatten = hk.Flatten()
        self.linear = hk.Sequential(
            generate_layers(output_dim, hidden_dims, activation))

    def __call__(self, x: np.ndarray) -> jax.Array:
        '''Forward pass through the layers.'''
        x = self.conv(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
