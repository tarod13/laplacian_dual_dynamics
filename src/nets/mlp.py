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
