from typing import List
import haiku as hk
import jax
import numpy as np


def generate_layers(
        output_dim: int,
        hidden_dims: List[int],
    ) -> List[hk.Module]:
    '''Generate layers for MLP.'''
    layers = []
    for dim in hidden_dims:
        layers.append(hk.Linear(dim))
        layers.append(jax.nn.relu)
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
            name: str = 'MLP',
        ) -> None:
        super().__init__(name=name)
        self.sequential = hk.Sequential(
            generate_layers(output_dim, hidden_dims))

    def __call__(self, x: np.ndarray) -> jax.Array:
        '''Forward pass through the layers.'''
        return self.sequential(x)
