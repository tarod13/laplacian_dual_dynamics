from typing import List

import numpy as np
import jax
import flax.linen as nn

class MLP(nn.Module):
    '''
        Standard multi-layer perceptron.
    '''
    # Set attributes
    output_dims: List[int]
    
    def setup(self) -> None:
        # Generate layers
        self.layers = [nn.Dense(dim) for dim in self.output_dims]

    def __call__(self, x: np.ndarray) -> jax.Array:
        '''Forward pass through the layers.'''
        for layer in self.layers[:-1]:
            x = nn.relu(layer(x))
        x = self.layers[-1](x)
        return x
