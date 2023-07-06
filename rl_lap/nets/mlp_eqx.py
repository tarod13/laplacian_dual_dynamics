from typing import List

import numpy as np
import jax
import equinox as eqx

from rl_lap.nets import LTriangular_eqx as LTriangular

def generate_layers(
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        key: jax.random.PRNGKey,
    ) -> List[eqx.Module]:
    '''Generate layers for MLP.'''
    layers = []
    for dim in hidden_dims:
        layers.append(eqx.nn.Linear(input_dim, dim, key=key))
        input_dim = dim
    layers.append(eqx.nn.Linear(input_dim, output_dim, key=key))
    return layers


class MLP(eqx.Module):
    '''
        Standard multi-layer perceptron.
    '''
    # Set attributes
    layers: List[eqx.Module]
    
    def __init__(
        self,
        input_dim: int, 
        output_dim: int, 
        hidden_dims: List[int],
        key: jax.random.PRNGKey,
        **kwargs,
    ) -> None:
        # Generate layers
        self.layers = generate_layers(
            input_dim, output_dim, hidden_dims, key)   # TODO: Add initialization

    def __call__(self, x: np.ndarray) -> jax.Array:
        '''Forward pass through the layers.'''
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        x = self.layers[-1](x)
        return x
    

class DualCoefficientExtendedMLP(eqx.Module):
    '''
        Multi-layer perceptron with dual coefficients 
        that have an upper triangular structure.
    '''
    # Set attributes
    model: eqx.Module
    dual_variables: LTriangular
    
    def __init__(
        self,
        ModelClass: type,
        *args,
        **kwargs,
    ) -> None:
        # Generate model
        self.model = ModelClass(*args, **kwargs)

        # Generate dual variables
        self.dual_variables = LTriangular(
            dim_matrix=kwargs.get("d", 10),   # TODO: handle this better
            initial_weight=kwargs.get("dual_initial_val", 1.0),
        )

    def __call__(self, x: np.ndarray) -> jax.Array:
        # Return model output
        return self.model(x)
    
    def get_duals(self) -> jax.Array:
        '''Return dual variables.'''
        return self.dual_variables()
    
