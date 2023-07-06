from typing import List
import haiku as hk
import jax
import numpy as np

from rl_lap.nets import LTriangular_kh as LTriangular

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
    

class DualCoefficientExtendedMLP(hk.Module):
    '''
        Multi-layer perceptron with dual coefficients 
        that have an upper triangular structure.
    '''
    def __init__(
        self,
        ModelClass: type,
        *args,
        name: str = 'DualCoefficientExtendedMLP',
        **kwargs,
    ) -> None:
        super().__init__(name=name)

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
