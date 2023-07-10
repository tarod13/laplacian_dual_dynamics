from typing import List
import haiku as hk
import jax
import numpy as np

from rl_lap.nets import LTriangular_kh as LTriangular, ParameterMatrix_kh as Matrix

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
        output_dim: int,
        hidden_dims: List[int],
        dual_initial_val: float = 1.0,
        use_lower_triangular: bool = True,
        name: str = 'DualCoefficientExtendedMLP',
    ) -> None:
        super().__init__(name=name)

        # Generate model
        self.model = ModelClass(output_dim, hidden_dims)

        # Generate dual variables
        if use_lower_triangular:
            ParamClass = LTriangular
        else:
            ParamClass = Matrix

        self.dual_variables = ParamClass(
            dim_matrix=output_dim,   # TODO: handle this better
            initial_weight=dual_initial_val,
        )

        self.error_estimates = ParamClass(
            dim_matrix=output_dim,   # TODO: handle this better
            initial_weight=0.0,
        )

        self.error_accumulation = ParamClass(
            dim_matrix=output_dim,   # TODO: handle this better
            initial_weight=0.0,
        )

        self.first_call = True

    def __call__(self, x: np.ndarray) -> jax.Array:
        # Return model output
        if self.first_call:
            self.first_call = False
            duals = self.dual_variables()
            error_estimates = self.error_estimates()
            error_accumulation = self.error_accumulation()
        return self.model(x)
    
    def get_duals(self) -> jax.Array:
        return self.dual_variables()
    
    def get_errors(self) -> jax.Array:
        return self.error_estimates()
    
    def get_error_accumulation(self) -> jax.Array:
        return self.error_accumulation()
