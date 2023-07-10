from typing import List
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class LowerTriangularParameterMatrix(hk.Module):
    '''Parameter matrix with lower triangular structure.'''
    def __init__(
        self, 
        dim_matrix: int = 1, 
        initial_weight: float = 1,
        name: str = 'LowerTriangularParameterMatrix', 
    ):
        super().__init__(name=name)
        self.dim_matrix = dim_matrix
        self.number_params = int(dim_matrix * (dim_matrix + 1) / 2)
        self.initial_weight = initial_weight
        self.lower_triangular_indices = np.tril_indices(dim_matrix)

    def __call__(self) -> jax.Array:
        '''Generate matrix with parameters in its lower triangular part.'''
        initializer = hk.initializers.Constant(self.initial_weight)
        dual_variables = hk.get_parameter(
            "duals", 
            shape=[self.number_params], 
            dtype=np.float32, 
            init=initializer,
        )
        matrix = jnp.zeros((self.dim_matrix, self.dim_matrix))
        matrix = matrix.at[self.lower_triangular_indices].set(dual_variables)
        return matrix
    

class ParameterMatrix(hk.Module):
    '''Parameter matrix.'''
    def __init__(
        self, 
        dim_matrix: int = 1, 
        initial_weight: float = 1,
        name: str = 'ParameterMatrix', 
    ):
        super().__init__(name=name)
        self.dim_matrix = dim_matrix
        self.initial_weight = initial_weight

    def __call__(self) -> jax.Array:
        '''Generate matrix with parameters in its lower triangular part.'''
        initializer = hk.initializers.Constant(self.initial_weight)
        dual_variables = jnp.tril(hk.get_parameter(
            "duals", 
            shape=[self.dim_matrix, self.dim_matrix], 
            dtype=np.float32, 
            init=initializer,
        ))
        return dual_variables
