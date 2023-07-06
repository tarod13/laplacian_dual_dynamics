import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx


class LowerTriangularParameterMatrix(eqx.Module):
    '''Parameter matrix with lower triangular structure.'''\
    
    # Set attributes
    dim_matrix: int
    number_params: int
    matrix_params: jax.Array
    lower_triangular_indices: np.ndarray
    
    def __init__(
        self, 
        dim_matrix: int = 1, 
        initial_weight: float = 1, 
    ):
        # Store attributes
        self.dim_matrix = dim_matrix
        self.number_params = int(dim_matrix * (dim_matrix + 1) / 2)

        # Initialize parameters and corresponding indices
        self.matrix_params = initial_weight * jnp.ones((self.number_params,))
        self.lower_triangular_indices = np.tril_indices(dim_matrix)

    def __call__(self) -> jax.Array:
        '''Generate matrix with parameters in its lower triangular part.'''

        matrix = jnp.zeros((self.dim_matrix, self.dim_matrix))
        matrix = matrix.at[self.lower_triangular_indices].set(self.matrix_params)
        return matrix
