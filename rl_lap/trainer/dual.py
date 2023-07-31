from typing import Tuple
from itertools import product
import numpy as np
import jax
import jax.numpy as jnp

import haiku as hk

from rl_lap.trainer.laplacian_encoder import LaplacianEncoderTrainer


class DualLaplacianEncoderTrainer(LaplacianEncoderTrainer):
    def compute_graph_drawing_loss(self, start_representation, end_representation):
        '''Compute reprensetation distances between start and end states'''
        
        loss = ((start_representation - end_representation)**2).sum(1).mean()
        return loss
    
    def compute_orthogonality_error_matrix(self, represetantation_batch_1, represetantation_batch_2):
        n = represetantation_batch_1.shape[0]

        inner_product_matrix_1 = jnp.einsum(
            'ij,ik->jk',
            represetantation_batch_1,
            jax.lax.stop_gradient(represetantation_batch_1),
        ) / n

        inner_product_matrix_2 = jnp.einsum(
            'ij,ik->jk',
            represetantation_batch_2,
            jax.lax.stop_gradient(represetantation_batch_2),
        ) / n

        error_matrix_1 = inner_product_matrix_1 - jnp.eye(self.d)
        error_matrix_2 = inner_product_matrix_2 - jnp.eye(self.d)

        orthogonality_error_matrix = jnp.tril(error_matrix_1 * error_matrix_2)

        inner_dict = {
            f'inner({i},{j})': inner_product_matrix_1[i,j]
            for i, j in product(range(self.d), range(self.d))
            if i >= j
        }

        return orthogonality_error_matrix, inner_dict
    
    def compute_orthogonality_loss(self, orthogonality_error_matrix):
        # Compute the loss
        dual_variables = self.dual_params

        if self.use_error_mean:
            coeff_vector = jnp.arange(self.d, 0, -1)
        else:
            coeff_vector = jnp.ones(self.d)

        error_matrix = jnp.tril(orthogonality_error_matrix - self.orthogonality_tolerance)
        orthogonality_loss = (jax.lax.stop_gradient(dual_variables) * error_matrix).dot(coeff_vector).sum()

        # Generate dictionary with dual variables and errors for logging 
        dual_dict = {
            f'beta({i},{j})': dual_variables[i,j]
            for i, j in product(range(self.d), range(self.d))
            if i >= j
        }
        
        return orthogonality_loss, error_matrix, dual_dict
    
    def update_error_estimates(self, errors):   # TODO: Handle better the fact that params are an array
        old = self.training_state['error']
        if old is None:
            update = errors
        else:
            update = old + self.error_estimate_update_rate * (errors - old)
        error_dict = {
            f'error({i},{j})': update[i,j]
            for i, j in product(range(self.d), range(self.d))
            if i >= j
        }
        return error_dict, update

    def loss_function(
            self, params_encoder, train_batch, **kwargs
        ) -> Tuple[jnp.ndarray]:

        # Get representations
        start_representation, end_representation, \
            constraint_representation_1, constraint_representation_2 \
                = self.encode_states(params_encoder, train_batch)
        
        # Compute primal loss
        graph_loss = self.compute_graph_drawing_loss(
            start_representation, end_representation
        )
        orthogonality_error_matrix, inner_dict = self.compute_orthogonality_error_matrix(
            constraint_representation_1, constraint_representation_2,
        )

        # Compute dual loss
        orthogonality_loss, error_matrix, dual_dict = self.compute_orthogonality_loss(
           orthogonality_error_matrix)
        
        # Update error estimates
        error_dict, error_update = self.update_error_estimates(error_matrix)

        # Compute total loss
        lagrangian = graph_loss + orthogonality_loss
        loss = lagrangian
        metrics_dict = {
            'train_loss': lagrangian,
            'graph_loss': graph_loss,
            'reg_loss': orthogonality_loss,
        }
        metrics_dict.update(inner_dict)
        metrics_dict.update(dual_dict)
        metrics_dict.update(error_dict)
        metrics = (loss, graph_loss, orthogonality_loss, metrics_dict)
        aux = (metrics, error_update)

        return loss, aux
    
    def update_duals(self):
        '''
            Update dual variables using some approximation 
            of the gradient of the lagrangian.
        '''
        error_matrix = self.training_state['error']
        dual_variables = self.dual_params

        # Calculate updated duals depending on whether 
        # we optimize the log of the duals or not.
        if self.optimize_dual_logs:
            log_duals = jnp.log(dual_variables)
            updates = jnp.tril(error_matrix)
            if self.normalize_dual_updates:
                update_norm = jnp.linalg.norm(updates)
            else:
                update_norm = 1
            updated_log_duals = log_duals + self.lr_duals * updates / update_norm
            updated_duals = jnp.exp(updated_log_duals)
        else:
            updates = jnp.tril(error_matrix * dual_variables)
            if self.normalize_dual_updates:
                update_norm = jnp.linalg.norm(updates)
            else:
                update_norm = 1

            updated_duals = dual_variables + self.lr_duals * updates / update_norm

        # Clip duals to be in the range [min_duals, max_duals]
        updated_duals = jnp.clip(
            updated_duals,
            a_min=self.min_duals,
            a_max=self.max_duals,
        )   # TODO: Cliping is probably not the best way to handle this

        # Update params, making sure that the duals are lower triangular
        self.dual_params = jnp.tril(updated_duals)
    
    def update_training_state(self, error_update):
        '''Update error estimates'''

        self.training_state['error'] = error_update