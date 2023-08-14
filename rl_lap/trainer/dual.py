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

        orthogonality_error_matrix = jnp.tril(error_matrix_1 * error_matrix_2) * self.implicit_orthogonality_weight

        inner_dict = {
            f'inner({i},{j})': inner_product_matrix_1[i,j]
            for i, j in product(range(self.d), range(self.d))
            if i >= j
        }

        return orthogonality_error_matrix, inner_dict
    
    def compute_orthogonality_loss(self, params, orthogonality_error_matrix):
        # Compute the loss
        dual_variables = params['duals']
        if self.use_additive_duals:
            # Obtain diagonal of dual variables
            duals_diag = jnp.diag(dual_variables)

            # Obtain cumulative sum of diagonal
            duals_diag_cumsum = jnp.cumsum(duals_diag, axis=0)

            # Obtain the dual variables
            dual_variables = jnp.tril(dual_variables, k=-1) + jnp.diag(duals_diag_cumsum)

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
    
    def update_error_estimates(self, params, errors):   # TODO: Handle better the fact that params are an array
        old = params['errors']
        update = old + self.error_estimate_update_rate * (errors - old)   # The first update might be too large
        error_dict = {
            f'error({i},{j})': update[i,j]
            for i, j in product(range(self.d), range(self.d))
            if i >= j
        }
        return error_dict, update

    def loss_function(
            self, params, train_batch, **kwargs
        ) -> Tuple[jnp.ndarray]:

        # Get representations
        start_representation, end_representation, \
            constraint_representation_1, constraint_representation_2 \
                = self.encode_states(params['encoder'], train_batch)
        
        # Compute primal loss
        graph_loss = self.compute_graph_drawing_loss(
            start_representation, end_representation
        )
        orthogonality_error_matrix, inner_dict = self.compute_orthogonality_error_matrix(
            constraint_representation_1, constraint_representation_2,
        )

        # Compute dual loss
        orthogonality_loss, error_matrix, dual_dict = self.compute_orthogonality_loss(
           params, orthogonality_error_matrix)
        
        # Update error estimates
        error_dict, error_update = self.update_error_estimates(params, error_matrix)

        # Compute total loss
        lagrangian = graph_loss + orthogonality_loss
        loss = lagrangian
        metrics_dict = {
            'train_loss': lagrangian,
            'graph_loss': graph_loss,
            'reg_loss': orthogonality_loss,
            'barrier_loss': 0.0,
        }
        metrics_dict.update(inner_dict)
        metrics_dict.update(dual_dict)
        metrics_dict.update(error_dict)
        metrics = (loss, graph_loss, orthogonality_loss, metrics_dict)
        aux = (metrics, error_update)

        return loss, aux
    
    def update_duals(self, params):
        '''
            Update dual variables using some approximation 
            of the gradient of the lagrangian.
        '''
        error_matrix = params['errors']
        dual_variables = params['duals']
        updates = jnp.tril(error_matrix)

        if self.normalize_dual_updates:
            update_norm = jnp.linalg.norm(updates)
        else:
            update_norm = 1
        
        if self.use_decreasing_dual_lr:
            if self.dual_lr_decay_rate != 1:
                coeff_vector = jnp.array([self.dual_lr_decay_rate**i for i in range(self.d)])
            else:
                coeff_vector = jnp.arange(self.d, 0, -1) / self.d
        else:
            coeff_vector = jnp.ones(self.d)
        
        if self.decay_only_diagonal_duals:
            # Matrix with coefficients only in the diagonal
            diag_matrix = jnp.diag(coeff_vector)
            ones_matrix = jnp.ones((self.d, self.d))
            diag_matrix = diag_matrix + jnp.tril(ones_matrix, k=-1)
            updates = updates * diag_matrix
        else:
            updates = updates * coeff_vector.reshape(-1, 1)

        if self.use_additive_duals and self.use_predecessor_decay:
            # Obtain diagonal errors
            diag_errors = jnp.diag(error_matrix)
            predecessor_errors = jnp.zeros_like(diag_errors)
            predecessor_errors = predecessor_errors.at[1:].set(diag_errors[:-1])
            if self.use_predecessor_abs:
                predecessor_errors = jnp.abs(predecessor_errors)
            decays = jnp.exp(
                -self.predecessor_decay_coefficient 
                * predecessor_errors
            )
            if self.use_predecessor_clip:
                decays = jnp.clip(decays, 0, 1)
            ones_matrix = jnp.ones((self.d, self.d))
            diag_matrix = jnp.diag(decays) + jnp.tril(ones_matrix, k=-1)
            updates = updates * diag_matrix

        # Calculate updated duals depending on whether 
        # we optimize the log of the duals or not.
        if self.optimize_dual_logs:
            log_duals = jnp.log(dual_variables)
            updated_log_duals = log_duals + self.lr_duals * updates / update_norm
            updated_duals = jnp.exp(updated_log_duals)
        else:
            updated_duals = dual_variables + self.lr_duals * updates / update_norm

        # Clip duals to be in the range [min_duals, max_duals]
        updated_duals = jnp.clip(
            updated_duals,
            a_min=self.min_duals,
            a_max=self.max_duals,
        )   # TODO: Cliping is probably not the best way to handle this

        # Update params, making sure that the duals are lower triangular
        params['duals'] = jnp.tril(updated_duals)
        return params
    
    def reset_duals(self, params):
        """
            Reset dual variables that are larger than the 
            initial value to the initial value.
        """
        dual_variables = params['duals']
        
        if self.constant_reset:
            replace_matrix = self.dual_initial_val * jnp.ones_like(dual_variables)
        else:
            replace_matrix = self.reset_proportion * dual_variables

        updated_duals = jnp.where(
            dual_variables > self.dual_threshold,
            replace_matrix,
            dual_variables,
        )
        params['duals'] = jnp.tril(updated_duals)
        return params
    
    def update_training_state(self, params, error_update):
        '''Update error estimates'''

        params['errors'] = error_update
        return params
    
    def update_barrier_coefficients(self, params, *args, **kwargs):
        '''Leave params unchanged'''

        return params