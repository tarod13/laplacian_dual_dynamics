from typing import Tuple
from itertools import product
import numpy as np
import jax
import jax.numpy as jnp

import haiku as hk

from rl_lap.trainer.laplacian_encoder import LaplacianEncoderTrainer


class ExactDualLaplacianEncoderTrainer(LaplacianEncoderTrainer):
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

        error_matrix_1 = jnp.tril(inner_product_matrix_1 - jnp.eye(self.d))
        error_matrix_2 = jnp.tril(inner_product_matrix_2 - jnp.eye(self.d))

        error_matrix = 0.5 * (error_matrix_1 + error_matrix_2)
        squared_error_matrix = error_matrix_1 * error_matrix_2 * self.implicit_orthogonality_weight

        inner_dict = {
            f'inner({i},{j})': inner_product_matrix_1[i,j]
            for i, j in product(range(self.d), range(self.d))
            if i >= j
        }

        error_matrix_dict = {
            'errors': error_matrix,
            'squared_errors': squared_error_matrix,
        }

        return error_matrix_dict, inner_dict

    def compute_orthogonality_loss(self, params, error_matrix_dict):
        # Compute the losses
        dual_variables = params['duals']
        barrier_coefficients = params['barrier_coefs']
        error_matrix = error_matrix_dict['errors']
        squared_error_matrix = error_matrix_dict['squared_errors']

        if self.use_additive_duals:
            # Obtain diagonal of dual variables
            duals_diag = jnp.diag(dual_variables)

            # Obtain cumulative sum of diagonal
            duals_diag_cumsum = jnp.cumsum(duals_diag, axis=0)

            # Obtain the dual variables
            dual_variables = jnp.tril(dual_variables, k=-1) + jnp.diag(duals_diag_cumsum)   # This is only valid if they decrease monotonically

        if self.use_error_mean:
            coeff_vector = jnp.arange(self.d, 0, -1)
        else:
            coeff_vector = jnp.ones(self.d)

        orthogonality_loss = (jax.lax.stop_gradient(dual_variables) * error_matrix).dot(coeff_vector).sum()
        barrier_loss = (jax.lax.stop_gradient(barrier_coefficients) * squared_error_matrix).sum()

        # Generate dictionary with dual variables and errors for logging 
        dual_dict = {
            f'beta({i},{j})': dual_variables[i,j]
            for i, j in product(range(self.d), range(self.d))
            if i >= j
        }
        barrier_dict = {
            f'barrier({i},{j})': barrier_coefficients[i,j]
            for i, j in product(range(self.d), range(self.d))
            if i >= j
        }
        dual_dict.update(barrier_dict)
        
        return orthogonality_loss, barrier_loss, dual_dict
    
    def update_error_estimates(self, params, errors) -> Tuple[dict]:   # TODO: Handle better the fact that params are an array
        updates = {}
        for error_type in ['errors', 'squared_errors']:
            old = params[error_type]
            update_rate = self.error_estimate_update_rate if error_type == 'errors' else self.sq_error_estimate_update_rate
            update = old + update_rate * (errors[error_type] - old)   # The first update might be too large
            updates[error_type] = update
            if error_type == 'errors':
                error_dict = {
                    f'error({i},{j})': update[i,j]
                    for i, j in product(range(self.d), range(self.d))
                    if i >= j
                }
        return error_dict, updates

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
        error_matrix_dict, inner_dict = self.compute_orthogonality_error_matrix(
            constraint_representation_1, constraint_representation_2,
        )

        # Compute dual loss
        orthogonality_loss, barrier_loss, dual_dict = self.compute_orthogonality_loss(
           params, error_matrix_dict)
        
        # Update error estimates
        error_dict, error_update = self.update_error_estimates(params, error_matrix_dict)

        # Compute total loss
        lagrangian = graph_loss + orthogonality_loss + barrier_loss
        loss = lagrangian
        metrics_dict = {
            'train_loss': lagrangian,
            'graph_loss': graph_loss,
            'reg_loss': orthogonality_loss,
            'barrier_loss': barrier_loss,
        }
        metrics_dict.update(inner_dict)
        metrics_dict.update(dual_dict)
        metrics_dict.update(error_dict)
        metrics = (loss, graph_loss, orthogonality_loss, barrier_loss, metrics_dict)
        aux = (metrics, error_update)

        return loss, aux
    
    def calculate_additive_duals(self, dual_variables):
        '''Calculate the dual variables using the additive method'''
        duals_diag = jnp.diag(dual_variables)
        duals_diag_cumsum = jnp.cumsum(duals_diag, axis=0)
        additive_dual_variables = jnp.tril(dual_variables, k=-1) + jnp.diag(duals_diag_cumsum)   # This is only valid if they decrease monotonically
        return additive_dual_variables

    def clip_duals(self, dual_variables):
        duals_diag = jnp.diag(dual_variables)
        striclty_non_positive_duals_diag = jnp.where(duals_diag > 0, 0, duals_diag)
        clipped_dual_variables = jnp.tril(dual_variables, k=-1) + jnp.diag(striclty_non_positive_duals_diag)
        return clipped_dual_variables
    
    def additional_update_step(self, step, params, *args, **kwargs):
        # Update the dual parameters
        is_dual_update_step = (
            (((step + 1) % self.update_dual_every) == 0)
            and (step > self.update_dual_after)
        )
        if is_dual_update_step:
            params = self.update_duals(params)

        is_dual_reset_step = (
            (((step + 1) % self.reset_dual_every) == 0)
            and (step > self.update_dual_after)
        )
        if is_dual_reset_step:
            params = self.reset_duals(params)

        is_barrier_update_step = (
            ((step + 1) % self.update_barrier_every) == 0
        )
        if is_barrier_update_step:
            params = self.update_barrier_coefficients(params)
        
        return params
    
    def update_duals(self, params):
        '''
            Update dual variables using some approximation 
            of the gradient of the lagrangian.
        '''
        error_matrix = params['errors']
        dual_variables = params['duals']
        updates = jnp.tril(error_matrix)

        # Calculate updated duals
        updated_duals = dual_variables + self.lr_duals * updates

        # Clip duals to be in the range [min_duals, max_duals]
        updated_duals = jnp.clip(
            updated_duals,
            a_min=self.min_duals,
            a_max=self.max_duals,
        )   # TODO: Cliping is probably not the best way to handle this

        # Clip diagonals to be negative
        if self.use_additive_duals:
            updated_duals = self.clip_duals(updated_duals)

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

        params['errors'] = error_update['errors']
        params['squared_errors'] = error_update['squared_errors']
        return params
    
    def update_barrier_coefficients(self, params):   # TODO: eliminate this function when the best version is found
        if not self.use_barrier_update_v2:
            params = self.update_barrier_coefficients_v1(params)
        else:
            params = self.update_barrier_coefficients_v2(params)
        return params

    def update_barrier_coefficients_v1(self, params):
        '''Increase barrier coefficient by a given factor'''
        barrier_coefficients = params['barrier_coefs']
        squared_errors = params['errors']**2
        update_factors = jnp.where(
            squared_errors > self.barrier_threshold, 
            jnp.ones_like(squared_errors), 
            self.barrier_increase_factor * jnp.ones_like(squared_errors),
        )
        params['barrier_coefs'] = update_factors * barrier_coefficients   # TODO: consider some velocity term (if error is decreasing, don't update barrier coefficient)
        return params
    
    def update_barrier_coefficients_v2(self, params):
        '''
            Update barrier coefficients using some approximation 
            of the barrier gradient in the modified lagrangian.
        '''
        barrier_coefficients = params['barrier_coefs']
        squared_error_matrix = params['squared_errors']
        updates = jnp.tril(squared_error_matrix - self.orthogonality_tolerance).clip(min=0)

        # Calculate updated coefficients
        updated_barrier_coefficients = barrier_coefficients + self.lr_barrier_coefs * updates

        # Clip coefficients to be in the range [min_barrier_coefs, max_barrier_coefs]
        updated_barrier_coefficients = jnp.clip(
            updated_barrier_coefficients,
            a_min=self.min_barrier_coefs,
            a_max=self.max_barrier_coefs,
        )   # TODO: Cliping is probably not the best way to handle this

        # Update params, making sure that the coefficients are lower triangular
        params['barrier_coefs'] = jnp.tril(updated_barrier_coefficients)
        return params