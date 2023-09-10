from typing import Tuple
from itertools import product
import numpy as np
import jax
import jax.numpy as jnp

import haiku as hk

from rl_lap.trainer.dual_exact import ExactDualLaplacianEncoderTrainer


class ScalarBarrierDualLaplacianEncoderTrainer(ExactDualLaplacianEncoderTrainer):
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
        barrier_loss = jax.lax.stop_gradient(barrier_coefficients[0,0]) * squared_error_matrix.mean()

        # Generate dictionary with dual variables and errors for logging 
        dual_dict = {
            f'beta({i},{j})': dual_variables[i,j]
            for i, j in product(range(self.d), range(self.d))
            if i >= j
        }
        barrier_dict = {
            f'barrier_coeff': barrier_coefficients[0,0],
        }
        dual_dict.update(barrier_dict)
        
        return orthogonality_loss, barrier_loss, dual_dict
    
    def update_barrier_coefficients_v1(self, params):
        '''Increase barrier coefficient by a given factor'''
        barrier_coefficients = params['barrier_coefs']
        squared_errors = params['errors']**2
        update_factors = jnp.where(
            squared_errors > self.barrier_threshold, 
            jnp.ones_like(squared_errors), 
            self.barrier_increase_factor * jnp.ones_like(squared_errors),
        )
        params['barrier_coefs'] = update_factors.mean() * barrier_coefficients   # TODO: consider some velocity term (if error is decreasing, don't update barrier coefficient)
        return params
    
    def update_barrier_coefficients_v2(self, params):
        '''
            Update barrier coefficients using some approximation 
            of the barrier gradient in the modified lagrangian.
        '''
        barrier_coefficients = params['barrier_coefs']
        squared_error_matrix = params['squared_errors']
        updates = jnp.tril(squared_error_matrix - self.orthogonality_tolerance).mean()

        # Calculate updated coefficients
        updated_barrier_coefficients = barrier_coefficients + self.lr_barrier_coefs * updates

        # Clip coefficients to be in the range [min_barrier_coefs, max_barrier_coefs]
        updated_barrier_coefficients = jnp.clip(
            updated_barrier_coefficients,
            a_min=self.min_barrier_coefs,
            a_max=self.max_barrier_coefs,
        )   # TODO: Cliping is probably not the best way to handle this

        # Update params, making sure that the coefficients are lower triangular
        params['barrier_coefs'] = updated_barrier_coefficients
        return params
    
    def update_barrier_coefficients_v3(self, params):
        '''
            Update barrier coefficients using some approximation 
            of the barrier gradient in the modified Lagrangian, and
            the change rate of the dual variables.
        '''
        barrier_coefficients = params['barrier_coefs']
        squared_error_matrix = params['squared_errors']
        dual_velocities = jnp.abs(params['dual_velocities'])

        updates = jnp.tril(squared_error_matrix - self.orthogonality_tolerance)
        updates = jnp.clip(updates, a_min=0, a_max=None)

        gates = jnp.exp(-self.barrier_beta / (dual_velocities + 1e-10))
        updates = (updates * gates).mean()

        # Calculate updated coefficients
        updated_barrier_coefficients = barrier_coefficients + self.lr_barrier_coefs * updates

        # Clip coefficients to be in the range [min_barrier_coefs, max_barrier_coefs]
        updated_barrier_coefficients = jnp.clip(
            updated_barrier_coefficients,
            a_min=self.min_barrier_coefs,
            a_max=self.max_barrier_coefs,
        )   # TODO: Cliping is probably not the best way to handle this

        # Update params, making sure that the coefficients are lower triangular
        params['barrier_coefs'] = updated_barrier_coefficients
        return params
    
    def loss_function(
            self, params, train_batch, **kwargs
        ) -> Tuple[jnp.ndarray]:
        loss, aux = super().loss_function(params, train_batch, **kwargs)
        barrier_coefficient = params['barrier_coefs'][0,0]

        # Normalize loss by barrier coefficient
        if self.use_barrier_normalization:
            loss /= jax.lax.stop_gradient(barrier_coefficient)

        return loss, aux