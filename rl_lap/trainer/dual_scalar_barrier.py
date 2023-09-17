from typing import Tuple
from itertools import product
import numpy as np
import jax
import jax.numpy as jnp

import haiku as hk

from rl_lap.trainer.dual_exact import ExactDualLaplacianEncoderTrainer

from collections.abc import MutableMapping

def flatten_dict(dictionary, parent_key='', separator='_'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)

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
        if self.barrier_mean_or_sum == 'mean':
            squared_error = squared_error_matrix.mean()
        elif self.barrier_mean_or_sum == 'sum':
            squared_error = squared_error_matrix.sum()
        else:
            raise ValueError(f'barrier_mean_or_sum must be either "mean" or "sum".')
        barrier_loss = jax.lax.stop_gradient(barrier_coefficients[0,0]) * squared_error

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
    
    def update_barrier_coefficients_v1(self, params, *args, **kwargs):
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
    
    def update_barrier_coefficients_v2(self, params, *args, **kwargs):
        '''
            Update barrier coefficients using some approximation 
            of the barrier gradient in the modified lagrangian.
        '''
        barrier_coefficients = params['barrier_coefs']
        squared_error_matrix = params['squared_errors']
        updates = jnp.tril(squared_error_matrix - self.orthogonality_tolerance)
        updates = jnp.clip(updates, a_min=0, a_max=None).mean()

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
    
    def update_barrier_coefficients_v3(self, params, *args, **kwargs):
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
    
    def update_duals(self, params):
        '''
            Update dual variables using some approximation 
            of the gradient of the lagrangian.
        '''
        error_matrix = params['errors']
        dual_variables = params['duals']
        updates = jnp.tril(error_matrix)
        dual_velocities = params['dual_velocities']
        barrier_coefficient = params['barrier_coefs'][0,0]

        # Calculate updated duals
        barrier_coefficient = 1 +  self.use_barrier_for_duals * (barrier_coefficient - 1)
        lr = self.lr_duals * barrier_coefficient
        updated_duals = dual_variables + lr * updates

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
        
        # Update dual velocity
        updates = updated_duals - params['duals']

        norm_dual_velocities = jnp.linalg.norm(dual_velocities)
        init_coeff = jnp.isclose(norm_dual_velocities, 0.0, rtol=1e-10, atol=1e-13) 
        update_rate = init_coeff + (1 - init_coeff) * self.lr_dual_velocities
        updated_dual_velocities = dual_velocities + update_rate * (updates - dual_velocities)
        params['dual_velocities'] = updated_dual_velocities
        
        return params
    
    def get_eigenvalues_ordering_error(self, params):
        '''Get eigenvalues of the dual variables'''
        dual_variables = params['duals']
        eigenvalues = jnp.diag(dual_variables)
        ordering_error = (eigenvalues[:-1] - eigenvalues[1:]).clip(min=0).sum()
        return ordering_error
    
    def update_barrier_coefficients_v4(self, params, *args, **kwargs):
        '''
            Update barrier coefficients using some approximation 
            of the barrier gradient in the modified lagrangian.
        '''
        barrier_coefficients = params['barrier_coefs']
        ordering_error = self.get_eigenvalues_ordering_error(params)
        updates = jnp.exp(-self.barrier_beta / (ordering_error + 1e-10))
        
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
    
    def update_barrier_coefficients_v5(self, params, *args, **kwargs):
        '''
            Update barrier coefficients using some approximation 
            of the barrier gradient in the modified lagrangian.
        '''
        barrier_coefficients = params['barrier_coefs']
        grad_norms_dict = kwargs['grad_norms']

        # Get all gradients from the nested dictionary:
        grad_norms_list = list(flatten_dict(grad_norms_dict).values())
        grad_norms = jnp.stack(grad_norms_list)
        avg_grad_norm = grad_norms.mean()
        
        # Calculate updated coefficients
        updated_barrier_coefficients = barrier_coefficients + self.lr_barrier_coefs * avg_grad_norm

        # Clip coefficients to be in the range [min_barrier_coefs, max_barrier_coefs]
        updated_barrier_coefficients = jnp.clip(
            updated_barrier_coefficients,
            a_min=self.min_barrier_coefs,
            a_max=self.max_barrier_coefs,
        )   # TODO: Cliping is probably not the best way to handle this

        # Update params, making sure that the coefficients are lower triangular
        params['barrier_coefs'] = updated_barrier_coefficients
        return params