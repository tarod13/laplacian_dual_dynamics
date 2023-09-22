from typing import Tuple
from itertools import product
import numpy as np
import jax
import jax.numpy as jnp

import haiku as hk

from src.trainer.generalized_augmented import GeneralizedAugmentedLagrangianTrainer

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

class AugmentedLagrangianSCZDTrainer(GeneralizedAugmentedLagrangianTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Matrix where each entry is the minimum of the corresponding row and column
        self.coefficient_matrix = jnp.minimum(
            jnp.arange(self.d,0,-1).reshape(-1,1),
            jnp.arange(self.d,0,-1).reshape(1,-1),
        )

    def compute_graph_drawing_loss(self, start_representation, end_representation):
        # Get vector of mononotically decreasing coefficients
        coeff_vector = jnp.arange(self.d, 0, -1)

        # Compute reprensetation distance between start and end states weighted by coeff_vector
        loss = ((start_representation - end_representation)**2).dot(coeff_vector).mean()

        # Normalize loss
        if self.coefficient_normalization:
            loss = loss / coeff_vector.sum()

        return loss

    def compute_orthogonality_error_matrix(self, represetantation_batch_1, represetantation_batch_2):
        n = represetantation_batch_1.shape[0]

        inner_product_matrix_1 = jnp.einsum(
            'ij,ik->jk',
            represetantation_batch_1,
            represetantation_batch_1,
        ) / n

        inner_product_matrix_2 = jnp.einsum(
            'ij,ik->jk',
            represetantation_batch_2,
            represetantation_batch_2,
        ) / n

        error_matrix_1 = inner_product_matrix_1 - jnp.eye(self.d)
        error_matrix_2 = inner_product_matrix_2 - jnp.eye(self.d)

        error_matrix = 0.5 * (error_matrix_1 + error_matrix_2)

        if self.use_abs_square_estimation:
            squared_error_matrix = jnp.abs(error_matrix_1) * jnp.abs(error_matrix_2) * self.implicit_orthogonality_weight
        else:
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

        orthogonality_loss = 0.0
        squared_error = (
            (self.coefficient_matrix * squared_error_matrix).sum() 
            / self.coefficient_matrix.sum()
        )

        if self.barrier_mean_or_sum == 'mean':
            squared_error = squared_error
        elif self.barrier_mean_or_sum == 'sum':
            squared_error = squared_error * self.d**2
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
        updates = squared_error_matrix - self.orthogonality_tolerance
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

        updates = squared_error_matrix - self.orthogonality_tolerance
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
        updates = error_matrix
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
        params['duals'] = jnp.zeros_like(updated_duals)
        
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
    
    def init_additional_params(self, *args, **kwargs):        
        additional_params = {
            'duals': self.dual_initial_val * jnp.ones(self.d, self.d),
            'barrier_coefs': self.barrier_initial_val * jnp.ones((1, 1)),
            'dual_velocities': jnp.zeros((self.d, self.d)),
            'errors': jnp.zeros((self.d, self.d)),
            'squared_errors': jnp.zeros((1, 1)),
        }
        return additional_params