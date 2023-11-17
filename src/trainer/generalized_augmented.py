from typing import Tuple
from abc import ABC, abstractmethod
from itertools import product
import numpy as np
import jax
import jax.numpy as jnp

import haiku as hk

from src.trainer.laplacian_encoder import LaplacianEncoderTrainer


class GeneralizedAugmentedLagrangianTrainer(LaplacianEncoderTrainer, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Matrix where each entry is the minimum of the corresponding row and column
        self.coefficient_vector = jnp.ones(self.d)

    def compute_graph_drawing_loss(self, start_representation, end_representation):
        '''Compute reprensetation distances between start and end states'''
        
        graph_induced_norms = ((start_representation - end_representation)**2).mean(0)
        loss = graph_induced_norms.dot(self.coefficient_vector)
        
        graph_induced_norm_dict = {
            f'graph_norm({i})': graph_induced_norms[i]
            for i in range(self.d)
        }       

        return loss, graph_induced_norm_dict
    
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
        quadratic_error_matrix = error_matrix_1 * error_matrix_2

        inner_dict = {
            f'inner({i},{j})': inner_product_matrix_1[i,j]
            for i, j in product(range(self.d), range(self.d))
            if i >= j
        }

        error_matrix_dict = {
            'errors': error_matrix,
            'quadratic_errors': quadratic_error_matrix,
        }

        return error_matrix_dict, inner_dict

    def compute_orthogonality_loss(self, params, error_matrix_dict):
        # Compute the losses
        dual_variables = params['duals']
        barrier_coefficients = params['barrier_coefs']
        error_matrix = error_matrix_dict['errors']
        quadratic_error_matrix = error_matrix_dict['quadratic_errors']

        dual_loss = (jax.lax.stop_gradient(dual_variables) * error_matrix).sum()
        barrier_loss = (jax.lax.stop_gradient(barrier_coefficients) * quadratic_error_matrix).sum()

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
        
        return dual_loss, barrier_loss, dual_dict
    
    def update_error_estimates(self, params, errors) -> Tuple[dict]:   # TODO: Handle better the fact that params are an array
        updates = {}
        for error_type in ['errors', 'quadratic_errors']:
            # Get old error estimates
            old = params[error_type]
            norm_old = jnp.linalg.norm(old)
            
            # Set update rate to 1 in the first iteration
            init_coeff = jnp.isclose(norm_old, 0.0, rtol=1e-10, atol=1e-13) 
            non_init_update_rate = self.error_update_rate if error_type == 'errors' else self.q_error_update_rate
            update_rate = init_coeff + (1 - init_coeff) * non_init_update_rate
            
            # Update error estimates
            update = old + update_rate * (errors[error_type] - old)   # The first update might be too large
            updates[error_type] = update
            
            # Generate dictionary with error estimates for logging
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
        graph_loss, graph_induced_norm_dict = self.compute_graph_drawing_loss(
            start_representation, end_representation
        )
        error_matrix_dict, inner_dict = self.compute_orthogonality_error_matrix(
            constraint_representation_1, constraint_representation_2,
        )

        # Compute dual loss
        dual_loss, barrier_loss, dual_dict = self.compute_orthogonality_loss(
        params, error_matrix_dict)
        
        # Update error estimates
        error_dict, error_update = self.update_error_estimates(params, error_matrix_dict)

        # Compute total loss
        lagrangian = graph_loss + dual_loss + barrier_loss
        
        loss = lagrangian
        if self.normalize_graph_loss:
            loss = loss / self.coefficient_vector.sum()

        # Generate dictionary with losses for logging
        metrics_dict = {
            'train_loss': loss,
            'graph_loss': graph_loss,
            'dual_loss': dual_loss,
            'barrier_loss': barrier_loss,
        }

        # Add additional metrics
        metrics_dict.update(graph_induced_norm_dict)
        metrics_dict.update(inner_dict)
        metrics_dict.update(dual_dict)
        metrics_dict.update(error_dict)
        metrics = (loss, graph_loss, dual_loss, barrier_loss, metrics_dict)
        aux = (metrics, error_update)

        return loss, aux

    def loss_function_non_permuted(
            self, params, train_batch, **kwargs
        ) -> Tuple[jnp.ndarray]:

        # Get representations
        start_representation, end_representation, \
            constraint_representation_1, constraint_representation_2 \
                = self.encode_states_non_permuted(params['encoder'], train_batch)
        
        # Compute primal loss
        graph_loss, graph_induced_norm_dict = self.compute_graph_drawing_loss(
            start_representation, end_representation
        )
        error_matrix_dict, inner_dict = self.compute_orthogonality_error_matrix(
            constraint_representation_1, constraint_representation_2,
        )

        # Compute dual loss
        dual_loss, barrier_loss, dual_dict = self.compute_orthogonality_loss(
           params, error_matrix_dict)
        
        # Update error estimates
        error_dict, error_update = self.update_error_estimates(params, error_matrix_dict)

        # Compute total loss
        lagrangian = graph_loss + dual_loss + barrier_loss
        
        loss = lagrangian
        if self.normalize_graph_loss:
            loss = loss / self.coefficient_vector.sum()

        # Generate dictionary with losses for logging
        metrics_dict = {
            'train_loss': loss,
            'graph_loss': graph_loss,
            'dual_loss': dual_loss,
            'barrier_loss': barrier_loss,
        }

        # Add additional metrics
        metrics_dict.update(graph_induced_norm_dict)
        metrics_dict.update(inner_dict)
        metrics_dict.update(dual_dict)
        metrics_dict.update(error_dict)
        metrics = (loss, graph_loss, dual_loss, barrier_loss, metrics_dict)
        aux = (metrics, error_update)

        return loss, aux
    
    def loss_function_permuted(
            self, params, train_batch, **kwargs
        ) -> Tuple[jnp.ndarray]:

        # Get representations
        start_representation, end_representation, \
            constraint_representation_1, constraint_representation_2 \
                = self.encode_states_permuted(params['encoder'], train_batch)
        
        # Compute primal loss
        graph_loss, graph_induced_norm_dict = self.compute_graph_drawing_loss(
            start_representation, end_representation
        )
        error_matrix_dict, inner_dict = self.compute_orthogonality_error_matrix(
            constraint_representation_1, constraint_representation_2,
        )

        # Compute dual loss
        dual_loss, barrier_loss, dual_dict = self.compute_orthogonality_loss(
           params, error_matrix_dict)
        
        # Update error estimates
        error_dict, error_update = self.update_error_estimates(params, error_matrix_dict)

        # Compute total loss
        lagrangian = graph_loss + dual_loss + barrier_loss
        
        loss = lagrangian
        if self.normalize_graph_loss:
            loss = loss / self.coefficient_vector.sum()

        # Generate dictionary with losses for logging
        metrics_dict = {
            'train_loss': loss,
            'graph_loss': graph_loss,
            'dual_loss': dual_loss,
            'barrier_loss': barrier_loss,
        }

        # Add additional metrics
        metrics_dict.update(graph_induced_norm_dict)
        metrics_dict.update(inner_dict)
        metrics_dict.update(dual_dict)
        metrics_dict.update(error_dict)
        metrics = (loss, graph_loss, dual_loss, barrier_loss, metrics_dict)
        aux = (metrics, error_update)

        return loss, aux
    
    def additional_update_step(self, step, params, *args, **kwargs):
        # Update the dual parameters
        
        params = self.update_duals(params)
        params = self.update_barrier_coefficients(params, *args, **kwargs)
        return params
    
    def update_duals(self, params):
        '''
            Update dual variables using some approximation 
            of the gradient of the lagrangian.
        '''
        error_matrix = params['errors']
        dual_variables = params['duals']
        updates = jnp.tril(error_matrix)
        dual_velocities = params['dual_velocities']

        # Calculate updated duals
        updated_duals = dual_variables + self.lr_duals * updates

        # Clip duals to be in the range [min_duals, max_duals]
        updated_duals = jnp.clip(
            updated_duals,
            a_min=self.min_duals,
            a_max=self.max_duals,
        )   # TODO: Cliping is probably not the best way to handle this

        # Update params, making sure that the duals are lower triangular
        params['duals'] = jnp.tril(updated_duals)
        
        # Update dual velocity
        updates = updated_duals - dual_variables
        updated_dual_velocities = dual_velocities + self.lr_dual_velocities * (updates - dual_velocities)
        params['dual_velocities'] = updated_dual_velocities
        
        return params
    
    def update_training_state(self, params, error_update):
        '''Update error estimates'''

        params['errors'] = error_update['errors']
        params['quadratic_errors'] = error_update['quadratic_errors']
        return params    
   
    @abstractmethod
    def update_barrier_coefficients(self, params, *args, **kwargs):   # TODO: eliminate this function when the best version is found
        raise NotImplementedError
    
    @abstractmethod
    def init_additional_params(self, *args, **kwargs):
        raise NotImplementedError
