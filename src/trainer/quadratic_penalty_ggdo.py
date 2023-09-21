from typing import Tuple
from itertools import product
import numpy as np
import jax
import jax.numpy as jnp

import equinox as eqx

from src.trainer.laplacian_encoder import LaplacianEncoderTrainer


class QuadraticPenaltyGGDOTrainer(LaplacianEncoderTrainer):
    def compute_graph_drawing_loss(self, start_representation, end_representation):
        # Get vector of mononotically decreasing coefficients
        coeff_vector = jnp.arange(self.d, 0, -1)

        # Compute reprensetation distance between start and end states weighted by coeff_vector
        loss = ((start_representation - end_representation)**2).dot(coeff_vector).mean()

        if self.coefficient_normalization:
            loss = loss / (self.d * (self.d + 1) / 2)

        return loss
    
    def compute_orthogonality_loss(self, representation_1, representation_2):
        representation_dim = representation_1.size
        if self.asymmetric_normalization:
            divisor = representation_dim
        else:
            divisor = 1
        loss = 0
        quadratic_error = 0
        for dim in range(representation_dim, 0, -1):
            norm_rep_1 = jnp.sqrt(jnp.dot(representation_1[:dim], representation_1[:dim]))
            norm_rep_2 = jnp.sqrt(jnp.dot(representation_2[:dim], representation_2[:dim]))
            dot_product = jnp.dot(representation_1[:dim], representation_2[:dim])
            loss += (
                dot_product ** 2 
                - (norm_rep_1 ** 2 / divisor)   # Why divide by rep_dim?
                - (norm_rep_2 ** 2 / divisor)
            )
            quadratic_error += (
                dot_product ** 2
                - norm_rep_1 ** 2
                - norm_rep_2 ** 2
                + dim
            )

        # Normalize loss
        coefficient_weight = self.d * (self.d + 1) / 2
        if self.coefficient_normalization:
            loss = loss / coefficient_weight
        quadratic_error = quadratic_error / coefficient_weight
                
        return loss, quadratic_error
        
    def update_error_estimates(self, params, quadratic_error) -> Tuple[dict]:   # TODO: Handle better the fact that params are an array
        # Get old error estimates
        old = params['squared_errors']   # TODO: change name
        
        # Compute update rate        
        update_rate = self.sq_error_estimate_update_rate   # TODO: Remove bias in the first iteration

        # Update error estimates
        update = old + update_rate * (quadratic_error - old)   # The first update might be too large
        updates = {'squared_errors': update}
            
        return updates

    def loss_function(
            self, params, train_batch, **kwargs
        ) -> Tuple[jnp.ndarray]:

        # Get representations
        start_representation, end_representation, \
            constraint_start_representation, constraint_end_representation \
                = self.encode_states(params['encoder'], train_batch)
        
        # Compute graph loss and regularization
        graph_loss = self.compute_graph_drawing_loss(
            start_representation, end_representation
        )

        compute_orthogonality_loss_vmap = jax.vmap(self.compute_orthogonality_loss)
        orthogonality_loss_vec, quadratic_error_vec = compute_orthogonality_loss_vmap(
            constraint_start_representation, constraint_end_representation,
        )
        orthogonality_loss = orthogonality_loss_vec.mean()
        barrier_coefficient = jax.lax.stop_gradient(params['barrier_coefs'][0,0])
        regularization_loss = barrier_coefficient * orthogonality_loss

        # Update error estimates
        error_update = self.update_error_estimates(params, quadratic_error_vec.mean())

        # Compute total loss
        loss = graph_loss + regularization_loss

        metrics_dict = {
            'train_loss': loss,
            'graph_loss': graph_loss,
            'linear_constraint_loss': 0.0,
            'quadratic_constraint_loss': orthogonality_loss,
            'barrier_coefficient': barrier_coefficient,
        }
        metrics = (loss, graph_loss, regularization_loss, metrics_dict)
        aux = (metrics, error_update)

        return loss, aux
    
    def update_barrier_coefficient(self, params, *args, **kwargs):
        '''
            Update barrier coefficient using some approximation 
            of the barrier gradient in the modified lagrangian.
        '''
        barrier_coefficient = params['barrier_coefs']
        squared_error_matrix = params['squared_errors']
        updates = jnp.tril(squared_error_matrix)
        updates = jnp.clip(updates, a_min=0, a_max=None).mean()

        # Calculate updated coefficients
        updated_barrier_coefficient = barrier_coefficient + self.lr_barrier_coefs * updates

        # Clip coefficients to be in the range [min_barrier_coefs, max_barrier_coefs]
        updated_barrier_coefficient = jnp.clip(
            updated_barrier_coefficient,
            a_min=self.min_barrier_coefs,
            a_max=self.max_barrier_coefs,
        )

        # Update params, making sure that the coefficients are lower triangular
        params['barrier_coefs'] = updated_barrier_coefficient
        return params
    
    def update_training_state(self, params, error_update): # TODO: 
        '''Update error estimates'''

        params['squared_errors'] = error_update['squared_errors']
        return params
    
    def additional_update_step(self, step, params, *args, **kwargs):
        # Update the dual parameters
        is_barrier_update_step = (
            ((step + 1) % self.update_barrier_every) == 0
        )
        if is_barrier_update_step:
            params = self.update_barrier_coefficient(params, *args, **kwargs)
        
        return params
    
    def init_additional_params(self, *args, **kwargs):
        barrier_initial_val = self.barrier_initial_val
        additional_params = {
            'barrier_coefs': jnp.tril(barrier_initial_val * jnp.ones((1, 1)), k=0),
            'squared_errors': jnp.zeros((1, 1)),
        }
        return additional_params
