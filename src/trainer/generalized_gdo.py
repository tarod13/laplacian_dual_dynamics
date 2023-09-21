from typing import Tuple
from itertools import product
import numpy as np
import jax
import jax.numpy as jnp

import equinox as eqx

from src.trainer.laplacian_encoder import LaplacianEncoderTrainer


class GeneralizedGraphDrawingObjectiveTrainer(LaplacianEncoderTrainer):
    def compute_graph_drawing_loss(self, start_representation, end_representation):
        # Get vector of mononotically decreasing coefficients
        coeff_vector = jnp.arange(self.d, 0, -1)

        # Compute reprensetation distance between start and end states weighted by coeff_vector
        loss = ((start_representation - end_representation)**2).dot(coeff_vector).mean()

        # Normalize loss
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
        for dim in range(representation_dim, 0, -1):
            norm_rep_1 = jnp.sqrt(jnp.dot(representation_1[:dim], representation_1[:dim]))
            norm_rep_2 = jnp.sqrt(jnp.dot(representation_2[:dim], representation_2[:dim]))
            dot_product = jnp.dot(representation_1[:dim], representation_2[:dim])
            loss += (
                dot_product ** 2 
                - (norm_rep_1 ** 2 / divisor)   # Why divide by rep_dim?
                - (norm_rep_2 ** 2 / divisor)
            )

        # Normalize loss
        if self.coefficient_normalization:
            loss = loss / (self.d * (self.d + 1) / 2)
                
        return loss

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
        orthogonality_loss_vec = compute_orthogonality_loss_vmap(
            constraint_start_representation, constraint_end_representation,
        )
        orthogonality_loss = orthogonality_loss_vec.mean()
        regularization_loss = self.barrier_initial_val * orthogonality_loss

        # Compute total loss
        loss = graph_loss + regularization_loss

        metrics_dict = {
            'train_loss': loss,
            'graph_loss': graph_loss,
            'reg_loss': orthogonality_loss,
            'barrier_loss': 0.0,
        }
        metrics = (loss, graph_loss, regularization_loss, metrics_dict)
        aux = (metrics, None)

        return loss, aux
    
    def update_training_state(self, params, *args, **kwargs):
        '''Leave params unchanged'''

        return params
    
    def additional_update_step(self, step, params, *args, **kwargs):
        '''Leave params unchanged'''

        return params
    
    def init_additional_params(self, *args, **kwargs):        
        additional_params = {}
        return additional_params
