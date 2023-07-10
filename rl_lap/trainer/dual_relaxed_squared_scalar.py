from typing import Tuple
from itertools import product
import numpy as np
import jax
import jax.numpy as jnp

import haiku as hk

from rl_lap.trainer.laplacian_encoder import LaplacianEncoderTrainer


class DualRelaxedSquaredScalarLaplacianEncoderTrainer(LaplacianEncoderTrainer):
    def compute_graph_drawing_loss(self, start_representation, end_representation):
        '''Compute reprensetation distances between start and end states'''
        
        loss = ((start_representation - end_representation)**2).sum(1).mean()
        return loss
    
    def compute_inner_product(self, feature_1, feature_2):
        '''Compute the inner product between two representations'''
        n = feature_1.size
        feature_2_stop = jax.lax.stop_gradient(feature_2)
        inner_product = feature_1.dot(feature_2_stop) / n
        return inner_product
    
    def compute_orthogonality_error_matrix(self, represetantation_batch_1, represetantation_batch_2):
        compute_inner_product_vec_vmap = jax.vmap(
            self.compute_inner_product, 
            in_axes=(1,None),
            out_axes=0,
        )
        compute_inner_product_matrix_vmap = jax.vmap(
            compute_inner_product_vec_vmap,
            in_axes=(None,1),
            out_axes=1,
        )
        inner_product_matrix_1 = compute_inner_product_matrix_vmap(
            represetantation_batch_1, represetantation_batch_1)   # Is there a better way? Half of the matrix will be filled with zeros
        inner_product_matrix_2 = compute_inner_product_matrix_vmap(
            represetantation_batch_2, represetantation_batch_2)
        
        # n = represetantation_batch_1.shape[0]

        # inner_product_matrix_1 = jnp.einsum(
        #     'ij,ik->jk',
        #     represetantation_batch_1,
        #     jax.lax.stop_gradient(represetantation_batch_1),
        # ) / n

        # inner_product_matrix_2 = jnp.einsum(
        #     'ij,ik->jk',
        #     represetantation_batch_2,
        #     jax.lax.stop_gradient(represetantation_batch_2),
        # ) / n

        error_matrix_1 = inner_product_matrix_1 - jnp.eye(self.d)
        error_matrix_2 = inner_product_matrix_2 - jnp.eye(self.d)

        orthogonality_error_matrix = error_matrix_1 * error_matrix_2

        inner_dict = {
            f'inner({i},{j})': inner_product_matrix_1[i,j]
            for i, j in product(range(self.d), range(self.d))
            if i >= j
        }

        return orthogonality_error_matrix, inner_dict
    
    # def update_error_estimates(self, params, errors):   # TODO: Handle better the fact that params are an array
    #     old = self.model_funcs['get_errors'].apply(params)
    #     update = self.error_estimate_update_rate * (errors - old)
    #     update_explicit = update[
    #         self.model.error_estimates.lower_triangular_indices[0],
    #         self.model.error_estimates.lower_triangular_indices[1],
    #     ]
    #     self.model.error_estimates.explicit_parameters.data.copy_(old_explicit + update_explicit)
    #     return update
    
    # def update_integral(self):
    #     # Update the integral
    #     self.model.integral.explicit_parameters.data.copy_(
    #         self.model.integral.explicit_parameters.data * self.integral_discount 
    #         + self.model.error_estimates.explicit_parameters.data
    #     )

    def compute_dual_loss(self, params, orthogonality_error_matrix):
        # Compute the loss
        log_dual_variables = self.model_funcs['get_duals'].apply(params)
        dual_variables = jnp.tril(jnp.exp(log_dual_variables))

        error_matrix = jnp.tril(orthogonality_error_matrix - self.orthogonality_tolerance)
        real_dual_loss = (jax.lax.stop_gradient(dual_variables) * error_matrix).sum()

        if self.optimize_dual_logs:
            dual_function = log_dual_variables
        else:
            dual_function = dual_variables

        virtual_dual_loss = (
            -(dual_function * jax.lax.stop_gradient(error_matrix)).sum()
            * self.learning_ratio
        )

        # Generate dictionary with dual variables and errors for logging 
        dual_dict = {
            f'beta({i},{j})': dual_variables[i,j]
            for i, j in product(range(self.d), range(self.d))
            if i >= j
        }
        
        return real_dual_loss, virtual_dual_loss, dual_dict

    def loss_function(
            self, params, train_batch, **kwargs
        ) -> Tuple[jnp.ndarray]:

        # Get representations
        start_representation, end_representation, \
            constraint_representation_1, constraint_representation_2 \
                = self.encode_states(params, train_batch)
        
        # Compute primal loss
        graph_loss = self.compute_graph_drawing_loss(
            start_representation, end_representation
        )
        orthogonality_error_matrix, inner_dict = self.compute_orthogonality_error_matrix(
            constraint_representation_1, constraint_representation_2,
        )

        # Compute dual loss
        real_dual_loss, virtual_dual_loss, dual_dict = self.compute_dual_loss(   # TODO: add dual_dict as output
           params, orthogonality_error_matrix)

        # Compute total loss
        lagrangian = graph_loss + real_dual_loss
        loss = lagrangian + virtual_dual_loss
        metrics_dict = {
            'train_loss': lagrangian,
            'graph_loss': graph_loss,
            'reg_loss': real_dual_loss,
        }
        metrics_dict.update(inner_dict)
        metrics_dict.update(dual_dict)
        metrics = (loss, graph_loss, real_dual_loss, metrics_dict)

        return loss, metrics