from typing import Tuple
from itertools import product
import numpy as np
import torch

from rl_lap.trainer.laplacian_encoder import LaplacianEncoderTrainer


class CoefficientAugmentedLaplacianEncoderTrainerM(LaplacianEncoderTrainer):
    def compute_graph_drawing_loss(self, start_representation, end_representation):
        coeff_vector = torch.arange(
            start_representation.shape[1], 0, -1, 
            dtype=self.dtype, device=self.device
        )
        diff_matrix = (start_representation - end_representation)**2
        loss = torch.einsum('ij,j->i', diff_matrix, coeff_vector).mean()
        return loss
    
    def compute_orthogonality_loss(self, start_representation, end_representation):
        rep_dim = start_representation.shape[1]
        loss = 0
        for dim in range(rep_dim, 0, -1):
            x_norm = torch.sqrt(((start_representation[:,:dim])**2).sum(dim=1, keepdims=True))
            y_norm = torch.sqrt(((end_representation[:,:dim])**2).sum(dim=1, keepdims=True))
            dot_product = (start_representation[:,:dim] * end_representation[:,:dim]).sum(dim=1, keepdims=True)
            loss += (
                dot_product ** 2 - x_norm ** 2 / rep_dim  - y_norm ** 2 / rep_dim  ).mean()   # Why divide by rep_dim?
                
        return loss

    def loss_function(
            self, representation_batch, **kwargs
        ) -> Tuple[torch.Tensor]:

        # Unpack batch
        start_representation, end_representation = representation_batch[:2]
        start_representation_uncorrelated, end_representation_uncorrelated = \
            representation_batch[2:4]
        
        # Compute graph loss and regularization
        graph_loss = self._compute_graph_drawing_loss(
            start_representation, end_representation
        )
        orthogonality_loss = self._compute_orthogonality_loss(
            start_representation_uncorrelated, end_representation_uncorrelated,
        )
        regularization_loss = self.regularization_weight * orthogonality_loss

        # Compute total loss
        loss = graph_loss + regularization_loss

        # Store metrics
        metrics_dict = {
            'train_loss': loss.detach().item(),
            'graph_loss': graph_loss.detach().item(),
            'reg_loss': regularization_loss.detach().item(),
        }

        return loss, metrics_dict