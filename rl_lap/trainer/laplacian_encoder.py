import os
import logging
from typing import Tuple
from abc import ABC, abstractmethod
from itertools import product
from collections import OrderedDict
from tqdm import tqdm

import torch
from torch.optim import Optimizer

from rl_lap.trainer.trainer import Trainer

# Libraries to generate episodes
import random
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

import rl_lap.env
from rl_lap.env.wrapper.norm_obs import NormObs
from rl_lap.agent.agent import BehaviorAgent as Agent
from rl_lap.policy import DiscreteUniformRandomPolicy as Policy

# Martin libraries
from ..tools import timer_tools


class LaplacianEncoderTrainer(Trainer, ABC):    # TODO: Handle device
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_counters()
        self.build_environment()
        self.collect_experience()
        self.train_info = OrderedDict()

    def train_step(self, batch, batch_global_idx) -> None:
        # Compute representations
        representations = self.encode_states(*batch)
        
        # Compute loss and associated intermediate metrics
        loss, metrics_dict = self.loss_function(representations)
        
        # Backpropagate loss
        # for param in self.model.parameters():
        #     param.grad = None
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log losses
        is_log_step = self.log_counter  == 0
        if is_log_step:
            # Compute additional metrics 
            additional_metrics_dict = self.metrics()
            metrics_dict.update(additional_metrics_dict)
            metrics_dict['grad_steps'] = batch_global_idx
            metrics_dict['examples'] = batch_global_idx * self.batch_size
            # TODO: Add own wall clock time

            if self.use_wandb:   # TODO: Use an alternative to wandb
                self.logger.log(metrics_dict)

        # Update target network
        self.update_counters()
        self.update_target()

        return loss, metrics_dict

    def train(   # TODO: Simplify this function (break it into smaller functions)
            self, 
            num_epochs: int, 
            profiler=None,
        ) -> None:
        
        # Create saving directory
        saver_dir = self.log_dir
        if not os.path.exists(saver_dir):
            os.makedirs(saver_dir)

		

        # Train model
        for epoch in tqdm(range(num_epochs)):

            # Train for one epoch
            for batch_idx, batch in enumerate(train_loader):
                batch_global_idx = epoch * n_batches_per_epoch + batch_idx
                loss, metrics_dict_ = self.train_step(batch, batch_global_idx)
                
                # Update profiler
                if profiler is not None:
                    profiler.step()

            # Log epoch
            if self.use_wandb:
                self.logger.log({'epochs': epoch})

        # Return final loss and metrics
        if loss in locals() and metrics_dict_ in locals():
            return loss, metrics_dict_
        else:
            return None, None

    def reset_counters(self) -> None:   
        self.step_counter = 0
        self.log_counter = 0

    def update_counters(self) -> None:
        self.step_counter += 1
        self.log_counter = (self.log_counter + 1) % self.log_every_n_steps
        
    def build_environment(self):
        # Create environment
        path_txt_grid = f'./rl_lap/env/grid/txts/{self.env_name}.txt'
        env = gym.make(
            self.env_family, 
            path=path_txt_grid, 
            render_mode="rgb_array", 
            use_target=False, 
        )
        # Wrap environment with observation normalization
        obs_wrapper = lambda e: NormObs(e)
        env = obs_wrapper(env)
        # Wrap environment with time limit
        time_wrapper = lambda e: TimeLimit(e, max_episode_steps=self.max_episode_steps)
        env = time_wrapper(env)

        # Set seed
        env.reset(seed=self.seed)

        # Set environment as attribute
        self.env = env

    def collect_experience(self) -> None:
        # Create agent
        policy = Policy(
            num_actions=self.env.action_space.n, 
            seed=self.seed
        )
        agent = Agent(policy)

        # Collect trajectories from random actions
        print('Start collecting samples.')   # TODO: Use logging
        timer = timer_tools.Timer()
        total_n_steps = 0
        collect_batch = 10_000   # TODO: Check if necessary
        while total_n_steps < self.n_samples:
            n_steps = min(collect_batch, 
                    self.n_samples - total_n_steps)
            steps = agent.collect_experience(self.env, n_steps)
            self.replay_buffer.add_steps(steps)
            total_n_steps += n_steps
            print(f'({total_n_steps}/{self.n_samples}) steps collected.')
        time_cost = timer.time_cost()
        print(f'Data collection finished, time cost: {time_cost}s')
    
    def encode_states(
            self, 
            start_states: torch.Tensor,
            end_states: torch.Tensor,
            start_states_constraints: torch.Tensor,
            end_states_constraints: torch.Tensor,
            *args, **kwargs,
        ) -> Tuple[torch.Tensor]:
        # Compute start representations
        start_representation = self.model(
            start_states, use_target=False)
        start_representation_constraints = self.model(
            start_states_constraints, use_target=False)

        # Compute end representations
        use_target_network = (
            hasattr(self, 'use_target_network') and 
            self.use_target_network
        )
        end_representation = self.model(end_states, use_target=use_target_network)
        if self.calculate_end_constraint_representation:
            end_representation_constraints = self.model(
                end_states_constraints, use_target=use_target_network)
        else:
            end_representation_constraints = None

        return (
            start_representation, end_representation, 
            start_representation_constraints, 
            end_representation_constraints
        )
    
    def update_target(self) -> None:
        has_target = hasattr(self.model, 'target') and self.model.target is not None
        if has_target:
            if hasattr(self, 'soft_target_update') and self.soft_target_update:
                self.model._update_target(
                    soft=True, target_update_rate=self.target_update_rate)
            else:
                if (self.step_counter % self.steps_to_update_target) == 0:
                    self.model._update_target(soft=False, target_update_rate=1.0)

    def compute_ground_truth_cosine_similarity(self):
        eigenvectors = self.model(
            self.states.type(self.dtype).to(self.device))
        eigenvectors_normalized = eigenvectors / eigenvectors.norm(dim=0, keepdim=True)
        ground_truth_eigenvectors = (
            self.ground_truth_eigenvectors 
            / self.ground_truth_eigenvectors.norm(dim=0, keepdim=True)
        )
        similarity_first_direction = (
            (eigenvectors_normalized * ground_truth_eigenvectors).sum(dim=0) 
        )
        similarity_second_direction = (
            (- eigenvectors_normalized * ground_truth_eigenvectors).sum(dim=0)
        )
        similarities = torch.maximum(
            similarity_first_direction, similarity_second_direction)
        cosine_similarity = similarities.mean()     # Notice that you are assuming a uniform distribution over the grid.
                                                    # This might be "unfair" with the methods that use a different distribution (maybe all?).
        return cosine_similarity.item(), eigenvectors
    
    def compute_orthogonality(self, eigenvectors=None):   # TODO: Check normalization (does it make sense to calculate here?)
        # Compute eigenvectors if not provided
        if eigenvectors is None:
            eigenvectors = self.model(
                self.states.type(self.dtype).to(self.device))
        
        # Compute inner products between eigenvectors
        n = eigenvectors.shape[0]
        inner_products = torch.absolute(
            torch.einsum('ij,ik->jk', eigenvectors, eigenvectors) / n)   # Notice that you are assuming a uniform distribution over the grid.

        # Create orthogonality dictionaries
        d = eigenvectors.shape[1]
        norm_dict = {
            f'norm({i})': inner_products[i,i].item()
            for i in range(d)
        }
        inner_dict = {   # TODO: Move dictionary generation to a separate function (?)
            f'inner({i},{j})': inner_products[i,j].item()
            for i, j in product(range(d), range(d))
            if i > j
        }
        return norm_dict, inner_dict
    
    def metrics(self):
        # Compute metrics
        with torch.no_grad():
            cosine_similarity, eigenvectors = self.compute_ground_truth_cosine_similarity()
            norm_dict, inner_dict = self.compute_orthogonality(eigenvectors)
        
        # Create metrics dictionary
        metrics_dict = {'cosine_similarity': cosine_similarity}
        metrics_dict.update(norm_dict)
        metrics_dict.update(inner_dict)

        return metrics_dict
    
    @abstractmethod
    def loss_function(self, *args, **kwargs):
        raise NotImplementedError
    
