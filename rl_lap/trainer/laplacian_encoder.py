import os
import logging
from typing import Tuple
from abc import ABC, abstractmethod
from itertools import product
from collections import OrderedDict, namedtuple
#from tqdm import tqdm

#import torch
#from torch.optim import Optimizer

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
from ..tools import summary_tools
import jax
import haiku as hk
import jax.numpy as jnp
import numpy as np
import optax

# Equinox version libraries
import equinox as eqx

Data = namedtuple("Data", "s1 s2 s_neg_1 s_neg_2")   # TODO: Change notation


class LaplacianEncoderTrainer(Trainer, ABC):    # TODO: Handle device
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_counters()
        self.build_environment()
        self.collect_experience()
        self.train_step = jax.jit(self.train_step)   # TODO: _train_step
        self.compute_cosine_similarity = jax.jit(self.compute_cosine_similarity)
        self.train_info = OrderedDict()
        self._global_step = 0

    def _get_obs_batch(self, steps):   # TODO: Check this function (way to build the batch)
        obs_batch = [s.step.agent_state["agent"].astype(np.float32)
                for s in steps]
        return np.stack(obs_batch, axis=0)

    def _get_train_batch(self):
        s1, s2 = self.replay_buffer.sample_pairs(
                batch_size=self.batch_size,
                discount=self.discount,
                )
        s_neg_1 = self.replay_buffer.sample_steps(self.batch_size)
        s_neg_2 = self.replay_buffer.sample_steps(self.batch_size)
        s_pos_1, s_pos_2, s_neg_1, s_neg_2 = map(self._get_obs_batch, [s1, s2, s_neg_1, s_neg_2])
        batch = Data(s_pos_1, s_pos_2, s_neg_1, s_neg_2)
        return batch

    def train_step(self, params, train_batch, opt_state) -> None:   # TODO: Check if batch_global_idx can be passed as a parameter with jax
        if self.nn_library in ['haiku', 'haiku-v2', 'flax']:
            # Compute the gradients and associated intermediate metrics
            grads, aux = jax.grad(self.loss_function, has_aux=True)(params, train_batch)
            
            # Determine the real parameter updates
            updates, opt_state = self.optimizer.update(grads, opt_state)

            # Update the parameters
            params = optax.apply_updates(params, updates)

        elif self.nn_library == 'equinox':
            # Compute the gradients and associated intermediate metrics
            grads, aux = eqx.filter_grad(self.loss_function, has_aux=True)(
                params, train_batch)
            
            # Determine the real parameter updates
            updates, opt_state = self.optimizer.update(grads, opt_state)

            # Update the parameters
            params = eqx.apply_updates(params, updates)

        else:
            raise ValueError(f'Unknown neural network library: {self.nn_library}')

        # # Log losses
        # is_log_step = self.log_counter  == 0
        # if is_log_step:
        #     # Compute additional metrics 
        #     additional_metrics_dict = self.metrics()
        #     metrics_dict.update(additional_metrics_dict)
        #     metrics_dict['grad_steps'] = batch_global_idx
        #     metrics_dict['examples'] = batch_global_idx * self.batch_size
        #     # TODO: Add own wall clock time

        #     if self.use_wandb:   # TODO: Use an alternative to wandb
        #         self.logger.log(metrics_dict)

        # # Update target network
        # self.update_counters()
        # self.update_target()

        return params, opt_state, aux

    def train(self) -> None:

        timer = timer_tools.Timer()

        if self.nn_library in ['haiku', 'haiku-v2']:
            rng = hk.PRNGSequence(self.rng_key)
            sample_input = self._get_train_batch()
            params = self.model_funcs['forward'].init(next(rng), sample_input.s1)
            for param in params.keys():
                print(param)
        elif self.nn_library == 'equinox':
            params = self.model_funcs['forward']
        elif self.nn_library == 'flax':
            sample_input = self._get_train_batch()
            params = self.model_funcs['forward'].init(self.rng_key, sample_input.s1)
        else:
            raise ValueError(f'Unknown neural network library: {self.nn_library}')
        
        opt_state = self.optimizer.init(params)

        # learning begins   # TODO: Better comments
        timer.set_step(0)
        for step in range(self.total_train_steps):

            train_batch = self._get_train_batch()
            params, opt_state, metrics = self.train_step(params, train_batch, opt_state)
            losses = metrics[:-1]
            metrics_dict = metrics[-1]
            cosine_similarity = self.compute_cosine_similarity(params)
            metrics_dict['cosine_similarity'] = cosine_similarity
            metrics_dict['grad_step'] = self._global_step
            metrics_dict['examples'] = self._global_step * self.batch_size
            metrics_dict['wall_clock_time'] = timer.time_cost()

            self._global_step += 1   # TODO: Replace with self.step_counter
            self.train_info['loss_total'] = np.array([jax.device_get(losses[0])])[0]
            self.train_info['loss_pos'] = np.array([jax.device_get(losses[1])])[0]
            self.train_info['loss_neg'] = np.array([jax.device_get(losses[2])])[0]
            self.train_info['cos_sim'] = np.array([jax.device_get(cosine_similarity)])[0]

            # print info
            if step == 0 or (step + 1) % self.print_freq == 0:   # TODO: Replace with self.log_counter
                steps_per_sec = timer.steps_per_sec(step)
                print(f'Training steps per second: {steps_per_sec:.4g}.')   # TODO: Use logging instead of print
                self._print_train_info()
                if self.use_wandb:   # TODO: Use an alternative to wandb if False
                    self.logger.log(metrics_dict)
        time_cost = timer.time_cost()
        print(f'Training finished, time cost {time_cost:.4g}s.')

    def _print_train_info(self):   # TODO: Replace this function
        summary_str = summary_tools.get_summary_str(
                step=self._global_step, info=self.train_info)
        print(summary_str)   # TODO: Use logging instead of print

        # # Train model
        # for epoch in tqdm(range(num_epochs)):

        #     # Train for one epoch
        #     for batch_idx, batch in enumerate(train_loader):
        #         batch_global_idx = epoch * n_batches_per_epoch + batch_idx
        #         loss, metrics_dict_ = self.train_step(batch, batch_global_idx)
                
        #         # Update profiler
        #         if profiler is not None:
        #             profiler.step()

        #     # Log epoch
        #     if self.use_wandb:
        #         self.logger.log({'epochs': epoch})

        # # Return final loss and metrics
        # if loss in locals() and metrics_dict_ in locals():
        #     return loss, metrics_dict_
        # else:
        #     return None, None

    def reset_counters(self) -> None:   
        self.step_counter = 0
        self.log_counter = 0

    def update_counters(self) -> None:
        self.step_counter += 1
        self.log_counter = (self.log_counter + 1) % self.print_freq
        
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
        
        # Plot visitation counts
        min_visitation, max_visitation, visitation_entropy, max_entropy, visitation_freq = \
            self.replay_buffer.plot_visitation_counts(
                self.env.get_states(),
                self.env_name,
                self.env.grid.astype(bool),
        )
        time_cost = timer.time_cost()
        print(f'Visitation evaluated, time cost: {time_cost}s')
        print(f'Min visitation: {min_visitation}')
        print(f'Max visitation: {max_visitation}')
        print(f'Visitation entropy: {visitation_entropy}/{max_entropy}')
    
    def encode_states(
            self, 
            params,
            train_batch: Data,
            *args, **kwargs,
        ) -> Tuple[jnp.ndarray]:

        if self.nn_library in ['haiku', 'haiku-v2', 'flax']:
            # Compute start representations
            start_representation = self.model_funcs['forward'].apply(params, train_batch.s1)
            constraint_start_representation = self.model_funcs['forward'].apply(params, train_batch.s_neg_1)

            # Compute end representations
            end_representation = self.model_funcs['forward'].apply(params, train_batch.s2)
            constraint_end_representation = self.model_funcs['forward'].apply(params, train_batch.s_neg_2)

        elif self.nn_library == 'equinox':
            # Compute start representations
            start_representation = jax.vmap(params)(train_batch.s1)
            constraint_start_representation = jax.vmap(params)(train_batch.s_neg_1)

            # Compute end representations
            end_representation = jax.vmap(params)(train_batch.s2)
            constraint_end_representation = jax.vmap(params)(train_batch.s_neg_2)

        else:
            raise ValueError(f'Unknown neural network library: {self.nn_library}')

        return (
            start_representation, end_representation, 
            constraint_start_representation, 
            constraint_end_representation,
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

    def compute_cosine_similarity(self, params):
        # Get baseline parameters
        states = self.env.get_states()
        real_eigvec = self.env.get_eigenvectors()[:,:self.d]
        real_norms = jnp.linalg.norm(real_eigvec, axis=0, keepdims=True)
        real_eigvec = real_eigvec / real_norms

        # Get approximated eigenvectors
        if self.nn_library in ['haiku', 'haiku-v2', 'flax']:
            approx_eigvec = self.model_funcs['forward'].apply(params, states)
        elif self.nn_library == 'equinox':
            approx_eigvec = jax.vmap(params)(states)
        else:
            raise ValueError(f'Unknown neural network library: {self.nn_library}')

        norms = jnp.linalg.norm(approx_eigvec, axis=0, keepdims=True)
        approx_eigvec = approx_eigvec / norms
        
        # Compute cosine similarities for both directions
        sim_first_dir = (approx_eigvec * real_eigvec).sum(axis=0)
        sim_second_dir = (- approx_eigvec * real_eigvec).sum(axis=0)

        # Take the maximum similarity for each eigenvector
        similarities = jnp.maximum(sim_first_dir, sim_second_dir)
        cosine_similarity = similarities.mean()
        return cosine_similarity
    
    # def compute_orthogonality(self, eigenvectors=None):   # TODO: Check normalization (does it make sense to calculate here?)
    #     # Compute eigenvectors if not provided
    #     if eigenvectors is None:
    #         eigenvectors = self.model(
    #             self.states.type(self.dtype).to(self.device))
        
    #     # Compute inner products between eigenvectors
    #     n = eigenvectors.shape[0]
    #     inner_products = torch.absolute(
    #         torch.einsum('ij,ik->jk', eigenvectors, eigenvectors) / n)   # Notice that you are assuming a uniform distribution over the grid.

    #     # Create orthogonality dictionaries
    #     d = eigenvectors.shape[1]
    #     norm_dict = {
    #         f'norm({i})': inner_products[i,i].item()
    #         for i in range(d)
    #     }
    #     inner_dict = {   # TODO: Move dictionary generation to a separate function (?)
    #         f'inner({i},{j})': inner_products[i,j].item()
    #         for i, j in product(range(d), range(d))
    #         if i > j
    #     }
    #     return norm_dict, inner_dict
    
    # def metrics(self):
    #     # Compute metrics
    #     with torch.no_grad():
    #         cosine_similarity, eigenvectors = self.compute_ground_truth_cosine_similarity()
    #         norm_dict, inner_dict = self.compute_orthogonality(eigenvectors)
        
    #     # Create metrics dictionary
    #     metrics_dict = {'cosine_similarity': cosine_similarity}
    #     metrics_dict.update(norm_dict)
    #     metrics_dict.update(inner_dict)

    #     return metrics_dict
    
    @abstractmethod
    def loss_function(self, *args, **kwargs):
        raise NotImplementedError
    
