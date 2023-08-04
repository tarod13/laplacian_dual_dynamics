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
       
        # Compute the gradients and associated intermediate metrics
        grads, aux = jax.grad(self.loss_function, has_aux=True)(params, train_batch)

        # Calculate grad norms
        grad_norms = jax.tree_map(jnp.linalg.norm, grads)

        # Add grad norms to the training info
        for k, v in grad_norms.items():
            aux[0][-1][k + '_grad_norm'] = v
        
        # Determine the real parameter updates
        updates, opt_state = self.optimizer.update(grads, opt_state)

        # Update the encoder parameters
        params = optax.apply_updates(params, updates)

        # Update the training state
        params = self.update_training_state(params, aux[1])

        return params, opt_state, aux[0]

    def train(self) -> None:

        timer = timer_tools.Timer()

        # Initialize the parameters
        rng = hk.PRNGSequence(self.rng_key)
        sample_input = self._get_train_batch()
        encoder_params = self.encoder_fn.init(next(rng), sample_input.s1)
        params = {
            'encoder': encoder_params,
            'duals': self.dual_params,
        }
        # Add state info to the params dictionary
        params.update(self.training_state)
        
        # Initialize the optimizer
        opt_state = self.optimizer.init(params)   # TODO: Should encoder_params be the only ones updated by the optimizer?

        # Learning begins   # TODO: Better comments
        timer.set_step(0)
        for step in range(self.total_train_steps):

            train_batch = self._get_train_batch()
            params, opt_state, metrics = self.train_step(params, train_batch, opt_state)
            
            self._global_step += 1   # TODO: Replace with self.step_counter

            # Update the dual parameters
            is_dual_update_step = (
                (((step + 1) % self.update_dual_every) == 0)
                and (step > self.update_dual_after)
            )
            if is_dual_update_step:
                params = self.update_duals(params)

            is_dual_reset_step = (
                (((step + 1) % self.reset_dual_every) == 0)
                and (step > self.update_dual_after)
            )
            if is_dual_reset_step:
                params = self.reset_duals(params)

            # Save and print info
            is_log_step = ((step + 1) % self.print_freq) == 0
            if is_log_step:   # TODO: Replace with self.log_counter

                losses = metrics[:-1]
                metrics_dict = metrics[-1]
                if self.use_cosine_similarity_v2:
                    cosine_function = self.compute_cosine_similarity_v2
                else:
                    cosine_function = self.compute_cosine_similarity

                cosine_similarity, similarities = cosine_function(params['encoder'])
                metrics_dict['cosine_similarity'] = cosine_similarity
                for feature in range(len(similarities)):
                    metrics_dict[f'cosine_similarity_{feature}'] = similarities[feature]
                metrics_dict['grad_step'] = self._global_step
                metrics_dict['examples'] = self._global_step * self.batch_size
                metrics_dict['wall_clock_time'] = timer.time_cost()
                
                self.train_info['loss_total'] = np.array([jax.device_get(losses[0])])[0]
                self.train_info['loss_pos'] = np.array([jax.device_get(losses[1])])[0]
                self.train_info['loss_neg'] = np.array([jax.device_get(losses[2])])[0]
                self.train_info['cos_sim'] = np.array([jax.device_get(cosine_similarity)])[0]

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

        # Log environment eigenvalues
        eigenvalues = self.env.get_eigenvalues()[:self.d]
        eigval_dict = {
            f'eigval_{i}': eigenvalues[i] for i in range(len(eigenvalues))
        }
        self.logger.log(eigval_dict)
        print(f'Environment eigenvalues: {eigenvalues}')

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
            params_encoder,
            train_batch: Data,
            *args, **kwargs,
        ) -> Tuple[jnp.ndarray]:

        # Compute start representations
        start_representation = self.encoder_fn.apply(params_encoder, train_batch.s1)
        constraint_start_representation = self.encoder_fn.apply(params_encoder, train_batch.s_neg_1)

        # Compute end representations
        end_representation = self.encoder_fn.apply(params_encoder, train_batch.s2)
        constraint_end_representation = self.encoder_fn.apply(params_encoder, train_batch.s_neg_2)

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

    def compute_cosine_similarity(self, params_encoder):
        # Get baseline parameters
        states = self.env.get_states()
        real_eigvec = self.env.get_eigenvectors()[:,:self.d]
        real_norms = jnp.linalg.norm(real_eigvec, axis=0, keepdims=True)
        real_eigvec = real_eigvec / real_norms

        # Get approximated eigenvectors
        approx_eigvec = self.encoder_fn.apply(params_encoder, states)
        norms = jnp.linalg.norm(approx_eigvec, axis=0, keepdims=True)
        approx_eigvec = approx_eigvec / norms
        
        # Compute cosine similarities for both directions
        sim_first_dir = (approx_eigvec * real_eigvec).sum(axis=0)
        sim_second_dir = (- approx_eigvec * real_eigvec).sum(axis=0)

        # Take the maximum similarity for each eigenvector
        similarities = jnp.maximum(sim_first_dir, sim_second_dir)
        cosine_similarity = similarities.mean()
        return cosine_similarity, similarities
    
    def compute_cosine_similarity_v2(self, params_encoder):
        # Get baseline parameters
        states = self.env.get_states()
        real_eigval = self.env.get_eigenvalues()[:self.d]
        real_eigvec = self.env.get_eigenvectors()[:,:self.d]
        real_norms = jnp.linalg.norm(real_eigvec, axis=0, keepdims=True)
        real_eigvec = real_eigvec / real_norms

        # Store eigenvectors in a dictionary corresponding to each eigenvalue
        eigvec_dict = {}
        for i, eigval in enumerate(real_eigval):
            if eigval not in eigvec_dict:
                eigvec_dict[eigval] = []
            eigvec_dict[eigval].append(real_eigvec[:,i])

        # Get approximated eigenvectors
        approx_eigvec = self.encoder_fn.apply(params_encoder, states)
        norms = jnp.linalg.norm(approx_eigvec, axis=0, keepdims=True)
        approx_eigvec = approx_eigvec / norms
        
        # Compute cosine similarities for both directions
        unique_real_eigval = sorted(eigvec_dict.keys(), reverse=True)
        print(f'Unique eigenvalues: {unique_real_eigval}')
        id_ = 0
        similarities = []
        for i, eigval in enumerate(unique_real_eigval):
            multiplicity = len(eigvec_dict[eigval])
            print(f'Eigenvalue {eigval} has multiplicity {multiplicity}')
            
            # Compute cosine similarity
            if multiplicity == 1:
                # Get eigenvectors associated with the current eigenvalue
                current_real_eigvec = eigvec_dict[eigval][0]
                current_approx_eigvec = approx_eigvec[:,id_]

                # Compute cosine similarity
                pos_sim = (current_real_eigvec).dot(current_approx_eigvec)
                similarities.append(jnp.maximum(pos_sim, -pos_sim))
            else:
                # Get eigenvectors associated with the current eigenvalue
                current_real_eigvec = eigvec_dict[eigval]
                current_approx_eigvec = approx_eigvec[:,id_:id_+multiplicity]
                optimal_approx_eigvec = self.rotate_eigenvectors(   # TODO: implement this function
                    current_real_eigvec, current_approx_eigvec)
                
                # Compute cosine similarity
                for j in range(multiplicity):
                    pos_sim = (current_real_eigvec[j]).dot(optimal_approx_eigvec[j])
                    similarities.append(jnp.maximum(pos_sim, -pos_sim))

            id_ += multiplicity

        # Convert to array
        similarities = jnp.array(similarities)
        # print(f'Similarities: {similarities}')

        # Compute average cosine similarity
        cosine_similarity = similarities.mean()

        return cosine_similarity, similarities
    
    def rotate_eigenvectors(
            self, 
            u_list: list, 
            E: jnp.ndarray
        ) -> list:
        '''
            Rotate the eigenvectors in E to match the 
            eigenvectors in u_list as close as possible.
            That is, we are finding the optimal basis of
            the subspace spanned by the eigenvectors in E
            such that the angle between the eigenvectors
            in u_list and the rotated eigenvectors is
            minimized.
        '''
        rotated_eigvec = []

        # Compute first eigenvector
        u1 = u_list[0]
        A = E.T.dot(E)
        b = 0.5*E.T.dot(u1)
        x = jnp.linalg.solve(A, b)
        u1_approx = E.dot(x)
        u1_approx = u1_approx / jnp.linalg.norm(u1_approx)
        rotated_eigvec.append(u1_approx)

        # Compute remaining eigenvectors
        for k in range(1, len(u_list)):
            uk = u_list[k]
            Uk = jnp.stack(rotated_eigvec, axis=1)
            bk = 0.5*E.T.dot(uk)
            Bk = 0.5*E.T.dot(Uk)
            xk = jnp.linalg.solve(A, bk)
            Xk = jnp.linalg.solve(A, Bk)
            Mk = Uk.T.dot(E)
            Ak = Mk.dot(Xk)
            bbk = Mk.dot(xk)
            xx_k = jnp.linalg.solve(Ak, bbk)
            uk_approx = E.dot(xx_k)
            uk_approx = uk_approx / jnp.linalg.norm(uk_approx)
            rotated_eigvec.append(uk_approx)

        return rotated_eigvec

    @abstractmethod
    def loss_function(self, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def update_duals(self, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def update_training_state(self, *args, **kwargs):
        raise NotImplementedError
