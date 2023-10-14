import os
from typing import Tuple
from abc import ABC, abstractmethod
from collections import OrderedDict, namedtuple
from datetime import datetime

from src.trainer.trainer import Trainer

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import wandb

import src.env
from src.env.wrapper.norm_obs import NormObs
from src.agent.agent import BehaviorAgent as Agent
from src.policy import DiscreteUniformRandomPolicy as Policy
from src.env.grid.utils import load_eig

from ..tools import timer_tools
from ..tools import summary_tools
from ..tools import saving
import jax
import haiku as hk
import jax.numpy as jnp
import numpy as np
import optax

import matplotlib.pyplot as plt
from scipy.interpolate import Rbf


Data = namedtuple("Data", "s1 s2 s_neg_1 s_neg_2")   # TODO: Change notation


class LaplacianEncoderTrainer(Trainer, ABC):    # TODO: Handle device
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_counters()
        self.build_environment()
        self.collect_experience()
        self.train_step = jax.jit(self.train_step)   # TODO: _train_step
        # self.compute_cosine_similarity = jax.jit(self.compute_cosine_similarity)
        self.train_info = OrderedDict()
        self._global_step = 0
        self._best_cosine_similarity = -1
        self._date_time = datetime.now().strftime("%Y%m%d%H%M%S")

    def _get_obs_batch(self, steps):   # TODO: Check this function (way to build the batch)
        if self.obs_mode in ["xy"]:
            obs_batch = [s.step.agent_state["xy_agent"].astype(np.float32)
                    for s in steps]
        elif self.obs_mode in ["pixels", "both"]:
            obs_batch = [s.step.agent_state["pixels"].astype(np.float32)/255 for s in steps]
        elif self.obs_mode in ["grid", "both-grid"]:
            obs_batch = [s.step.agent_state["grid"].astype(np.float32)/255 for s in steps]
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
        }
        # Add duals and state info to the params dictionary
        additional_params = self.init_additional_params()
        params.update(additional_params)
        
        # Initialize the optimizer
        opt_state = self.optimizer.init(params)   # TODO: Should encoder_params be the only ones updated by the optimizer?

        # Test the initial parameters
        sample_output = self.encoder_fn.apply(params['encoder'], sample_input.s1)
        avg_sample_output_norm = jnp.linalg.norm(sample_output, axis=1, keepdims=True).mean()
        if avg_sample_output_norm < 1e-6:
            raise Warning(f'Initial parameters have an average norm of {avg_sample_output_norm}. There might be a problem with the ConvNet layers')

        # Learning begins   # TODO: Better comments
        timer.set_step(0)
        for step in range(self.total_train_steps):

            train_batch = self._get_train_batch()
            params, opt_state, metrics = self.train_step(params, train_batch, opt_state)
            
            self._global_step += 1   # TODO: Replace with self.step_counter

            params = self.additional_update_step(step, params)

            # Save and print info
            is_log_step = ((step + 1) % self.print_freq) == 0
            if is_log_step:   # TODO: Replace with self.log_counter

                losses = metrics[:-1]
                metrics_dict = metrics[-1]
                
                cosine_similarity, similarities = self.compute_cosine_similarity(params['encoder'])
                maximal_cosine_similarity, maximal_similarities = self.compute_maximal_cosine_similarity(params['encoder'])
                
                metrics_dict['cosine_similarity'] = cosine_similarity
                metrics_dict['maximal_cosine_similarity'] = maximal_cosine_similarity
                for feature in range(len(similarities)):
                    metrics_dict[f'cosine_similarity_{feature}'] = similarities[feature]
                    metrics_dict[f'maximal_cosine_similarity_{feature}'] = maximal_similarities[feature]
                metrics_dict['grad_step'] = self._global_step
                metrics_dict['examples'] = self._global_step * self.batch_size
                metrics_dict['wall_clock_time'] = timer.time_cost()
                
                self.train_info['loss_total'] = np.array([jax.device_get(losses[0])])[0]
                self.train_info['graph_loss'] = np.array([jax.device_get(losses[1])])[0]
                self.train_info['dual_loss'] = np.array([jax.device_get(losses[2])])[0]
                self.train_info['barrier_loss'] = np.array([jax.device_get(losses[3])])[0]
                self.train_info['cos_sim'] = np.array([jax.device_get(cosine_similarity)])[0]
                self.train_info['max_cos_sim'] = np.array([jax.device_get(maximal_cosine_similarity)])[0]

                steps_per_sec = timer.steps_per_sec(step)
                print(f'Training steps per second: {steps_per_sec:.4g}.')   # TODO: Use logging instead of print

                self._print_train_info()
                if self.use_wandb:   # TODO: Use an alternative to wandb if False
                    # Log metrics
                    self.logger.log(metrics_dict)

            is_last_step = (step + 1) == self.total_train_steps
            is_plot_step = (
                self.do_plot_eigenvectors
                and is_last_step
            )
            if is_plot_step:
                self.plot_eigenvectors(params['encoder'])

            is_save_step = (
                self.save_model 
                and (
                    (((step + 1) % self.save_model_every) == 0)
                    or is_last_step
                )
            )
            if is_save_step:
                self._save_model(params, opt_state, cosine_similarity)
                    
        time_cost = timer.time_cost()
        print(f'Training finished, time cost {time_cost:.4g}s.')

    def _save_model(self, params, optim_state, cosine_similarity):
        # Save the model if the cosine similarity is better than the previous best
        if cosine_similarity > self._best_cosine_similarity:
            save_path_best = f'./results/models/{self.env_name}/best_{self._date_time}.pkl'

            self._best_cosine_similarity = cosine_similarity
            saving.save_model(
                params=params,
                optim_state=optim_state,
                path=save_path_best,
                overwrite=True,
            )

            # Log parameters
            if self.use_wandb:                    
                best_model = wandb.Artifact(
                    name='best_model', 
                    type='model',
                    description='Best model found during training.',
                )

                # Add model parameters to the artifact
                best_model.add_file(save_path_best)

                # Save the artifact
                self.logger.log_artifact(best_model)
            
        # Save the model every log step
        save_path_last = f'./results/models/{self.env_name}/last_{self._date_time}.pkl'        
        saving.save_model(
            params=params,
            optim_state=optim_state,
            path=save_path_last,
            overwrite=True,
        )

        # Log parameters
        if self.use_wandb:
            last_model = wandb.Artifact(
                name='last_model', 
                type='model',
                description='Most recent model.',
            )

            # Add model parameters to the artifact
            last_model.add_file(save_path_last)

            # Save the artifact
            self.logger.log_artifact(last_model)

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
        # Load eigenvectors and eigenvalues of the transition dynamics matrix (if they exist)
        path_eig = f'./src/env/grid/eigval/{self.env_name}.npz'
        eig, eig_not_found = load_eig(path_eig)

        # Create environment
        path_txt_grid = f'./src/env/grid/txts/{self.env_name}.txt'
        env = gym.make(
            self.env_family, 
            path=path_txt_grid, 
            render_mode="rgb_array", 
            use_target=False, 
            eig=eig,
            obs_mode=self.obs_mode,
        )
        # Wrap environment with observation normalization
        obs_wrapper = lambda e: NormObs(e)
        env = obs_wrapper(env)
        # Wrap environment with time limit
        time_wrapper = lambda e: TimeLimit(
            e, max_episode_steps=self.max_episode_steps)
        env = time_wrapper(env)

        # Set seed
        env.reset(seed=self.seed)

        # Set environment as attribute
        self.env = env

        # Save eigenvectors and eigenvalues
        if eig_not_found and self.save_eig:
            self.env.save_eigenpairs(path_eig)

        # Log environment eigenvalues
        self.env.round_eigenvalues(self.eigval_precision_order)
        eigenvalues = self.env.get_eigenvalues()
        print(f'Environment: {self.env_name}')
        print(f'Environment eigenvalues: {eigenvalues}')

        # Create eigenvector dictionary
        real_eigval = eigenvalues[:self.d]
        real_eigvec = self.env.get_eigenvectors()[:,:self.d]

        assert not np.isnan(real_eigvec).any(), \
            f'NaN values in the real eigenvectors: {real_eigvec}'

        jnp_real_eigvec = jnp.array(real_eigvec, dtype=jnp.float32)

        assert not jnp.isnan(jnp_real_eigvec).any(), \
            f'NaN values in the real eigenvectors: {real_eigvec}'

        jnp_real_norms = jnp.linalg.norm(jnp_real_eigvec, axis=0, keepdims=True)
        jnp_real_eigvec_norm = jnp_real_eigvec / jnp_real_norms

        # Check if any NaN values are present
        assert not jnp.isnan(jnp_real_eigvec_norm).any(), \
            f'NaN values in the real eigenvectors: {jnp_real_eigvec_norm}'

        # Store eigenvectors in a dictionary corresponding to each eigenvalue
        eigvec_dict = {}
        for i, eigval in enumerate(real_eigval):
            if eigval not in eigvec_dict:
                eigvec_dict[eigval] = []
            eigvec_dict[eigval].append(jnp_real_eigvec_norm[:,i])
        self.eigvec_dict = eigvec_dict
        
        # Print multiplicity of first eigenvalues
        multiplicities = [len(eigvec_dict[eigval]) for eigval in eigvec_dict.keys()]
        for i, eigval in enumerate(eigvec_dict.keys()):
            print(f'Eigenvalue {eigval} has multiplicity {multiplicities[i]}')

        if self.use_wandb:
            eigval_dict = {
                f'eigval_{i}': eigenvalues[i] for i in range(len(eigenvalues[:self.d]))
            }
            self.logger.log(eigval_dict)

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
                self.env.get_states()['xy_agent'],   # TODO: Make this more general (not only for xy or both)
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

    def get_states(self):
        state_dict = self.env.get_states()
        if self.obs_mode in ["pixels", "both"]:
            states = state_dict['pixels']
        elif self.obs_mode in ["grid", "both-grid"]:
            states = state_dict['grid']
        elif self.obs_mode in ["xy"]:
            states = state_dict['xy_agent']
        else:
            raise ValueError(f'Invalid observation mode: {self.obs_mode}')
        return states

    def compute_cosine_similarity(self, params_encoder):
        # Get states
        states = self.get_states()

        # Get approximated eigenvectors
        approx_eigvec = self.encoder_fn.apply(params_encoder, states)   # TODO: Do some minibatch processing here
        norms = jnp.linalg.norm(approx_eigvec, axis=0, keepdims=True)
        approx_eigvec = approx_eigvec / norms.clip(min=1e-10)
        
        # Compute cosine similarities for both directions
        unique_real_eigval = sorted(self.eigvec_dict.keys(), reverse=True)
        # print(f'Unique eigenvalues: {unique_real_eigval}')
        id_ = 0
        similarities = []
        for i, eigval in enumerate(unique_real_eigval):
            multiplicity = len(self.eigvec_dict[eigval])
            # print(f'Eigenvalue {eigval} has multiplicity {multiplicity}')
            
            # Compute cosine similarity
            if multiplicity == 1:
                # Get eigenvectors associated with the current eigenvalue
                current_real_eigvec = self.eigvec_dict[eigval][0]
                current_approx_eigvec = approx_eigvec[:,id_]

                # Check if any NaN values are present
                assert not jnp.isnan(current_approx_eigvec).any(), \
                    f'NaN values in the approximated eigenvector: {current_approx_eigvec}'
                
                assert not jnp.isnan(current_real_eigvec).any(), \
                    f'NaN values in the real eigenvector: {current_real_eigvec}'

                # Compute cosine similarity
                pos_sim = (current_real_eigvec).dot(current_approx_eigvec)
                similarities.append(jnp.maximum(pos_sim, -pos_sim))

            else:
                # Get eigenvectors associated with the current eigenvalue
                current_real_eigvec = self.eigvec_dict[eigval]
                current_approx_eigvec = approx_eigvec[:,id_:id_+multiplicity]
                
                # Rotate approximated eigenvectors to match the space spanned by the real eigenvectors
                optimal_approx_eigvec = self.rotate_eigenvectors(
                    current_real_eigvec, current_approx_eigvec)

                norms = jnp.linalg.norm(optimal_approx_eigvec, axis=0, keepdims=True)
                optimal_approx_eigvec = optimal_approx_eigvec / norms.clip(min=1e-10)   # We normalize, since the cosine similarity is invariant to scaling
                
                # Compute cosine similarity
                for j in range(multiplicity):
                    real = current_real_eigvec[j]
                    approx = optimal_approx_eigvec[:,j]
                    pos_sim = (real).dot(approx)
                    similarities.append(jnp.maximum(pos_sim, -pos_sim))

            id_ += multiplicity

        # Convert to array
        similarities = jnp.array(similarities)
        # print(f'Similarities: {similarities}')

        # Compute average cosine similarity
        cosine_similarity = similarities.mean()

        assert not jnp.isnan(similarities).any(), \
            f'NaN values in the cosine similarities: {similarities}'

        return cosine_similarity, similarities
    
    def find_best_basis_for_eigenvectors(
            self, 
            u_list: list, 
            E: jnp.ndarray
        ) -> jnp.ndarray:
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
        u1 = u_list[0].reshape(-1,1)
        A = E.T.dot(E)
        b = 0.5*E.T.dot(u1)
        x = jnp.linalg.solve(A, b)
        u1_approx = E.dot(x).reshape(-1,1).clip(min=1e-10)
        u1_approx = u1_approx / jnp.linalg.norm(u1_approx)
        rotated_eigvec.append(u1_approx)

        # Compute remaining eigenvectors
        for k in range(1, len(u_list)):
            uk = u_list[k].reshape(-1,1)
            Uk = jnp.concatenate(rotated_eigvec, axis=1)
            bk = 0.5*E.T.dot(uk)
            Bk = 0.5*E.T.dot(Uk)
            xk = jnp.linalg.solve(A, bk)
            Xk = jnp.linalg.solve(A, Bk)
            Mk = Uk.T.dot(E)
            Ak = Mk.dot(Xk)
            bbk = Mk.dot(xk)
            mu_leq_k = jnp.linalg.solve(Ak, bbk)
            wk = xk - Xk.dot(mu_leq_k)
            uk_approx = E.dot(wk).reshape(-1,1)
            uk_approx = uk_approx / jnp.linalg.norm(uk_approx).clip(min=1e-10)
            rotated_eigvec.append(uk_approx)

        rotated_eigvec = jnp.concatenate(rotated_eigvec, axis=1)
        return rotated_eigvec
    
    def rotate_eigenvectors(
            self, 
            u_list: list, 
            E: jnp.ndarray
        ) -> jnp.ndarray:
        '''
            Rotate the eigenvectors in E to match the 
            eigenvectors in u_list as close as possible.
            That is, we are finding the optimal basis of
            the subspace spanned by the eigenvectors in E
            such that the angle between the eigenvectors
            in u_list and the rotated eigenvectors is
            minimized.
        '''
        rotation_vectors = []

        # Compute first eigenvector
        u1 = u_list[0].reshape(-1,1)
        w1_times_lambda_1 = 0.5*E.T.dot(u1)
        w1 = w1_times_lambda_1 / jnp.linalg.norm(w1_times_lambda_1).clip(min=1e-10)
        rotation_vectors.append(w1)

        # Compute remaining eigenvectors
        for k in range(1, len(u_list)):
            uk = u_list[k].reshape(-1,1)
            Wk = jnp.concatenate(rotation_vectors, axis=1)
            improper_wk = E.T.dot(uk)
            bk = Wk.T.dot(improper_wk)
            Ak = Wk.T.dot(Wk)
            mu_k = jnp.linalg.solve(Ak, bk)
            wk_times_lambda_k = 0.5*(improper_wk - Wk.dot(mu_k))
            wk = wk_times_lambda_k / jnp.linalg.norm(wk_times_lambda_k).clip(min=1e-10)
            rotation_vectors.append(wk)

        # Use rotation vectors as columns of the optimal rotation matrix
        R = jnp.concatenate(rotation_vectors, axis=1)

        # Obtain list of rotated eigenvectors
        rotated_eigvec = E.dot(R)
        return rotated_eigvec
    
    def compute_maximal_cosine_similarity(self, params_encoder):
        # Get states
        states = self.get_states()

        # Get approximated eigenvectors
        approx_eigvec = self.encoder_fn.apply(params_encoder, states)
        norms = jnp.linalg.norm(approx_eigvec, axis=0, keepdims=True)
        approx_eigvec = approx_eigvec / norms.clip(min=1e-10)
        
        # Select rotation function
        rotation_function = self.rotate_eigenvectors
        
        real_eigvec = []
        for eigval in self.eigvec_dict.keys():
            real_eigvec = real_eigvec + self.eigvec_dict[eigval]
                
        # Rotate approximated eigenvectors to match the space spanned by the real eigenvectors
        optimal_approx_eigvec = rotation_function(
            real_eigvec, approx_eigvec)
        norms = jnp.linalg.norm(optimal_approx_eigvec, axis=0, keepdims=True)
        optimal_approx_eigvec = optimal_approx_eigvec / norms.clip(min=1e-10)   # We normalize, since the cosine similarity is invariant to scaling
        
        # Compute cosine similarity
        similarities = []
        for j in range(self.d):
            real = real_eigvec[j]
            approx = optimal_approx_eigvec[:,j]
            pos_sim = (real).dot(approx)
            similarities.append(jnp.maximum(pos_sim, -pos_sim))

        # Convert to array
        similarities = jnp.array(similarities)

        # Compute average cosine similarity
        cosine_similarity = similarities.mean()

        return cosine_similarity, similarities
    
    def plot_eigenvectors(self, params_encoder):
        """Plot each of the eigenvectors."""
        # Get states
        states = self.get_states()

        # Get approximated eigenvectors
        approx_eigvec = self.encoder_fn.apply(params_encoder, states)
        norms = jnp.linalg.norm(approx_eigvec, axis=0, keepdims=True)
        approx_eigvec = approx_eigvec / norms.clip(min=1e-10)   

        # Obtain sign of first non-zero element of eigenvectors
        first_non_zero_id = jnp.argmax(approx_eigvec != 0, axis=0)
        
        # Choose directions of eigenvectors
        signs = jnp.sign(approx_eigvec[jnp.arange(approx_eigvec.shape[1]), first_non_zero_id])
        approx_eigvec = approx_eigvec * signs.reshape(1,-1)

        grid = self.env.grid.astype(bool)
        vmin = jnp.min(approx_eigvec)
        vmax = jnp.max(approx_eigvec)

        # Plot approximated eigenvectors
        for i in range(self.d):
            eigenvector = approx_eigvec[:,i]
            self.plot_single_eigenvector(states, i, eigenvector, grid, vmin, vmax)

        print('Eigenvectors plotted.')
    
    def plot_single_eigenvector(self, states, eigenvector_id, eigenvector, grid, vmin, vmax):
        """Plot each of the eigenvectors."""
        
        # Obtain x, y, z coordinates, where z is the visitation count
        y = states[:,0]
        x = states[:,1]
        z = eigenvector
                    
        # Calculate tile size
        x_num_tiles = np.unique(x).shape[0]
        x_tile_size = (np.max(x) - np.min(x)) / x_num_tiles
        y_num_tiles = np.unique(y).shape[0]
        y_tile_size = (np.max(y) - np.min(y)) / y_num_tiles

        # Create grid for interpolation
        ti_x = np.linspace(x.min()-x_tile_size, x.max()+x_tile_size, x_num_tiles+2)
        ti_y = np.linspace(y.min()-y_tile_size, y.max()+y_tile_size, y_num_tiles+2)
        XI, YI = np.meshgrid(ti_x, ti_y)

        # Interpolate
        rbf = Rbf(x, y, z, function='cubic')
        ZI = rbf(XI, YI)
        ZI_bounds = 85 * np.ma.masked_where(grid, np.ones_like(ZI))
        ZI_free = np.ma.masked_where(~grid, ZI)
        
        # Generate color mesh
        fig, ax = plt.subplots(1,1, figsize=(10,10))
        mesh = ax.pcolormesh(XI, YI, ZI_free, shading='auto', cmap='coolwarm', vmin=vmin, vmax=vmax)
        ax.pcolormesh(XI, YI, ZI_bounds, shading='auto', cmap='Greys', vmin=0, vmax=255)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(mesh, ax=ax, shrink=0.5, pad=0.05)

        # Save figure
        fig_path = f'./results/visuals/{self.env_name}/learned_eigenvector_{eigenvector_id}_{self.logger.id}.pdf'

        if not os.path.exists(os.path.dirname(fig_path)):
            os.makedirs(os.path.dirname(fig_path))

        plt.savefig(
            fig_path, 
            bbox_inches='tight', 
            dpi=300, 
            transparent=True, 
        )

    @abstractmethod
    def init_additional_params(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def loss_function(self, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def update_training_state(self, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def additional_update_step(self, *args, **kwargs):
        raise NotImplementedError    
