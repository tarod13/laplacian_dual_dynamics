import os
import logging
import collections

import haiku as hk
from haiku import nets
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import optax
import pickle

from . import episodic_replay_buffer
from ..envs_old import actors
from ..tools import py_tools
from ..tools import flag_tools
from ..tools import summary_tools
from ..tools import timer_tools

# New libraries to use gym environments
import random
import gymnasium as gym
from gymnasium.wrappers import (
    TimeLimit, TransformObservation
)

import rl_lap.env
from rl_lap.env.wrapper.transform import normalize_obs_dict
from rl_lap.agent.agent import BehaviorAgent as Agent
from rl_lap.policy import DiscreteUniformRandomPolicy as Policy

Data = collections.namedtuple("Data", "s1 s2 s_neg s_neg_2")


def neg_loss_fn(phi_u, phi_v):
    rep_dim = phi_u.size
    loss = 0
    for dim in range(rep_dim, 0, -1):
        x_norm = jnp.sqrt(jnp.dot(phi_u[:dim], phi_u[:dim]))
        y_norm = jnp.sqrt(jnp.dot(phi_v[:dim], phi_v[:dim]))
        dot_product = jnp.dot(phi_u[:dim], phi_v[:dim])
        loss += (
            dot_product ** 2 - x_norm ** 2 / rep_dim  - y_norm ** 2 / rep_dim  )
            
    return loss


def generalized_graph_drawing_loss_haiku(pos_rep_i, pos_rep_j, neg_rep, neg_rep_2, reward=None, alpha=1.0, beta=2.0):
    coeff_vector = jnp.arange(pos_rep_i.shape[1], 0, -1)
    pos_loss = ((pos_rep_i - pos_rep_j)**2).dot(coeff_vector).mean()
    neg_loss_vmap = jax.vmap(neg_loss_fn)
    neg_loss  = neg_loss_vmap(neg_rep, neg_rep_2).mean()
    loss = pos_loss + beta * neg_loss
    return loss, pos_loss, neg_loss


def _build_model_haiku(d):
    def lap_net(obs):
        network = hk.Sequential([
            # hk.Linear(256),
            # jax.nn.relu,
            hk.Linear(256),
            jax.nn.relu,
            hk.Linear(256),
            jax.nn.relu,
            hk.Linear(d),
        ])
        return network(obs.astype(np.float32))
    return hk.without_apply_rng(hk.transform(lap_net))


class LapReprLearner:

    @py_tools.store_args
    def __init__(self,
            d,
            max_distance,
            # pytorch
            device=None,
            # env args
            action_spec=None,
            obs_shape=None,
            obs_prepro=None,
            env_factory=None,
            env_name=None,
            env_family=None,
            # learner args
            model_cfg=None,
            optimizer_cfg=None,
            n_samples=10000,
            batch_size=128,
            discount=0.0,
            w_neg=1.0,
            c_neg=1.0,
            reg_neg=0.0,
            replay_buffer_size=100000,
            # trainer args
            log_dir='/tmp/rl/log',
            total_train_steps=50000,
            print_freq=1000,
            save_freq=10000,
            ):
        self._build()

    def _build(self):
        logging.info('device: {}.'.format(self._device))

        self._repr_fn = _build_model_haiku(self._d)
        self._optimizer = optax.adam(0.0001)
        self._train_step = jax.jit(self._train_step)

        self._replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
                max_size=self._replay_buffer_size)
        self._global_step = 0
        self._train_info = collections.OrderedDict()
    
    def _random_policy_fn(self, state):   # TODO: Fix this function
        action = self._action_spec.sample()
        return action, None

    def _get_obs_batch(self, steps):   # TODO: Check this function (way to build the batch)
        obs_batch = [self._obs_prepro(s.step.agent_state["agent"])
                for s in steps]
        return np.stack(obs_batch, axis=0)

    def _get_rew_batch(self, steps):   # TODO: Fix this function
        rew_batch = [s.step.time_step.reward
                for s in steps]
        return np.stack(rew_batch, axis=0)

    def _tensor(self, x):
        return torch_tools.to_tensor(x, self._device)

    def _get_train_batch(self):
        s1, s2 = self._replay_buffer.sample_pairs(
                batch_size=self._batch_size,
                discount=self._discount,
                )
        s_neg = self._replay_buffer.sample_steps(self._batch_size)
        s_neg_2 = self._replay_buffer.sample_steps(self._batch_size)
        s1_pos, s2_pos, s_neg, s_neg_2 = map(self._get_obs_batch, [s1, s2, s_neg, s_neg_2])
        batch = Data(s1_pos, s2_pos, s_neg, s_neg_2)
        return batch

    def _loss(self, params, batch, alpha, beta):
        s1_repr = self._repr_fn.apply(params, batch.s1)
        s2_repr = self._repr_fn.apply(params, batch.s2)
        s_neg_x_repr = self._repr_fn.apply(params, batch.s_neg)
        s_neg_y_repr = self._repr_fn.apply(params, batch.s_neg_2)
        
        loss, loss_positive, loss_negative = generalized_graph_drawing_loss_haiku(
            s1_repr, s2_repr, s_neg_x_repr, s_neg_y_repr, alpha=alpha, beta=beta)

        return loss, (loss, loss_positive, loss_negative)

    def _train_step(self, train_batch, params, opt_state, alpha=1.0, beta=5.0):
        # _, aux = self._loss(params, train_batch, alpha, beta)
        grads, aux = jax.grad(self._loss, has_aux=True)(params, train_batch, alpha, beta)
        updates, opt_state = self._optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        cosine_similarity = self._compute_cosine_similarity(params)
        return params, opt_state, aux, cosine_similarity
    
    def _compute_cosine_similarity(self, params):
        # Get baseline parameters
        states = self._env.task.states
        real_eigvec = self._env.task.maze.eigvec[:,:self._d]
        real_norms = jnp.linalg.norm(real_eigvec, axis=0, keepdims=True)
        real_eigvec = real_eigvec / real_norms

        # Get approximated eigenvectors
        approx_eigvec = self._repr_fn.apply(params, states)
        norms = jnp.linalg.norm(approx_eigvec, axis=0, keepdims=True)
        approx_eigvec = approx_eigvec / norms
        
        # Compute cosine similarities for both directions
        sim_first_dir = (approx_eigvec * real_eigvec).sum(axis=0)
        sim_second_dir = (- approx_eigvec * real_eigvec).sum(axis=0)

        # Take the maximum similarity for each eigenvector
        similarities = jnp.maximum(sim_first_dir, sim_second_dir)
        cosine_similarity = similarities.mean()
        return cosine_similarity

    def _print_train_info(self):
        summary_str = summary_tools.get_summary_str(
                step=self._global_step, info=self._train_info)
        logging.info(summary_str)

    def train(
            self, 
            max_episode_steps: int = 50,
            random_number_generator: random.Random = None,
            seed: int = 1337,
        ):
        saver_dir = self._log_dir
        if not os.path.exists(saver_dir):
            os.makedirs(saver_dir)

        # Create environment
        path_txt_grid = f'./rl_lap/env/grid/txts/{self._env_name}.txt'
        env = gym.make(
            self._env_family, 
            path=path_txt_grid, 
            render_mode="rgb_array", 
            use_target=False, 
        )
        # Wrap environment with observation normalization
        obs_wrapper = lambda e: TransformObservation(
            e, lambda o: normalize_obs_dict(o, np.array([e.size, e.size]))
        )
        env = obs_wrapper(env)
        # Wrap environment with time limit
        time_wrapper = lambda e: TimeLimit(e, max_episode_steps=max_episode_steps)
        env = time_wrapper(env)
        env.reset(seed=seed)

        # Create agent
        seed_policy = seed if random_number_generator is None else None
        policy = Policy(
            num_actions=env.action_space.n, 
            random_number_generator=random_number_generator, 
            seed=seed_policy
        )
        agent = Agent(policy)

        # start actors, collect trajectories from random actions
        logging.info('Start collecting samples.')
        timer = timer_tools.Timer()
        # collect initial transitions
        total_n_steps = 0
        collect_batch = 10_000
        while total_n_steps < self._n_samples:
            n_steps = min(collect_batch, 
                    self._n_samples - total_n_steps)
            steps = agent.collect_experience(env, n_steps)
            self._replay_buffer.add_steps(steps)
            total_n_steps += n_steps
            logging.info('({}/{}) steps collected.'
                .format(total_n_steps, self._n_samples))
        time_cost = timer.time_cost()
        logging.info('Data collection finished, time cost: {}s'
            .format(time_cost))

        
        rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
        sample_input = self._get_train_batch()
        params = self._repr_fn.init(next(rng), sample_input.s1)
        opt_state = self._optimizer.init(params)

        # learning begins
        timer.set_step(0)
        for step in range(self._total_train_steps):

            train_batch = self._get_train_batch()
            params, opt_state, losses, cosine_similarity = self._train_step(train_batch, params, opt_state)

            self._global_step += 1
            self._train_info['loss_total'] = np.array([jax.device_get(losses[0])])[0]
            self._train_info['loss_pos'] = np.array([jax.device_get(losses[1])])[0]
            self._train_info['loss_neg'] = np.array([jax.device_get(losses[2])])[0]
            self._train_info['cos_sim'] = np.array([jax.device_get(cosine_similarity)])[0]

            # save
            if (step + 1) % self._save_freq == 0:
                saver_path = os.path.join(saver_dir, 'model-{}.pkl'.format(step+1))
                self.save_ckpt(saver_path, params)
            # # print info
            if step == 0 or (step + 1) % self._print_freq == 0:
                steps_per_sec = timer.steps_per_sec(step)
                logging.info('Training steps per second: {:.4g}.'
                        .format(steps_per_sec))
                self._print_train_info()
        saver_path = os.path.join(saver_dir, 'model.pkl')
        self.save_ckpt(saver_path, params)
        time_cost = timer.time_cost()
        logging.info('Training finished, time cost {:.4g}s.'.format(time_cost))


        # plot_dir = saver_dir.replace("laprepr", "visuals")
        # if not os.path.exists(plot_dir):
        #     os.makedirs(plot_dir)
        # # get all states representation
        # n_states = self._env.task.maze.n_states
        # pos_batch = self._env.task.maze.all_empty_grids()
        # obs_batch = [self._env.task.pos_to_obs(pos_batch[i]) for i in range(n_states)]
        # states_batch = np.array([self._obs_prepro(obs) for obs in obs_batch])

        # # get goal state representation
        # goal_pos = self._env.task.goal_pos
        # goal_obs = self._env.task.pos_to_obs(goal_pos)
        # goal_state = self._obs_prepro(goal_obs)[None]

        # # get representations from loaded model
        # goal_repr = self._repr_fn.apply(params, goal_state)
        # states_reprs = self._repr_fn.apply(params, states_batch)

        # # compute l2 distances from states to goal
        # l2_dists = np.sqrt(np.sum(np.square(states_reprs - goal_repr), axis=-1))
        # image_shape = goal_obs.agent.image.shape
        # map_ = np.zeros(image_shape[:2], dtype=np.float32)
        # map_[pos_batch[:, 0], pos_batch[:, 1]] = l2_dists
        # im_ = plt.imshow(map_, interpolation='none', cmap='Blues')
        # plt.colorbar()

        # # add the walls to the normalized distance plot
        # walls = np.expand_dims(self._env.task.maze.render(), axis=-1)
        # map_2 = im_.cmap(im_.norm(map_))
        # map_2[:, :, :-1] = map_2[:, :, :-1] * (1 - walls) + 0.5 * walls
        # map_2[:, :, -1:] = map_2[:, :, -1:] * (1 - walls) + 1.0 * walls
        # map_2[goal_pos[0], goal_pos[1]] = [1, 0, 0, 1]
        # plt.cla()
        # plt.imshow(map_2, interpolation='none')
        # plt.xticks([])
        # plt.yticks([])
        # plt.title('Distance to goal')
        # figfile = os.path.join(plot_dir, 'l2dist_goal.png')
        # plt.savefig(figfile, bbox_inches='tight')
        # plt.clf()

        # # -- visialize state representations --
        # # plot raw distances with the walls
        # image_shape = goal_obs.agent.image.shape
        # map_ = np.zeros(image_shape[:2], dtype=np.float32)
        # eigen=0
        # for eigen in range(min(50, self._d)):
        #     map_[pos_batch[:, 0], pos_batch[:, 1]] = states_reprs[:, eigen]
        #     im_ = plt.imshow(map_, interpolation='none', cmap='Blues')
        #     plt.colorbar()

        #     # add the walls
        #     walls = np.expand_dims(self._env.task.maze.render(), axis=-1)
        #     map_2 = im_.cmap(im_.norm(map_))
        #     map_2[:, :, :-1] = map_2[:, :, :-1] * (1 - walls) + 0.5 * walls
        #     map_2[:, :, -1:] = map_2[:, :, -1:] * (1 - walls) + 1.0 * walls
        #     plt.cla()
        #     plt.imshow(map_2, interpolation='none')
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.title(f'Representation dim={eigen}')
        #     figfile = os.path.join(plot_dir, f'eigen{eigen}.png')
        #     plt.savefig(figfile, bbox_inches='tight')
        #     plt.clf()


    def save_ckpt(self, filepath, params):
        numpy_params = jax.device_get(params)
        with open(filepath, 'wb') as file:
            pickle.dump(numpy_params, file)
