import os
import yaml
from argparse import ArgumentParser
import random
import numpy as np

import jax
import jax.numpy as jnp
import optax

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

import src.env
from src.env.wrapper.norm_obs import NormObs

from src.tools import timer_tools
from src.tools import saving

# from src.trainer import (
#     ScalarBarrierDualLaplacianEncoderTrainer,
# )
from src.agent.episodic_replay_buffer import EpisodicReplayBuffer

from src.nets import (
    MLP, generate_hk_module_fn,
)
import wandb

os.environ['WANDB_API_KEY']='83c25550226f8a86fdd4874026d2c0804cd3dc05'
os.environ['WANDB_ENTITY']='tarod13'
# os.environ['WANDB_MODE']='offline'

def create_env(
        model_fn,
        model_params,
        reward_weights,
        env_name, 
        env_family, 
        seed, 
        max_episode_steps
    ):
    # Create environment
        path_txt_grid = f'./src/env/grid/txts/{env_name}.txt'
        env = gym.make(
            model_fn,
            model_params,
            reward_weights,
            env_family, 
            full_representation = True,
            termination_reward: float = 0.0,
            path=path_txt_grid, 
            render_mode="rgb_array", 
            use_target=False, 
            eig=None,
        )
        # Wrap environment with observation normalization
        obs_wrapper = lambda e: NormObs(e)
        env = obs_wrapper(env)
        # Wrap environment with time limit
        time_wrapper = lambda e: TimeLimit(e, max_episode_steps=max_episode_steps)
        env = time_wrapper(env)

        # Set seed
        env.reset(seed=seed)

        return env

def main(hyperparams):
    # Load YAML hyperparameters
    with open(f'./src/hyperparam/{hyperparams.config_file}', 'r') as f:
        hparam_yaml = yaml.safe_load(f)   # TODO: Check necessity of hyperparams

    # Replace hparams with command line arguments
    for k, v in vars(hyperparams).items():
        if v is not None:
            hparam_yaml[k] = v

    # Set random seed
    np.random.seed(hparam_yaml['seed'])   # TODO: Check if this is the best way to set the seed
    random.seed(hparam_yaml['seed'])

    # Initialize timer
    timer = timer_tools.Timer()

    # Create trainer
    d = hparam_yaml['d']
    rng_key = jax.random.PRNGKey(hparam_yaml['seed'])
    hidden_dims = hparam_yaml['hidden_dims']
    env_name = hparam_yaml['env_name']
    date_time = hparam_yaml['date_time']

    laprep_fn = generate_hk_module_fn(MLP, d, hidden_dims, hparam_yaml['activation'])

    load_path_last = f'./results/models/{env_name}/last_{date_time}.pkl'
    laprep_params = saving.load_model(path=load_path_last)[0]    
    
    # Create environment
    env = create_env(env_name, 'LapGrid-v0', hparam_yaml['seed'], hparam_yaml['max_episode_steps'])
    
    # Print training time
    print('Total time cost: {:.4g}s.'.format(timer.time_cost()))


if __name__ == '__main__':

    parser = ArgumentParser()
    hyperparams = parser.parse_args()
    main(hyperparams)
