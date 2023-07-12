import os
import yaml
from argparse import ArgumentParser
import importlib
import random
import numpy as np

import jax
import haiku as hk
import optax

from rl_lap.agent import laprepr_jax   # TODO: Check if this is needed
from rl_lap.tools import flag_tools
from rl_lap.tools import timer_tools
from rl_lap.tools import logging_tools

from rl_lap.trainer import (
    CALaplacianEncoderTrainerM,
    DRSSLaplacianEncoderTrainer,
)   # TODO: Add this class to rl_lap\trainer\__init__.py
from rl_lap.agent.episodic_replay_buffer import EpisodicReplayBuffer

# Equinox version libraries
from rl_lap.nets import (
    MLP_eqx, MLP_flax, MLP_hk, DualMLP_hk,
    generate_hk_module_fn, generate_hk_get_variables_fn,
)
import wandb

os.environ['WANDB_API_KEY']='83c25550226f8a86fdd4874026d2c0804cd3dc05'
os.environ['WANDB_ENTITY']='tarod13'

def _build_model_haiku(d):   # TODO: Choose a better location for this function
    def lap_net(obs):
        network = hk.Sequential([
            hk.Linear(256),   # TODO: Add hyperparameters to config file
            jax.nn.relu,
            hk.Linear(256),
            jax.nn.relu,
            hk.Linear(d),
        ])
        return network(obs.astype(np.float32))
    return hk.without_apply_rng(hk.transform(lap_net))

def main(hyperparams):
    # Load YAML hyperparameters
    with open(f'./rl_lap/hyperparam/{hyperparams.config_file}', 'r') as f:
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
    algorithm = hparam_yaml['algorithm']
    nn_library = hparam_yaml['nn_library']
    rng_key = jax.random.PRNGKey(hparam_yaml['seed'])

    if (nn_library != 'haiku-v2') and (algorithm == 'dual-rs'):
        raise ValueError(f'Algorithm {algorithm} is not supported with neural network library {nn_library} yet.')

    if nn_library == 'haiku':
        model_funcs = {'forward': _build_model_haiku(d)}
    elif nn_library == 'equinox':
        model_funcs = {'forward': MLP_eqx(2, d, [256, 256], rng_key)}   # TODO: Add hyperparameters to config file   
    elif nn_library == 'flax':
        model_funcs = {'forward': MLP_flax([256, 256, d])}
    elif nn_library == 'haiku-v2':
        if algorithm == 'coef-a':
            model_funcs = {'forward': generate_hk_module_fn(MLP_hk, d, [256, 256])}
        elif algorithm == 'dual-rs':
            args_ = [
                MLP_hk, d, [256, 256], 
                hparam_yaml['dual_initial_val'], 
                hparam_yaml['use_lower_triangular'],
            ]
            model_funcs = {
                'forward': generate_hk_module_fn(
                    DualMLP_hk, *args_),
                'get_duals': generate_hk_get_variables_fn(
                    DualMLP_hk, 'get_duals', *args_),
                'get_errors': generate_hk_get_variables_fn(
                    DualMLP_hk, 'get_errors', *args_),
                'get_error_accumulation': generate_hk_get_variables_fn(
                    DualMLP_hk, 'get_error_accumulation', *args_),
            }
    else:
        raise ValueError(f'Unknown neural network library: {nn_library}')
    
    optimizer = optax.adam(hparam_yaml['lr'])   # TODO: Add hyperparameter to config file
    replay_buffer = EpisodicReplayBuffer(max_size=hparam_yaml['replay_buffer_size'])

    if hparam_yaml['use_wandb']:
        logger = wandb.init(
            project='laplacian-encoder', 
            dir=hyperparams.save_dir,
            config=hparam_yaml,
        )   
        # wandb_logger.watch(laplacian_encoder)   # TODO: Test overhead
    else:
        logger = None

    if algorithm == 'coef-a':
        Trainer = CALaplacianEncoderTrainerM
    elif algorithm == 'dual-rs':
        Trainer = DRSSLaplacianEncoderTrainer

    trainer = Trainer(
        model_funcs=model_funcs,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        logger=logger,
        rng_key=rng_key,
        **hparam_yaml,
    )
    trainer.train()

    # Print training time
    print('Total time cost: {:.4g}s.'.format(timer.time_cost()))


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument(
        '--config_file', 
        type=str, 
        default='coefficient_augmented_martin.yaml', # 'dual_relaxed_squared.yaml'
        help='Configuration file to use.'
    )
    parser.add_argument(
        '--save_dir', 
        type=str, 
        default=None, 
        help='Directory to save the model.'
    )
    parser.add_argument(
        '--n_samples', 
        type=int, 
        default=None, 
        help='Batch size.'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=None, 
        help='Batch size.'
    )
    parser.add_argument(
        '--discount', 
        type=float, 
        default=None, 
        help='Lambda discount used for sampling states.'
    )
    parser.add_argument(
        '--total_train_steps', 
        type=int, 
        default=None, 
        help='Number of training steps for laplacian encoder.'
    )
    parser.add_argument(
        '--max_episode_steps', 
        type=int, 
        default=None, 
        help='Maximum trajectory length.'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=None, 
        help='Seed for random number generators.'
    )
    parser.add_argument(
        '--env_name', 
        type=str, 
        default=None, 
        help='Environment name.'
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=None, 
        help='Learning rate of the Adam optimizer used to train the laplacian encoder.'
    )
    parser.add_argument(
        '--hidden_dims',
        nargs='+',
        type=int,
        help='Hidden dimensions of the laplacian encoder.'
    )
    parser.add_argument(
        '--regularization_weight', 
        type=float, 
        default=None, 
        help='Regularization weight.'
    )
    
    hyperparams = parser.parse_args()

    main(hyperparams)
