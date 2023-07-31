import os
import yaml
from argparse import ArgumentParser
import importlib
import random
import numpy as np

import jax
import jax.numpy as jnp
import haiku as hk
import optax

from rl_lap.agent import laprepr_jax   # TODO: Check if this is needed
from rl_lap.tools import flag_tools
from rl_lap.tools import timer_tools
from rl_lap.tools import logging_tools

from rl_lap.trainer import (
    CALaplacianEncoderTrainerM,
    DRSSLaplacianEncoderTrainer,
    DualLaplacianEncoderTrainer,
)   # TODO: Add this class to rl_lap\trainer\__init__.py
from rl_lap.agent.episodic_replay_buffer import EpisodicReplayBuffer

from rl_lap.nets import (
    MLP, generate_hk_module_fn, generate_hk_get_variables_fn,
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
    hidden_dims = hparam_yaml['hidden_dims']

    if (nn_library != 'haiku-v2') and (algorithm == 'dual-rs'):
        raise ValueError(f'Algorithm {algorithm} is not supported with neural network library {nn_library} yet.')

    encoder_fn = generate_hk_module_fn(MLP, d, hidden_dims, hparam_yaml['activation'])
    dual_params = None
    training_state = {}
    
    if algorithm in ['dual', 'dual-rs']:
        # Initialize dual parameters as lower triangular matrix with ones
        dual_params = jnp.tril(jnp.ones((d, d)), k=0)

        # Initialize state dict with error and accumulated error matrices
        training_state['error'] = jnp.zeros((d, d))
        training_state['acc_error'] = jnp.zeros((d, d))
    
    optimizer = optax.adam(hparam_yaml['lr'])   # TODO: Add hyperparameter to config file
    replay_buffer = EpisodicReplayBuffer(max_size=hparam_yaml['n_samples'])   # TODO: Separate hyperparameter for replay buffer size (?)

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
    elif algorithm == 'dual':
        Trainer = DualLaplacianEncoderTrainer

    trainer = Trainer(
        encoder_fn=encoder_fn,
        dual_params=dual_params,
        training_state=training_state,
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
        default= 'coefficient_augmented_martin.yaml', #'dual.yaml', #'coefficient_augmented_martin.yaml', # 'dual_relaxed_squared.yaml'
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
        help='Number of samples.'
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
