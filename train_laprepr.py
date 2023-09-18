import os
import yaml
from argparse import ArgumentParser
import random
import numpy as np

import jax
import jax.numpy as jnp
import optax

from rl_lap.tools import timer_tools

from rl_lap.trainer import (
    CALaplacianEncoderTrainerM,
    DRSSLaplacianEncoderTrainer,
    DualLaplacianEncoderTrainer,
    ExactDualLaplacianEncoderTrainer,
    ScalarBarrierDualLaplacianEncoderTrainer,
)   # TODO: Add this class to rl_lap\trainer\__init__.py
from rl_lap.agent.episodic_replay_buffer import EpisodicReplayBuffer

from rl_lap.nets import (
    MLP, generate_hk_module_fn,
)
import wandb

os.environ['WANDB_API_KEY']='83c25550226f8a86fdd4874026d2c0804cd3dc05'
os.environ['WANDB_ENTITY']='tarod13'
# os.environ['WANDB_MODE']='offline'

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
    additional_params = {}
    
    if algorithm in ['dual', 'dual-rs', 'dual-exact', 'dual-b1']:
        if 'regularization_weight' in hparam_yaml:
            hparam_yaml['barrier_initial_val'] = hparam_yaml['regularization_weight']

        # Initialize dual parameters as lower triangular matrix with ones
        dual_initial_val = hparam_yaml['dual_initial_val']
        additional_params['duals'] = jnp.tril(dual_initial_val * jnp.ones((d, d)), k=0)
        additional_params['dual_velocities'] = jnp.zeros_like(additional_params['duals'])

        # Initialize state dict with error and accumulated error matrices
        additional_params['errors'] = jnp.zeros((d, d))
        
        if algorithm in ['dual-exact']:
            barrier_initial_val = hparam_yaml['barrier_initial_val']
            additional_params['barrier_coefs'] = jnp.tril(barrier_initial_val * jnp.ones((d, d)), k=0)
            additional_params['squared_errors'] = jnp.zeros((d, d))
        elif algorithm in ['dual-b1']:
            barrier_initial_val = hparam_yaml['barrier_initial_val']
            additional_params['barrier_coefs'] = jnp.tril(barrier_initial_val * jnp.ones((1, 1)), k=0)
            additional_params['squared_errors'] = jnp.zeros((1, 1))
    
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
    elif algorithm == 'dual-exact':
        Trainer = ExactDualLaplacianEncoderTrainer
    elif algorithm == 'dual-b1':
        Trainer = ScalarBarrierDualLaplacianEncoderTrainer
    else:
        raise ValueError(f'Algorithm {algorithm} is not supported.')

    trainer = Trainer(
        encoder_fn=encoder_fn,
        additional_params=additional_params,
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
        "exp_label", 
        type=str, 
        help="Experiment label",
    )

    parser.add_argument(
        '--config_file', 
        type=str, 
        default= 'dual_b1.yaml', # 'dual_b1.yaml', #'dual.yaml', #'dual_exact.yaml', #'coefficient_augmented_martin.yaml', # 'dual_relaxed_squared.yaml'
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
