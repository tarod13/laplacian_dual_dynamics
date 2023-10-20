import os
import yaml
from argparse import ArgumentParser
import random
import subprocess
import numpy as np

import jax
import jax.numpy as jnp
import optax

from src.tools import timer_tools

from src.trainer import (
    GeneralizedGraphDrawingObjectiveTrainer,
    AugmentedLagrangianTrainer,
    SQPTrainer,
    CQPTrainer,
)
from src.agent.episodic_replay_buffer import EpisodicReplayBuffer

from src.nets import (
    MLP, ConvNet, generate_hk_module_fn,
)
import wandb

os.environ['WANDB_API_KEY']='83c25550226f8a86fdd4874026d2c0804cd3dc05'
os.environ['WANDB_ENTITY']='tarod13'

def main(hyperparams):
    if hyperparams.wandb_offline:
        os.environ['WANDB_MODE']='offline'

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

    # Override observation mode if Atari
    if hparam_yaml['env_family'] in ['Atari-v5']:
        hparam_yaml['obs_mode'] = 'pixels'

    # Create trainer
    d = hparam_yaml['d']
    algorithm = hparam_yaml['algorithm']
    rng_key = jax.random.PRNGKey(hparam_yaml['seed'])
    hidden_dims = hparam_yaml['hidden_dims']
    obs_mode = hparam_yaml['obs_mode']
    
    # Set encoder network
    if obs_mode not in ['xy']:
        encoder_net = ConvNet
        with open(f'./src/hyperparam/env_params.yaml', 'r') as f:
            env_params = yaml.safe_load(f)
        n_conv_layers = env_params[hparam_yaml['env_name']]['n_conv_layers']
        specific_params = {
            'n_conv_layers': n_conv_layers,
        }
    else:
        encoder_net = MLP
        specific_params = {}
    hparam_yaml.update(specific_params)

    encoder_fn = generate_hk_module_fn(
        encoder_net, 
        d, hidden_dims, 
        hparam_yaml['activation'], 
        **specific_params    
    )   # TODO: Consider the observation space (e.g. pixels)
    
    optimizer = optax.adam(hparam_yaml['lr'])   # TODO: Add hyperparameter to config file
    
    replay_buffer = EpisodicReplayBuffer(max_size=hparam_yaml['n_samples'])   # TODO: Separate hyperparameter for replay buffer size (?)

    if hparam_yaml['use_wandb']:
        # Set wandb save directory
        if hparam_yaml.get('save_dir', None) is None:
            save_dir = os.getcwd()
            os.makedirs(save_dir, exist_ok=True)
            hparam_yaml['save_dir'] = save_dir

        # Initialize wandb logger
        logger = wandb.init(
            project='laplacian-encoder', 
            dir=hparam_yaml['save_dir'],
            config=hparam_yaml,
        )
        # wandb_logger.watch(laplacian_encoder)   # TODO: Test overhead
    else:
        logger = None

    if algorithm == 'ggdo':
        Trainer = GeneralizedGraphDrawingObjectiveTrainer
    elif algorithm == 'al':
        Trainer = AugmentedLagrangianTrainer
    elif algorithm == 'sqp':
        Trainer = SQPTrainer
    elif algorithm == 'cqp':
        Trainer = CQPTrainer
    else:
        raise ValueError(f'Algorithm {algorithm} is not supported.')

    trainer = Trainer(
        encoder_fn=encoder_fn,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        logger=logger,
        rng_key=rng_key,
        **hparam_yaml,
    )
    trainer.train()

    if hparam_yaml['use_wandb'] and hyperparams.wandb_offline:
        os.environ['WANDB_MODE']='online'
        trainer.logger.finish()        

        bash_command = f"wandb sync {os.path.dirname(logger.dir)}"
        subprocess.call(bash_command, shell=True)

    if hparam_yaml['use_wandb']:
        # Delete wandb directory
        bash_command = f"rm -rf {os.path.dirname(logger.dir)}"
        subprocess.call(bash_command, shell=True)
        
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
        "--use_wandb", 
        action="store_true",
        help="Raise the flag to use wandb."
    )

    parser.add_argument(
        "--wandb_offline", 
        action="store_true",
        help="Raise the flag to use wandb offline."
    )

    parser.add_argument(
        "--save_model", 
        action="store_true",
        help="Raise the flag to save the model."
    )

    parser.add_argument(
        '--obs_mode', 
        type=str, 
        default= 'xy',
        help='Observation mode.'
    )

    parser.add_argument(
        '--config_file', 
        type=str, 
        default= 'barrier.yaml',
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
        '--barrier_initial_val', 
        type=float, 
        default=None, 
        help='Initial value for barrier coefficient in the quadratic penalty.'
    )
    parser.add_argument(
        '--lr_barrier_coefs', 
        type=float, 
        default=None, 
        help='Learning rate of the barrier coefficient in the quadratic penalty.'
    )
    
    hyperparams = parser.parse_args()

    main(hyperparams)
