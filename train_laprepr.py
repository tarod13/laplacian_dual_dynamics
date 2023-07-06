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

from rl_lap.trainer.coefficient_augmented_martin import (
    CoefficientAugmentedLaplacianEncoderTrainerM as Trainer
)   # TODO: Add this class to rl_lap\trainer\__init__.py
from rl_lap.agent.episodic_replay_buffer import EpisodicReplayBuffer

# Equinox version libraries
from rl_lap.nets import MLPeqx, MLPflax, MLPhk, generate_hk_module_fn

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

    # Set random seed
    np.random.seed(hparam_yaml['seed'])   # TODO: Check if this is the best way to set the seed
    random.seed(hparam_yaml['seed'])

    # Initialize timer
    timer = timer_tools.Timer()

    # Create trainer
    nn_library = hparam_yaml['nn_library']
    rng_key = jax.random.PRNGKey(hparam_yaml['seed'])

    if nn_library == 'haiku':
        model = _build_model_haiku(hparam_yaml['d'])
    elif nn_library == 'equinox':
        model = MLPeqx(2, hparam_yaml['d'], [256, 256], rng_key)   # TODO: Add hyperparameters to config file
    elif nn_library == 'flax':
        model = MLPflax([256, 256, hparam_yaml['d']])
    elif nn_library == 'haiku-v2':
        model = generate_hk_module_fn(MLPhk, hparam_yaml['d'], [256, 256])
    else:
        raise ValueError(f'Unknown neural network library: {nn_library}')
    
    optimizer = optax.adam(hparam_yaml['lr'])   # TODO: Add hyperparameter to config file
    replay_buffer = EpisodicReplayBuffer(max_size=hparam_yaml['replay_buffer_size'])
    logger = None
    trainer = Trainer(
        model=model,
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
        default='coefficient_augmented_martin.yaml', 
        help='Configuration file to use.'
    )

    hyperparams = parser.parse_args()

    main(hyperparams)
