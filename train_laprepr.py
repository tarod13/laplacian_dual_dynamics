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
    model = _build_model_haiku(hparam_yaml['d'])
    optimizer = optax.adam(0.0001)   # TODO: Add hyperparameter to config file
    replay_buffer = EpisodicReplayBuffer(max_size=hparam_yaml['replay_buffer_size'])
    logger = None
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        logger=logger,
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
