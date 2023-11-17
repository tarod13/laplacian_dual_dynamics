import yaml
from argparse import ArgumentParser
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

import jax
import jax.numpy as jnp
import gymnasium as gym

from src.tools import timer_tools
from src.nets import (
    ConvNet, generate_hk_module_fn,
)
from src.tools.saving import load_model

def find_color_indices(samples, target_color):
    @jax.jit
    def find_indices():
        indices = []
        for sample in samples:
            # Find the indices where the color matches the target_color
            id = jnp.argwhere(jnp.all(sample == target_color, axis=-1))
            indices.append(id)
        return indices
    return find_indices

def main(hyperparams):
    # Load YAML hyperparameters
    with open(f'./src/hyperparam/{hyperparams.config_file}', 'r') as f:
        hparam_yaml = yaml.safe_load(f)

    # Replace hparams with command line arguments
    for k, v in vars(hyperparams).items():
        if v is not None:
            hparam_yaml[k] = v

    # Set random seed
    np.random.seed(hparam_yaml['seed'])   
    random.seed(hparam_yaml['seed'])

    # Initialize timer
    timer = timer_tools.Timer()

    # Get hyperparameters
    d = hparam_yaml['d']
    hidden_dims = hparam_yaml['hidden_dims']
    env_name = hparam_yaml['env_name']
    
    # Set encoder network
    encoder_net = ConvNet
    with open(f'./src/hyperparam/env_params.yaml', 'r') as f:
        env_params = yaml.safe_load(f)
    n_conv_layers = env_params[env_name]['n_conv_layers'] + 1
    specific_params = {
        'n_conv_layers': n_conv_layers,
    }
    
    hparam_yaml.update(specific_params)

    encoder_fn = generate_hk_module_fn(
        encoder_net, 
        d, hidden_dims, 
        hparam_yaml['activation'], 
        **specific_params    
    )
        
    # Load encoder params
    encoder_path = f'./results/models/{env_name}/{hparam_yaml["encoder_id"]}.pkl'
    loaded_params, _ = load_model(encoder_path)
    loaded_params = loaded_params['encoder']

    # Create environment
    env = gym.make(
        f'ALE/{env_name}-v5',
        render_mode=hparam_yaml["render_mode"],
    )

    # Generate observations
    observation, info = env.reset()
    obs_list = [observation]

    for i in range(hparam_yaml["n_samples"]):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        obs_list.append(observation)

        if terminated or truncated:
            observation, info = env.reset()
            obs_list.append(observation)

    env.close()

    # Process observations
    obs = jnp.stack(obs_list, axis=0)
    obs = jnp.transpose(obs, (0, 3, 1, 2))   # TODO: transpose during training?
    obs_transformed = obs.astype(jnp.float32) / 255
    obs_transformed = jax.device_put(obs_transformed)
    obs = jnp.transpose(obs, (0, 2, 3, 1))

    # Initialize model with the loaded parameters
    init_fn, apply_fn = jax.jit(encoder_fn.init), jax.jit(encoder_fn.apply)
    rng_key = jax.random.PRNGKey(hparam_yaml['seed'])
    params = init_fn(rng_key, obs_transformed[:2,:,:,:])
    #params = jax.tree_util.tree_map(lambda x, y: x._replace(init=y), params, loaded_params)

    # Encode observations
    reps = apply_fn(loaded_params, obs_transformed)
    reps = reps - reps.min(axis=0)
    reps = reps / reps.max(axis=0)
    
    # Green images
    n, h, w, c = obs.shape
    ones = jnp.ones((n,h,w,1), dtype=jnp.float32)
    zeros = jnp.zeros((n,h,w,1), dtype=jnp.float32)
    greens = 255*jnp.concatenate((zeros, ones, zeros), axis=-1)
    whites = 255*jnp.ones((h,w,3), dtype=jnp.float32)
    blacks = jnp.zeros((n,h,w,3)).astype(jnp.float32)

    # Get mean representation:
    # Find first pixel with agent color
    agent_color = jnp.array(hparam_yaml['agent_color'])
    # agent_loc_fn = find_color_indices(obs, agent_color)
    # diff_colors = set([tuple(x) for x in obs.reshape(-1,3).tolist()])

    count_rep = jnp.where(
        jnp.all(obs == agent_color, axis=-1, keepdims=True),
        ones,
        zeros
    ).sum(0)
    agent_loc = jnp.tile(jnp.all(obs == agent_color, axis=-1, keepdims=True), (1,1,1,3))

    for i in tqdm(range(0, d)):
        colors = 255 * cm.winter(reps[:,i])[:,:3]
        sum_rep = jnp.where(
            agent_loc,
            jnp.tile(colors.reshape(-1,1,1,3), (1,h,w,1)),
            blacks,
        ).sum(0)
        
        mean_rep = (sum_rep/(count_rep + 1e-6)).astype(np.uint8).clip(0, 255) # agent_loc_fn()
        mean_obs = obs.astype(jnp.float32).mean(0).astype(np.uint8).clip(0, 255)
        mean_rep = jnp.where(
            mean_rep == 0,
            mean_obs,
            mean_rep,
        )
        # agent_dict = {}
        # for i in range(agent_loc.shape[0]):
        #     n = agent_loc[i,0].item()
        #     if n not in agent_dict:
        #         agent_dict[n] = []
        #     agent_dict[n].append(agent_loc[i,:])

        # Display mean representation
        plt.imshow(mean_rep)
        save_path = f'./results/visuals/atari/{env_name}/eigenfunction_{i}_{hparam_yaml["encoder_id"]}.svg'
        plt.savefig(save_path, dpi=300)

    # Print execution time
    print('Total time cost: {:.4g}s.'.format(timer.time_cost()))

if __name__ == '__main__':

    parser = ArgumentParser()
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
    
    hyperparams = parser.parse_args()

    main(hyperparams)
