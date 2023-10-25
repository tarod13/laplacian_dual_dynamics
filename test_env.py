# from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
import src.env
from src.env.wrapper.norm_obs import NormObs
from src.env.grid.utils import load_eig

import jax

# import cv2

def resize_pixels(original_image, reduction_factor=1):
    if reduction_factor == 1:
        return original_image

    original_height, original_width, n_channels = original_image.shape
    new_width = original_width // reduction_factor
    new_height = original_height // reduction_factor

    # resized_image = cv2.resize(
    #     original_image, 
    #     (new_width, new_height), 
    #     interpolation = cv2.INTER_AREA
    # )
    resized_image = jax.numpy.array(original_image)
    resized_image = jax.image.resize(
        resized_image, 
        (new_width, new_height, n_channels), 
        method=jax.image.ResizeMethod.LANCZOS3
    ).astype(jax.numpy.float32) / 255

    # JAX to numpy
    resized_image = np.array(resized_image)

    return resized_image

if __name__ == "__main__":
    use_wrapper = True
    reduction_factor = 1
    # filter = Image.Resampling.BOX
    env_name = 'GridRoom-64'
    obs_mode = 'pixels'
    path_txt_grid = f'./src/env/grid/txts/{env_name}.txt'
    path_eig = f'./src/env/grid/eigval/{env_name}.npz'

    eig = load_eig(path_eig)[0]
    env = gym.make(
        'Grid-v0', 
        path=path_txt_grid, 
        render_mode="human", 
        render_fps=20, 
        eig=eig, 
        obs_mode=obs_mode, 
        calculate_eig=False,
        window_size=172,
        use_target=False,
    )

    if use_wrapper:
        obs_wrapper = lambda e: NormObs(e, reduction_factor=1)
        env = obs_wrapper(env)

    print(env.grid.shape)
    observation, info = env.reset()

    for i in range(10):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)


        if (obs_mode in observation.keys()) and (obs_mode != 'xy') and (i == 0):
            original_image = observation['pixels' if obs_mode in ['pixels', 'both'] else 'grid']
            original_image = np.array(original_image)
            resized_image = resize_pixels(original_image, reduction_factor=reduction_factor)
            # original_image = Image.fromarray(original_image)

            # original_width, original_height = original_image.size
            # new_width = original_width // reduction_factor
            # new_height = original_height // reduction_factor
            # resized_image = original_image.resize((new_width, new_height), resample=filter)
            # resized_image = np.array(resized_image)
            # original_image = np.array(original_image)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(original_image)
            axes[0].set_title('Original Image')
            axes[1].imshow(resized_image)
            axes[1].set_title('Resized Image')
            for ax in axes:
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(f'./results/visuals/{env_name}/{obs_mode}_obs{"_wrapper" if use_wrapper else ""}.png', dpi=300)
            
        if terminated or truncated:
            observation, info = env.reset()
            
    env.close()