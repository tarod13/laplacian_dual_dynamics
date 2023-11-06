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
    env_name = 'ALE/MontezumaRevenge-v5'
    
    env = gym.make(
        env_name,
    )

    observation, info = env.reset()

    for i in range(10):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
            
    env.close()