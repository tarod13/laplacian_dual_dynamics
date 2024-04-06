import numpy as np
# import cv2
# from PIL import Image
from gymnasium import ObservationWrapper
from gymnasium import spaces
from src.env.wrapper.transform import (
    normalize_obs_dict,
    normalize_pos_vec,
)
import jax

class NormObs(ObservationWrapper):
    def __init__(self, env, reduction_factor=1):
        super().__init__(env)
        self.obs_mode = self.env.unwrapped.obs_mode

        obs_dict = {}
        if self.obs_mode in ["xy", "both", "both-grid"]:
            obs_dict["xy_agent"] = spaces.Box(-0.5, 0.5, shape=(2,), dtype=np.float32)
            if self.unwrapped.use_target:
                obs_dict["xy_target"] = spaces.Box(-0.5, 0.5, shape=(2,), dtype=np.float32)
        
        if self.obs_mode in ["pixels", "both"]:
            obs_dict["pixels"] = spaces.Box(
                low=0, high=255, 
                shape=(
                    self.window_size, 
                    self.window_size, 
                    3), 
                dtype=jax.numpy.uint8)
            
        if self.obs_mode in ["grid", "both-grid"]:
            obs_dict["grid"] = spaces.Box(
                low=0, high=255, 
                shape=(
                    self.height//reduction_factor, 
                    self.width//reduction_factor, 
                    3), 
                dtype=np.uint8)

        self.observation_space = spaces.Dict(obs_dict)
        self.reduction_factor = reduction_factor

    def _resize_pixels(self, pixels):
        if self.reduction_factor == 1:
            return pixels
        
        shape = list(pixels.shape)
        shape[-3] //= self.reduction_factor
        shape[-2] //= self.reduction_factor

        pixels = jax.numpy.array(pixels)
        pixels = jax.image.resize(
                pixels, 
                shape, 
                method=jax.image.ResizeMethod.LANCZOS3
        ).astype(jax.numpy.uint8).clip(0, 255)
        return pixels

    def observation(self, observation):
        grid_shape = np.array(list(self.env.unwrapped.get_grid().shape))
        if self.obs_mode in ["xy", "both", "both-grid"]:
            xy = observation["xy_agent"]
            xy = xy.astype(np.float32) / grid_shape - 0.5
            observation["xy_agent"] = xy
        if self.obs_mode in ["pixels", "both"]:
            observation["pixels"] = self._resize_pixels(observation["pixels"])
        if self.obs_mode in ["grid", "both-grid"]:
            observation["grid"] = observation["grid"]
        return observation
    
    def get_states(self):
        state_dict = self.env.unwrapped.get_states()
        
        if self.obs_mode in ["xy", "both", "both-grid"]:
            xy_states = state_dict["xy_agent"]
            xy_states = xy_states.copy().astype(np.float32)
            grid_shape = np.array(list(self.env.unwrapped.get_grid().shape))
            xy_states = xy_states / grid_shape.reshape(1,-1) - 0.5
            state_dict["xy_agent"] = xy_states
        
        if self.obs_mode in ["pixels", "both"]:
            pixels = state_dict["pixels"]
            pixels = self._resize_pixels(pixels)
            pixels = pixels.copy().astype(np.float32)
            pixels /= 255
            state_dict["pixels"] = pixels
        
        if self.obs_mode in ["grid", "both-grid"]:
            grid = state_dict["grid"]
            grid = grid.copy().astype(np.float32)
            grid /= 255
            state_dict["grid"] = grid

        return state_dict
    
    # def resize_pixels(self, original_image):
    #     if self.reduction_factor == 1:
    #         return original_image

    #     original_height, original_width, _ = original_image.shape
    #     new_width = original_width // self.reduction_factor
    #     new_height = original_height // self.reduction_factor

    #     resized_image = cv2.resize(
    #         original_image, 
    #         (new_width, new_height), 
    #         interpolation = cv2.INTER_AREA
    #     )

    #     return resized_image
    
    # def resize_pixels(self, original_image):
    #     if self.reduction_factor == 1:
    #         return original_image
        
    #     original_image = Image.fromarray(original_image)
    #     original_width, original_height = original_image.size
    #     new_width = original_width // self.reduction_factor
    #     new_height = original_height // self.reduction_factor
    #     resized_image = original_image.resize(
    #         (new_width, new_height), resample=Image.Resampling.BOX)
    #     resized_image = np.array(resized_image).astype(np.uint8).clip(0, 255)
    #     return resized_image
    
class NormObsAtari(ObservationWrapper):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        obs_dict = {}
        obs_dict["pixels"] = spaces.Box(
            low=0, high=255, shape=(210,160,3), dtype=np.uint8)
        self.observation_space = spaces.Dict(obs_dict)
        
    def observation(self, observation):
        obs_dict = {'pixels': observation}
        return obs_dict
