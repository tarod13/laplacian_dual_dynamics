import numpy as np
from PIL import Image
from gymnasium import ObservationWrapper
from gymnasium import spaces
from src.env.wrapper.transform import (
    normalize_obs_dict,
    normalize_pos_vec,
)

class NormObs(ObservationWrapper):
    def __init__(self, env, reduction_factor=1):
        super().__init__(env)

        obs_dict = {}
        if self.obs_mode in ["xy", "both", "both-grid"]:
            obs_dict["xy_agent"] = spaces.Box(-0.5, 0.5, shape=(2,), dtype=np.float32)
            if self.use_target:
                obs_dict["xy_target"] = spaces.Box(-0.5, 0.5, shape=(2,), dtype=np.float32)
        
        if self.obs_mode in ["pixels", "both"]:
            obs_dict["pixels"] = spaces.Box(
                low=0, high=255, 
                shape=(
                    self.window_size//reduction_factor, 
                    self.window_size//reduction_factor, 
                    3), 
                dtype=np.uint8)
            
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

    def observation(self, observation):
        grid_shape = self.env.grid.shape
        lims_ = []
        if self.obs_mode in ["xy", "both", "both-grid"]:
            lims_.append(grid_shape)
            if self.use_target:
                lims_.append(grid_shape)
        if self.obs_mode in ["pixels", "both"]:
            lims_.append(255)
            observation["pixels"] = observation["pixels"]
        if self.obs_mode in ["grid", "both-grid"]:
            lims_.append(255)
            observation["grid"] = observation["grid"]
        return normalize_obs_dict(observation, lims_)
    
    def get_states(self):
        state_dict = self.env.get_states()
        
        if self.obs_mode in ["xy", "both", "both-grid"]:
            xy_states = state_dict["xy_agent"]
            xy_states = xy_states.copy().astype(np.float32)
            grid_shape = self.env.grid.shape
            xy_states = normalize_pos_vec(xy_states, grid_shape)
            state_dict["xy_agent"] = xy_states
        
        if self.obs_mode in ["pixels", "both"]:
            pixels = state_dict["pixels"]
            pixels = pixels.copy().astype(np.float32)
            pixels /= 255
            state_dict["pixels"] = pixels
        
        if self.obs_mode in ["grid", "both-grid"]:
            grid = state_dict["grid"]
            grid = grid.copy().astype(np.float32)
            grid /= 255
            state_dict["grid"] = grid
        
        return state_dict
    
    def resize_pixels(self, original_image):
        if self.reduction_factor == 1:
            return original_image
        
        original_image = Image.fromarray(original_image)
        original_width, original_height = original_image.size
        new_width = original_width // self.reduction_factor
        new_height = original_height // self.reduction_factor
        resized_image = original_image.resize(
            (new_width, new_height), resample=Image.Resampling.BOX)
        resized_image = np.array(resized_image).astype(np.uint8).clip(0, 255)
        return resized_image
    
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
