import numpy as np
from gymnasium import ObservationWrapper
from gymnasium import spaces
from src.env.wrapper.transform import (
    normalize_obs_dict,
    normalize_pos_vec,
)

class NormObs(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        obs_dict = {}
        if self.obs_mode in ["xy", "both"]:
            obs_dict["xy_agent"] = spaces.Box(-0.5, 0.5, shape=(2,), dtype=np.float32)
            if self.use_target:
                obs_dict["xy_target"] = spaces.Box(-0.5, 0.5, shape=(2,), dtype=np.float32)
        
        if self.obs_mode in ["pixels", "both"]:
            obs_dict["pixels"] = spaces.Box(
                low=-0.5, high=0.5, shape=(self.height, self.width, 3), dtype=np.float32)

        self.observation_space = spaces.Dict(obs_dict)

    def observation(self, observation):
        grid_shape = self.env.grid.shape
        lims_ = []
        if self.obs_mode in ["xy", "both"]:
            lims_.append(grid_shape)
            if self.use_target:
                lims_.append(grid_shape)
        if self.obs_mode in ["pixels", "both"]:
            lims_.append(255)
        return normalize_obs_dict(observation, lims_)
    
    def get_states(self):
        states = self.env.get_states().copy().astype(np.float32)
        grid_shape = self.env.grid.shape
        states = normalize_pos_vec(states, grid_shape)
        return states
