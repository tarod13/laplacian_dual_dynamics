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
        self.observation_space = spaces.Dict(
            {"agent": spaces.Box(-0.5, 0.5, shape=(2,), dtype=np.float32),})
        if self.use_target:
            self.observation_space["target"] = spaces.Box(
                -0.5, 0.5, shape=(2,), dtype=np.float32)

    def observation(self, observation):
        grid_shape = self.env.grid.shape
        return normalize_obs_dict(observation, grid_shape)
    
    def get_states(self):
        states = self.env.get_states().copy().astype(np.float32)
        grid_shape = self.env.grid.shape
        states = normalize_pos_vec(states, grid_shape)
        return states
