import numpy as np
import jax

from rl_lap.env.grid.grid import GridEnv

class LaplacianGridEnv(GridEnv):
    def __init__(self, 
            model_fn: callable,
            model_params: jax.Array,
            reward_weights: np.ndarray,
            *grid_args,
            full_representation: bool = True,
            termination_reward: float = 0.0,
            **grid_kwargs,
        ):
        super().__init__(*grid_args, **grid_kwargs)
        self.model_fn = model_fn
        self.model_params = model_params
        self.reward_weights = reward_weights
        self.full_representation = full_representation
        self.termination_reward = termination_reward

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        agent_location_new = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        if self.grid[agent_location_new[0], agent_location_new[1]]:
            self._agent_location = agent_location_new

        # An episode is done iff the agent has reached the target
        terminated = False
        if self.use_target:
            terminated = np.array_equal(self._agent_location, self._target_location)

        reward = self.get_reward(terminated)
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        truncated = False

        return observation, reward, terminated, truncated, info

    def get_reward(self, terminated: bool):
        '''Get reward from Laplacian representation'''
        # Agent location to jax array
        agent_location = jax.numpy.array(self._agent_location)

        # Get Laplacian representation from model
        laplacian_representation = self.model_fn.apply(self.model_params, agent_location)
        laplacian_representation = np.array(laplacian_representation).flatten()

        # Get reward from Laplacian representation
        if self.full_representation:
            reward = laplacian_representation
        else:
            reward = np.dot(laplacian_representation, self.reward_weights)

        # Add termination reward
        if terminated:
            reward += self.termination_reward

        return reward

    