from typing import Optional
from itertools import product
import numpy as np

import gymnasium as gym
from gymnasium import spaces
import pygame

from rl_lap.env.grid.utils import txt_to_grid

class GridEnv(gym.Env):
    '''
    Grid environment 
    '''
    metadata = {
        "render_modes": ["human", "rgb_array"], "render_fps": 5000}

    def __init__(
            self, 
            path, 
            render_mode=None, 
            use_target: bool = True
        ):
        self.grid = txt_to_grid(path)
        self.size = self.grid.shape[0]
        self.window_size = 512   # Size of the PyGame window
        self.use_target = use_target

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(   # TODO: use MultiDiscrete instead of Box
            {"agent": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),})
        if self.use_target:
            self.observation_space["target"] = spaces.Box(
                0, self.size - 1, shape=(2,), dtype=int)

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        # Get the agent's location after applying the transformations
        obs = {"agent": self._agent_location}
        
        # Add the target's location to the observation
        if self.use_target:
            obs["target"] = self._target_location

        return obs
    
    def _get_info(self):
        info = {}
        if self.use_target:
            info["distance"] = np.linalg.norm(self._agent_location - self._target_location, ord=1)
        return info
    
    def _on_grid(self, location: np.ndarray) -> bool:
        return self.grid[location[0], location[1]]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        while not self._on_grid(self._agent_location):   # TODO: randomly generate an integer state that is mapped to a valid grid cell
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        if self.use_target:
            self._target_location = self._agent_location
            while (np.array_equal(self._target_location, self._agent_location)
                or (not self._on_grid(self._target_location))   # TODO: randomly generate an integer state that is mapped to a valid grid cell
            ):
                self._target_location = self.np_random.integers(
                    0, self.size, size=2, dtype=int
                )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

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

        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        truncated = False

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        if self.use_target:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * self._target_location,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Now we draw the walls
        for i, j in product(range(self.size),range(self.size)):
            if not self.grid[i,j]:
                pygame.draw.rect(
                    canvas,
                    (110, 110, 110),
                    pygame.Rect(
                        pix_square_size * np.array([i,j]),
                        (pix_square_size, pix_square_size),
                    ),
                )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()