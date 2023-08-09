import os
import sys
from itertools import product
from typing import Optional, List, Tuple

import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from mpmath import mp
mp.prec = 128
try:
    from flint import acb_mat, ctx
    ctx.prec = 128
    FLINT_INSTALLED = True
except ImportError:
    print("Warning: flint not installed. Using mpmath instead.")
    FLINT_INSTALLED = False

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
            use_target: bool = True,
            eig: Optional[Tuple] = None,
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

        # Create a state index dictionary
        self._states = np.argwhere(self.grid) #.astype(np.float32)
        self.n_states = self._states.shape[0]
        self._state_indices = {}
        for i, pos in enumerate(self._states):
            self._state_indices[tuple(pos)] = i

        # Compute the dynamics matrix
        self._dyn_mat = self._maze_to_uniform_policy_dynamics()

        # Compute the eigenvectors and eigenvalues of the dynamics matrix
        if eig is None:
            self._eigval, self._eigvec = self._compute_eigenvectors()
        else:
            self._eigval, self._eigvec = eig

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

    def _find_neighbors(self, row, col, state):
        if (row-1, col) in self._state_indices:
            state_up = self._state_indices[(row-1, col)]
        else:
            state_up = state
        
        if (row+1, col) in self._state_indices:
            state_down = self._state_indices[(row+1, col)]
        else:
            state_down = state
        
        if (row, col-1) in self._state_indices:
            state_left = self._state_indices[(row, col-1)]
        else:
            state_left = state
        
        if (row, col+1) in self._state_indices:
            state_right = self._state_indices[(row, col+1)]
        else:
            state_right = state
        return state_up, state_down, state_left, state_right

    def _maze_to_transition_tensor(self):
        '''Convert grid to transition tensor, assuming deterministic dynamics.'''
        n_actions = 4   # up, down, left, right
        n_states = self.n_states
        T = np.zeros([n_states, n_actions, n_states])   # Transition tensor (state, action, state')

        # Fill transition tensor with probabilities of going from state to state' given action
        for (row, col), state in self._state_indices.items():
            state_up, state_down, state_left, state_right = \
                self._find_neighbors(row, col, state)
            
            # Fill transition tensor of visited neighbor states with 1 (deterministic)
            T[state, 0, state_up] = 1
            T[state, 1, state_down] = 1
            T[state, 2, state_left] = 1
            T[state, 3, state_right] = 1

        return T
    
    def _maze_to_uniform_policy_dynamics(self, policy=None):
        '''Convert grid to transition matrix, assuming uniform policy.'''
        
        # Initialize policy if not given
        if policy is None:
            n_actions = 4
            policy = np.ones([self.n_states, n_actions]) / n_actions
        
        # Obtain transition tensor
        T = self._maze_to_transition_tensor()

        # Obtain dynamics matrix from transition tensor and policy
        M = np.einsum('ijk,ij->ik', T, policy)   # Dynamics matrix (state, state')
        
        return M

    def _compute_eigenvectors(self) -> List[np.ndarray]:
        # if np.allclose(self._dyn_mat, self._dyn_mat.T):
        #     eig_function = 

        mp_mat = mp.matrix(self._dyn_mat.tolist())
        # Calculate eigenvectors
        if FLINT_INSTALLED:
            flint_mat = acb_mat(mp_mat)
            eigvals, eigvecs = flint_mat.eig(right=True, algorithm="approx")
            eigvals = np.array(eigvals).astype(np.clongdouble).real.flatten()   # real since we assume the dynamics matrix is symmetric
            eigvecs = np.array(eigvecs.tolist()).astype(np.clongdouble).real
        else:
            eigvals, eigvecs = mp.eigsy(mp_mat)   # eigsy since we assume the dynamics matrix is symmetric
            eigvals = np.array(eigvals.tolist()).astype(np.longdouble).flatten()  
            eigvecs = np.array(eigvecs.tolist()).astype(np.longdouble)

        # Sort eigenvectors from largest to smallest eigenvalue, 
        # given that we are using the dynamics matrix instead of 
        # the successor representation matrix
        idx = np.flip((eigvals**2).argsort())
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:,idx]

        # Normalize eigenvectors
        eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=0, keepdims=True)

        # Choose directions of eigenvectors
        eigvecs = np.sign(eigvecs[0,:].reshape(1,-1)) * eigvecs

        # Check if symmetric
        if np.allclose(self._dyn_mat, self._dyn_mat.T):
            print('Dynamics matrix is symmetric.')
        else:
            print('Dynamics matrix is not symmetric.')

        return eigvals, eigvecs
    
    def get_states(self):
        return self._states
    
    def get_eigenvectors(self):
        return self._eigvec
    
    def get_eigenvalues(self):
        return self._eigval
    
    def round_eigenvalues(self, decimals=5):
        self._eigval = np.round(self._eigval, decimals=decimals)

    def save_eigenpairs(self, filename):
        # Create directory if it does not exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save eigenvalues and eigenvectors
        with open(filename, 'wb') as f:
            np.savez_compressed(f, eigval=self._eigval, eigvec=self._eigvec)
    