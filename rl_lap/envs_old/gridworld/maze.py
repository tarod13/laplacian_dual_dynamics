from typing import List
import numpy as np


DEFAULT_MAZE = '''
+-----+
|     |
|     |
|     |
|     |
|     |
+-----+
'''

CUSTOM_MAZE = '''
+-----------+
|     |     |
|     |     |
|           |
|     |     |
|     |     |
+- ---+     |
|     +-- --+
|     |     |
|     |     |
|           |
|     |     |
+-----------+
'''

ROOM64_MAZE = '''
+----+----+----+----+----+----+----+----+
|    |    |    |    |    |    |    |    |
|         |         |         |         |
|    |         |    |    |         |    |
|    |    |    |    |    |    |    |    |
+----+----+----+-- -+- --+----+----+----+
|    |    |    |    |    |    |    |    |
|         |    |    |    |    |         |
|    |         |    |    |         |    |
|    |    |    |    |    |    |    |    |
+-- -+----+-- -+- --+-- -+- --+----+- --+
|    |    |    |    |    |    |    |    |
|    |    |    |    |    |    |    |    |
|    |         |    |    |         |    |
|    |    |    |    |    |    |    |    |
+- --+- --+----+-- -+- --+----+-- -+-- -+
|    |    |    |    |    |    |    |    |
|    |         |    |    |         |    |
|    |    |         |         |    |    |
|    |    |    |         |    |    |    |
+-- -+----+----+----+----+----+----+- --+
|    |    |    |         |    |    |    |
|    |    |         |         |    |    |
|    |         |    |    |         |    |
|    |    |    |    |    |    |    |    |
+- --+- --+----+-- -+- --+----+-- -+-- -+
|    |    |    |    |    |    |    |    |
|    |    |    |    |    |    |    |    |
|    |         |    |    |         |    |
|    |    |    |    |    |    |    |    |
+-- -+----+-- -+- --+-- -+- --+----+- --+
|    |    |    |    |    |    |    |    |
|    |         |    |    |         |    |
|         |    |    |    |    |         |
|    |    |    |    |    |    |    |    |
+----+----+----+-- -+- --+----+----+----+
|    |    |    |    |    |    |    |    |
|    |         |    |    |         |    |
|         |         |         |         |
|    |    |    |    |    |    |    |    |
+----+----+----+----+----+----+----+----+
'''

HARD_MAZE = '''
+--------+-----+
|              |
|              |
+-----+  +-----+
|     |        |
|     |        |
|  +--+-  --+--+
|              |
|              |
|  +  +  +-----+
|  |  |  |     |
|  |  |  |     |
|  +--+  +---  |
|     |        |
|     |        |
+-----+--------+
'''


class MazeFactoryBase:
    def __init__(self, maze_str=DEFAULT_MAZE):
        self._maze = self._parse_maze(maze_str)
        # import pdb; pdb.set_trace()

    def _parse_maze(self, maze_source):
        width = 0
        height = 0
        maze_matrix = []
        for row in maze_source.strip().split('\n'):
            row_vector = row.strip()
            maze_matrix.append(row_vector)
            height += 1
            width = max(width, len(row_vector))
        maze_array = np.zeros([height, width], dtype=str)
        maze_array[:] = ' '
        for i, row in enumerate(maze_matrix):
            for j, val in enumerate(row):
                maze_array[i, j] = val
        return maze_array

    def get_maze(self):
        return self._maze


class SquareRoomFactory(MazeFactoryBase):
    """generate a square room with given size"""
    def __init__(self, size):
        maze_array = np.zeros([size+2, size+2], dtype=str)
        maze_array[:] = ' '
        maze_array[0] = '-'
        maze_array[-1] = '-'
        maze_array[:, 0] = '|'
        maze_array[:, -1] = '|'
        maze_array[0, 0] = '+'
        maze_array[0, -1] = '+'
        maze_array[-1, 0] = '+'
        maze_array[-1, -1] = '+'
        self._maze = maze_array


class FourRoomsFactory(MazeFactoryBase):
    """generate four rooms, each with the given size"""
    def __init__(self, size):
        maze_array = np.zeros([size*2+3, size*2+3], dtype=str)
        maze_array[:] = ' '
        wall_idx = [0, size+1, size*2+2]
        maze_array[wall_idx] = '-'
        maze_array[:, wall_idx] = '|'
        maze_array[wall_idx][:, wall_idx] = '+'
        door_idx = [int((size+1)/2), int((size+1)/2)+1, 
                int((size+1)/2)+size+1, int((size+1)/2)+size+2]
        maze_array[size+1, door_idx] = ' '
        maze_array[door_idx, size+1] = ' '
        self._maze = maze_array


class TwoRoomsFactory(MazeFactoryBase):
    def __init__(self, size):
        maze_array = np.zeros([size+2, size+2], dtype=str)
        maze_array[:] = ' '
        hwall_idx = [0, int((size+1)/2), size+1]
        vwall_idx = [0, size+1]
        maze_array[hwall_idx] = '-'
        maze_array[:, vwall_idx] = '|'
        maze_array[hwall_idx][:, vwall_idx] = '+'
        door_idx = [int((size+1)/2), int((size+1)/2)+1]
        maze_array[hwall_idx[1], door_idx] = ' '
        self._maze = maze_array


class Maze:
    def __init__(self, maze_factory):
        self._maze_factory = maze_factory
        # parse maze ...
        self._maze = None
        self._height = None
        self._width = None
        self._build_maze()
        self._all_empty_grids = np.argwhere(self._maze==' ')
        self._n_states = self._all_empty_grids.shape[0]
        self._pos_indices = {}
        for i, pos in enumerate(self._all_empty_grids):
            self._pos_indices[tuple(pos)] = i
        self._dyn_mat = self._maze_to_uniform_policy_dynamics()
        self._eigvec = self._compute_eigenvectors()
        self._states = self._set_states()

    def _build_maze(self):
        self._maze = self._maze_factory.get_maze()
        self._height = self._maze.shape[0]
        self._width = self._maze.shape[1]

    def rebuild(self):
        self._build_maze()

    def __getitem__(self, key):
        return self._maze[key]

    def __setitem__(self, key, val):
        self._maze[key] = val

    def is_empty(self, pos):
        if (pos[0] >= 0 and pos[0] < self._height 
                and pos[1] >= 0 and pos[1] < self._width):
            return self._maze[tuple(pos)] == ' '
        else:
            return False
    
    @property
    def maze_array(self):
        return self._maze

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def n_states(self):
        return self._n_states
    
    @property
    def dyn_mat(self):
        return self._dyn_mat
    
    @property
    def eigvec(self):
        return self._eigvec
    
    @property
    def states(self):
        return self._states

    def pos_index(self, pos):
        return self._pos_indices[tuple(pos)]

    def all_empty_grids(self):
        return np.argwhere(self._maze==' ')

    def random_empty_grids(self, k):
        '''Return k random empty positions.'''
        empty_grids = np.argwhere(self._maze==' ')
        selected = np.random.choice(
                np.arange(empty_grids.shape[0]),
                size=k,
                replace=False
                )
        return empty_grids[selected]

    def first_empty_grid(self):
        empty_grids = np.argwhere(self._maze==' ')
        assert empty_grids.shape[0] > 0
        return empty_grids[0]

    def render(self):
        # 0 for ground, 1 for wall
        return (self._maze!=' ').astype(np.float32)
    
    def _find_neighbors(self, row, col, state):
        if (row-1, col) in self._pos_indices:
            state_up = self._pos_indices[(row-1, col)]
        else:
            state_up = state
        
        if (row+1, col) in self._pos_indices:
            state_down = self._pos_indices[(row+1, col)]
        else:
            state_down = state
        
        if (row, col-1) in self._pos_indices:
            state_left = self._pos_indices[(row, col-1)]
        else:
            state_left = state
        
        if (row, col+1) in self._pos_indices:
            state_right = self._pos_indices[(row, col+1)]
        else:
            state_right = state
        return state_up, state_down, state_left, state_right
    
    def _maze_to_transition_tensor(self):
        '''Convert grid to transition tensor, assuming deterministic dynamics.'''
        n_actions = 4   # up, down, left, right
        n_states = self._n_states
        T = np.zeros([n_states, n_actions, n_states])   # Transition tensor (state, action, state')

        # Fill transition tensor with probabilities of going from state to state' given action
        for (row, col), state in self._pos_indices.items():
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
            policy = np.ones([self._n_states, n_actions]) / n_actions
        
        # Obtain transition tensor
        T = self._maze_to_transition_tensor()

        # Obtain dynamics matrix from transition tensor and policy
        M = np.einsum('ijk,ij->ik', T, policy)   # Dynamics matrix (state, state')
        
        return M

    def _compute_eigenvectors(self) -> List[np.ndarray]:
        # Calculate eigenvectors
        eigvals, eigvecs = np.linalg.eig(self._dyn_mat)
        eigvals = eigvals.real
        eigvecs = eigvecs.real

        # Sort eigenvectors from largest to smallest eigenvalue, 
        # given that we are using the dynamics matrix instead of 
        # the successor representation matrix
        idx = np.flip(eigvals.argsort())
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:,idx]

        # Normalize eigenvectors
        eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=0, keepdims=True)

        # Choose directions of eigenvectors
        eigvecs = np.sign(eigvecs[0,:].reshape(1,-1)) * eigvecs

        return eigvecs
    
    def _set_states(self):
        maze_shape = self._maze.shape
        states = self._all_empty_grids.copy().astype(np.float32)
        states[:, 0] = states[:, 0] / maze_shape[0] - 0.5
        states[:, 1] = states[:, 1] / maze_shape[1] - 0.5
        return states
