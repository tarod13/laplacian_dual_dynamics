import os
from itertools import product
import numpy as np


def txt_to_grid(path):
    def x2zero(cell):
        if cell in ['X','x']:
            return 0
        else:
            return 1

    row_list = []
    with open(path) as f:
        for line in f:
            row = list(line[:-1])
            row = list(map(x2zero, row))
            # Save row if same size as first. This is in
            # order to accept only rectangular grids
            is_first_row = len(row_list) == 0
            print(len(row), row)
            if is_first_row or (len(row_list[0]) == len(row)):
                row_list.append(np.array(row))
    
    grid = np.stack(row_list, axis=0)
    return grid

def grid_to_state_map(grid):
    n_rows, n_cols = grid.shape
    state_map = dict()
    state = 0

    # Assign states to grid cells
    for row, col in product(range(n_rows), range(n_cols)):
        if grid[row,col]:
            state_map[(row, col)] = state
            state += 1

    return state_map

def find_neighbors(row, col, state, state_map):
    if (row-1, col) in state_map:
        state_up = state_map[(row-1, col)]
    else:
        state_up = state
    
    if (row+1, col) in state_map:
        state_down = state_map[(row+1, col)]
    else:
        state_down = state
    
    if (row, col-1) in state_map:
        state_left = state_map[(row, col-1)]
    else:
        state_left = state
    
    if (row, col+1) in state_map:
        state_right = state_map[(row, col+1)]
    else:
        state_right = state
    return state_up, state_down, state_left, state_right

def grid_to_transition_tensor(grid, state_map=None):
    '''Convert grid to transition tensor, assuming deterministic dynamics.'''
    n_actions = 4   # up, down, left, right
    n_states = grid.sum()
    T = np.zeros([n_states, n_actions, n_states])   # Transition tensor (state, action, state')

    # Assign states to grid cells
    if state_map is None:
        state_map = grid_to_state_map(grid)
    
    # Fill transition tensor with probabilities of going from state to state' given action
    for (row, col), state in state_map.items():
        state_up, state_down, state_left, state_right = \
            find_neighbors(row, col, state, state_map)
        
        # Fill transition tensor of visited neighbor states with 1 (deterministic)
        T[state, 0, state_up] = 1
        T[state, 1, state_down] = 1
        T[state, 2, state_left] = 1
        T[state, 3, state_right] = 1

    return T

def obtain_state_map_from_path(path_txt_grid):
    '''Obtain state map from txt file.'''
    grid = txt_to_grid(path_txt_grid)
    state_map = grid_to_state_map(grid)
    return state_map, grid

def obtain_grid_dynamics_from_path(path_txt_grid, policy=None):
    '''Obtain dynamics matrix and transition tensor from grid and policy.'''

    # Obtain grid and state map from txt file
    state_map, grid = obtain_state_map_from_path(path_txt_grid)
    n_states = len(state_map)

    # Initialize policy if not given
    if policy is None:
        n_actions = 4
        policy = np.ones([n_states, n_actions]) / n_actions

    # Obtain transition tensor from grid and state map
    T = grid_to_transition_tensor(grid, state_map)

    # Obtain dynamics matrix from transition tensor and policy
    M = np.einsum('ijk,ij->ik', T, policy)   # Dynamics matrix (state, state')
    
    return M, T, state_map, grid

def load_eig(path_eig):
    if not os.path.exists(path_eig):
        eig = None
        eig_not_found = True
    else:
        with open(path_eig, 'rb') as f:
            eig = np.load(f)
            eigval, eigvec = eig['eigval'], eig['eigvec']

            # Sort eigenvalues and eigenvectors
            idx = np.flip((eigval).argsort())   # TODO: consider negative eigenvalues
            eigval = eigval[idx]
            eigvec = eigvec[:,idx]

            eig = (eigval, eigvec)
        eig_not_found = False
    return eig, eig_not_found