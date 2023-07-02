import numpy as np

def normalize_pos(pos, grid_shape):
    # import pdb;pdb.set_trace()
    x = pos[0] / grid_shape[0] - 0.5
    y = pos[1] / grid_shape[1] - 0.5
    return np.array([x, y])

def normalize_pos_vec(pos, grid_shape):
    # import pdb;pdb.set_trace()
    x = pos[:,0] / grid_shape[0] - 0.5
    y = pos[:,1] / grid_shape[1] - 0.5
    return np.stack([x, y], axis=1)

def normalize_obs_dict(obs_dict, grid_shape):
    for obs_type in obs_dict.keys():
        obs_dict[obs_type] = normalize_pos(
            obs_dict[obs_type].astype(np.float32), grid_shape)
    return obs_dict