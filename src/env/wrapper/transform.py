import numpy as np

def normalize_obs(obs, lim_):
    if isinstance(obs, list):
        obs_list = []
        for o, l in zip(obs, lim_):
            obs_list.append(o.astype(np.float32)/l - 0.5)
        return np.array(obs_list)
    
    else:
        return obs

def normalize_pos_vec(pos, grid_shape):
    x = pos[:,0] / grid_shape[0] - 0.5
    y = pos[:,1] / grid_shape[1] - 0.5
    return np.stack([x, y], axis=1)

def normalize_obs_dict(obs_dict, grid_sizes):
    for obs_type, lim_ in zip(obs_dict.keys(), grid_sizes):
        obs_dict[obs_type] = normalize_obs(
            obs_dict[obs_type], lim_)
    return obs_dict