from pathlib import Path
from typing import Union
import pickle
import jax

suffix = '.pkl'


def save_model(params, optim_state, path: Union[str, Path], overwrite: bool = False):
    # Check path
    path = Path(path)
    if path.suffix != suffix:
        path = path.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise RuntimeError(f'File {path} already exists.')

    # Create output dictionary   
    params = jax.device_get(params)
    optim_state = jax.device_get(optim_state)
    param_dict = {
        'params': params,
        'optim_state': optim_state,
    }

    # Save output dictionary
    with open(path, 'wb') as file:
        pickle.dump(param_dict, file)

def load_model(path: Union[str, Path]):
    # Check path
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Not a file: {path}')
    if path.suffix != suffix:
        raise ValueError(f'Not a {suffix} file: {path}')   

    # Load input dictionary
    with open(path, 'rb') as f:
        param_dict = pickle.load(f)
    params = param_dict['params']
    optim_state = param_dict['optim_state']
    return params, optim_state
