from typing import Callable
import haiku as hk
import jax
import numpy as np

def generate_hk_module_fn(module_class: hk.Module, *args, **kwargs) -> Callable:
    def module_fn(x: np.ndarray, **fn_kwargs) -> jax.Array:
        module = module_class(*args, **kwargs)
        return module(x.astype(np.float32), **fn_kwargs)
    return hk.without_apply_rng(hk.transform(module_fn))

def generate_hk_get_variables_fn(module_class: hk.Module, function_name: str, *args, **kwargs) -> Callable:
    def module_fn() -> jax.Array:
        module = module_class(*args, **kwargs)
        module_fn_ = getattr(module, function_name)
        return module_fn_()
    return hk.without_apply_rng(hk.transform(module_fn))

def update_params(params, updates):
    params_mutable = hk.data_structures.to_mutable_dict(params)
    for key, update in updates.items():
        params_mutable[key] += update
    return hk.data_structures.to_immutable_dict(params_mutable)
