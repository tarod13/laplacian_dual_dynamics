from typing import Callable
import haiku as hk
import jax
import numpy as np

def generate_hk_module_fn(module_class: hk.Module, *args, **kwargs) -> Callable:
    def module_fn(x: np.ndarray) -> jax.Array:
        module = module_class(*args, **kwargs)
        return module(x.astype(np.float32))
    return hk.without_apply_rng(hk.transform(module_fn))