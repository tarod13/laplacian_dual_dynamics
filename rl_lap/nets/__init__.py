from rl_lap.nets.hk_utils import generate_hk_module_fn, generate_hk_get_variables_fn
from rl_lap.nets.triangular_eqx import LowerTriangularParameterMatrix as LTriangular_eqx
from rl_lap.nets.matrices_hk import LowerTriangularParameterMatrix as LTriangular_kh, ParameterMatrix as ParameterMatrix_kh
from rl_lap.nets.mlp_eqx import MLP as MLP_eqx, DualCoefficientExtendedMLP as DualMLP_eqx
from rl_lap.nets.mlp_flax import MLP as MLP_flax
from rl_lap.nets.mlp_hk import MLP as MLP_hk, DualCoefficientExtendedMLP as DualMLP_hk