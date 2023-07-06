from rl_lap.nets.hk_utils import generate_hk_module_fn
from rl_lap.nets.triangular_eqx import LowerTriangularParameterMatrix as LTriangular_eqx
from rl_lap.nets.triangular_hk import LowerTriangularParameterMatrix as LTriangular_kh
from rl_lap.nets.mlp_eqx import MLP as MLPeqx, DualCoefficientExtendedMLP as DualMLPeqx
from rl_lap.nets.mlp_flax import MLP as MLPflax
from rl_lap.nets.mlp_hk import MLP as MLPhk