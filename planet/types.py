from typing import TypeAlias, Any, List, Tuple
from torch import Tensor
import numpy as np
from numpy.typing import NDArray


_TypeNpFloat64: TypeAlias = NDArray[np.float64]
_TypeNpFloat32: TypeAlias = NDArray[np.float32]
_TypeNpFloat: TypeAlias = NDArray[np.floating]


_TypeBatch: TypeAlias = List[
    Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
]
