from typing import Dict, Type
from torch import nn

from .planet import PlaNetCore
from .planet_slim import PlaNetCoreSlimMLP, PlaNetCoreSlim


MODELS: Dict[str, Type[nn.Module]] = {
    "planet": PlaNetCore,
    "planet_slim": PlaNetCoreSlim,
    "planet_slim_mlp": PlaNetCoreSlimMLP,
}
