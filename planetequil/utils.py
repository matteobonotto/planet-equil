from argparse import ArgumentParser, Namespace
from typing import Optional, Dict, Any, Tuple
import yaml  # type: ignore
from pathlib import Path
import re
import numpy as np
import torch
from torch import nn, Tensor
import pickle
from lightning import Trainer
import json
import h5py
from sklearn.preprocessing import StandardScaler
from safetensors.torch import save_file
from .config import Config
from .constants import DTYPE
from .types import _TypeNpFloat


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def tensor_to_numpy(t: Tensor) -> _TypeNpFloat:
    if t.device == torch.device("cpu"):
        return t.numpy().astype(np.float64)
    else:
        return t.cpu().numpy().astype(np.float64)


def dummy_planet_input_np(
    batch_size: int = 32,
    nr: int = 64,
    nz: int = 64,
    n_measures: int = 302,
) -> Tuple[_TypeNpFloat, _TypeNpFloat, _TypeNpFloat]:
    return (
        np.random.normal(size=(batch_size, n_measures)),
        np.random.normal(size=(batch_size, nr, nz)),
        np.random.normal(size=(batch_size, nr, nz)),
    )


def dummy_planet_input_tensor(
    batch_size: int = 32,
    nr: int = 64,
    nz: int = 64,
    n_measures: int = 302,
    device: Optional[str | torch.device] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    if device is None:
        device_ = get_device()
    else:
        device_ = torch.device(device)
    return (
        torch.rand(size=(batch_size, n_measures), device=device_, dtype=DTYPE),
        torch.rand(size=(batch_size, nr, nz), device=device_, dtype=DTYPE),
        torch.rand(size=(batch_size, nr, nz), device=device_, dtype=DTYPE),
    )


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("config", help="path to config file")
    args, _ = parser.parse_known_args()
    return args


def load_config(path: str) -> Config:
    config_dict = yaml.safe_load(open(path, "r"))
    return Config.from_dict(config_dict=config_dict)


def last_ckp_path(ckpt_path: str | Path) -> Path:
    if isinstance(ckpt_path, str):
        ckpt_path = Path(ckpt_path)
    # for ckp in Path(ckpt_path).iterdir():
    # Regex to extract epoch and step
    pattern = re.compile(r"epoch=(\d+)-step=(\d+)")

    # Extract (epoch, step) tuples + path
    parsed = []
    for path in ckpt_path.iterdir():
        match = pattern.search(path.name)
        if match:
            epoch, step = map(int, match.groups())
            parsed.append(((epoch, step), path))

    # Find the path with max (epoch, step)
    _, latest = max(parsed, key=lambda x: (x[0][0], x[0][1]))
    return latest


def write_h5(
    data: Dict[str, Any],
    filename: str,
    dtype: str = "float64",
    # compression : str = 'lzf',
    # compression_opts : int = 1,
    # verbose : bool = False,
) -> None:

    compression: str = "lzf"
    # compression: int = 1  # -> gzip compression level

    kwargs = {
        "dtype": dtype,
        "compression": compression,
    }

    # t_start = time.time()
    with h5py.File(filename + ".h5", "w") as hf:
        for key, item in data.items():
            hf.create_dataset(key, data=item, shape=item.shape, **kwargs)
    hf.close()


def read_h5_numpy(
    filename: str,
) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    with h5py.File(filename, "r") as hf:
        for key, item in hf.items():
            data.update({key: item[()]})
    return data
