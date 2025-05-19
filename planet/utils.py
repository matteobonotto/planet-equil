from argparse import ArgumentParser, Namespace
from typing import Optional, Dict, Any, Tuple
import yaml  # type: ignore
from pathlib import Path
import re
import numpy as np
import torch
from torch import nn
import pickle
from lightning import Trainer
import json
import h5py
from sklearn.preprocessing import StandardScaler

from .config import Config
from .constants import DTYPE
from .types import _TypeNpFloat


def dummy_planet_input(
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


def save_model_and_scaler(
    planet_model: nn.Module, scaler: StandardScaler, config: Config
) -> None:
    save_dir = Path(config.save_path)
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"Saving model and scaler to {save_dir}")

    # save model config
    json.dump(config.planet.to_dict(), open(save_dir / Path("config.json"), "w"))

    # save model
    planet_model.eval()
    torch.save(planet_model.state_dict(), save_dir / Path("model.pt"))

    # save scaler
    with open(save_dir / Path("scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)


def get_accelerator() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "auto"


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
