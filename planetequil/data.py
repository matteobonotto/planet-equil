from __future__ import annotations
import random
import numpy as np
from pathlib import Path
from functools import partial
import json
from typing import Optional, Tuple, Any, List
import torch
from torch.utils.data import Dataset
from torch import Tensor
from sklearn.preprocessing import StandardScaler

import h5py
from scipy.interpolate import RegularGridInterpolator

from .utils import read_h5_numpy, get_device, tensor_to_numpy
from .types import _TypeNpFloat
from .constants import RANDOM_SEED, DTYPE


random.seed(RANDOM_SEED)


def sample_random_subgrids(
    RR_min: float,
    RR_max: float,
    ZZ_min: float,
    ZZ_max: float,
    nr: int,
    nz: int,
    seed: Optional[int],
) -> Tuple[_TypeNpFloat, _TypeNpFloat]:
    delta_r_min = (RR_max - RR_min) / 3
    delta_r_max = RR_max - RR_min

    delta_z_min = (ZZ_max - ZZ_min) / 6
    delta_z_max = ZZ_max - ZZ_min

    if seed is not None:
        np.random.seed(seed)
    delta_r = np.random.uniform(delta_r_min, delta_r_max, 1)
    r0 = np.random.uniform(RR_min, RR_min + delta_r_max - delta_r, 1)

    delta_z = np.random.uniform(delta_z_min, delta_z_max, 1)
    z0 = np.random.uniform(ZZ_min, ZZ_min + delta_z_max - delta_z, 1)

    rr = np.linspace(r0, r0 + delta_r, nr)
    zz = np.linspace(z0, z0 + delta_z, nz)

    rr_grid, zz_grid = np.meshgrid(rr, zz, indexing="xy")

    return (rr_grid, zz_grid)


def get_box_from_grid(rr_grid: _TypeNpFloat, zz_grid: _TypeNpFloat) -> _TypeNpFloat:
    return np.array(
        [
            [rr_grid.min(), zz_grid.min()],
            [rr_grid.max(), zz_grid.min()],
            [rr_grid.max(), zz_grid.max()],
            [rr_grid.min(), zz_grid.max()],
            [rr_grid.min(), zz_grid.min()],
        ]
    )


def interp_fun(
    f: _TypeNpFloat,
    RR: _TypeNpFloat,
    ZZ: _TypeNpFloat,
    rr: _TypeNpFloat,
    zz: _TypeNpFloat,
) -> _TypeNpFloat:
    x_pts = RR[0, :].ravel()
    y_pts = ZZ[:, 0].ravel()
    interp_func = RegularGridInterpolator((x_pts, y_pts), f.T)
    f_int = interp_func(
        np.column_stack(
            (
                rr.reshape(-1, 1),
                zz.reshape(-1, 1),
            )
        ),
        method="quintic",
    ).reshape(rr.shape)
    return f_int


def compute_Grad_Shafranov_kernels(
    RR: _TypeNpFloat | Tensor, ZZ: _TypeNpFloat | Tensor
) -> Tuple[_TypeNpFloat | Tensor, _TypeNpFloat | Tensor]:
    return_tensor = False
    if isinstance(RR, Tensor):
        device = RR.device()
        return_tensor = True
    hr = RR[1, 2] - RR[1, 1]
    hz = ZZ[2, 1] - ZZ[1, 1]
    alfa = -2 * (hr**2 + hz**2)
    Laplace_kernel = np.array(
        [[0, hr**2 / alfa, 0], [hz**2 / alfa, 1, hz**2 / alfa], [0, hr**2 / alfa, 0]]
    )
    Df_dr_kernel = (
        np.array(([0, 0, 0], [+1, 0, -1], [0, 0, 0]))
        / (2 * hr * alfa)
        * (hr**2 * hz**2)
    )
    if return_tensor:
        return (
            torch.tensor(Laplace_kernel, device=device, dtype=DTYPE),
            torch.tensor(Df_dr_kernel, device=device, dtype=DTYPE),
        )
    else:
        return Laplace_kernel, Df_dr_kernel


def _to_tensor(
    device: torch.device, inputs: List[Any], dtype: torch.dtype
) -> List[Tensor]:
    inputs_t: List[Tensor] = []
    for x in inputs:
        inputs_t.append(
            torch.tensor(
                x,
                dtype=dtype,
                # device=device,
            )
        )
    return inputs_t


class Scaler:
    """Custom implementation of sklearn standard scaler that automatically handles
    both numpy arrays and torch tensors, and allow for inplace scaling."""

    mean: _TypeNpFloat
    std: _TypeNpFloat
    mean_tensor: Tensor
    std_tensor: Tensor
    device: torch.device = get_device()
    not_fitted: bool = True
    eps: float = 1e-7

    def __init__(
        self, mean: Optional[_TypeNpFloat] = None, std: Optional[_TypeNpFloat] = None
    ) -> None:
        if mean is not None:
            self.mean = mean
            self.std = std
            self.mean_tensor = torch.tensor(mean, dtype=DTYPE)[None, ...]
            self.std_tensor = torch.tensor(std, dtype=DTYPE)[None, ...]
            self.not_fitted = False

    def set_device(self, device: torch.device | str) -> None:
        self.device = torch.device(device) if isinstance(device, str) else device
        self.mean_tensor = self.mean_tensor.to(device)
        self.std_tensor = self.std_tensor.to(device)

    def transform(
        self, x: _TypeNpFloat | Tensor, inplace: bool = False
    ) -> _TypeNpFloat | Tensor:
        if self.not_fitted:
            raise ValueError("Scaler not fitted yet, please run .fit()")
        if isinstance(x, np.ndarray):
            return self._transform_ndarray(x, inplace)
        if isinstance(x, Tensor):
            return self._transform_tensor(x, inplace)

    def _transform_ndarray(
        self, x: _TypeNpFloat, inplace: bool = False
    ) -> _TypeNpFloat:
        if inplace:
            x -= self.mean
            x /= self.std + self.eps
        else:
            x = x - self.mean
            x = x / (self.std + self.eps)
        return x

    def _transform_tensor(self, x: Tensor, inplace: bool = False) -> Tensor:
        if inplace:
            x -= self.mean_tensor
            x /= self.std_tensor + self.eps
        else:
            x = x - self.mean_tensor
            x = x / (self.std_tensor + self.eps)
        return x

    def fit(self, x: _TypeNpFloat | Tensor) -> None:
        self.not_fitted = False
        if isinstance(x, np.ndarray):
            self.mean = x.mean(axis=0)
            self.std = x.std(axis=0)
            self.mean_tensor = torch.tensor(self.mean, device=self.device, dtype=DTYPE)
            self.std_tensor = torch.tensor(self.std, device=self.device, dtype=DTYPE)
        if isinstance(x, Tensor):
            self.mean_tensor = x.mean(0, keepdim=True)
            self.std_tensor = x.std(0, unbiased=False, keepdim=True)
            self.mean = tensor_to_numpy(self.mean_tensor)
            self.std = tensor_to_numpy(self.std_tensor)

    def fit_transform(self, x: _TypeNpFloat | Tensor) -> _TypeNpFloat | Tensor:
        self.fit(x)
        return self.transform(x)

    @classmethod
    def from_config(cls, path: str | Path) -> Scaler:
        if not isinstance(path, Path):
            path = Path(path)
        config = {k: np.array(v) for k, v in json.load(open(path, "r")).items()}
        return cls(**config)


# class Scaler_:
#     def __init__(self) -> None:
#         self.scaler = StandardScaler()

#     def fit(self, x: _TypeNpFloat) -> _TypeNpFloat:
#         self.scaler.fit(x)
#         self.scaler._mean_tensor = torch.tensor(
#             self.scaler.mean_, dtype=self.dtype, device=self.device
#         )[None, ...]
#         self.scaler._std_tensor = torch.tensor(
#             self.scaler.scaler.var_**0.5, dtype=self.dtype, device=self.device
#         )[None, ...]

#     def transform(self, x: _TypeNpFloat) -> _TypeNpFloat:
#         return self.scaler.transform(x)

#     def fit_transform(self, x: _TypeNpFloat) -> _TypeNpFloat:
#         return self.scaler.fit_transform(x)


class PlaNetDataset(Dataset):  # type: ignore[type-arg]
    def __init__(
        self,
        path: str,
        dtype: torch.dtype = torch.float32,
        is_physics_informed: bool = True,
        nr: int = 64,
        nz: int = 64,
        do_super_resolution: bool = False,
        scale_inputs: bool = True,
    ) -> None:
        self.dtype = dtype
        self.device = get_device()
        self.scaler = Scaler()
        self.is_physics_informed = is_physics_informed
        self.nr = nr
        self.nz = nz
        self.do_super_resolution = do_super_resolution

        data = read_h5_numpy(path)
        if scale_inputs:
            self.inputs = self.scaler.fit_transform(data["measures"])
        else:
            self.inputs = data["measures"]

        self.flux = data["flux"]
        self.RR = data["RR_grid"]
        self.ZZ = data["ZZ_grid"]
        self.rhs = data.get("rhs", None)
        if self.rhs is None and is_physics_informed:
            print(
                "Dataset has no 'rhs' info and is_physics_informed=True, setting is_physics_informed=False"
            )

        if self.nr != self.RR.shape[0] or self.nz != self.RR.shape[1]:
            rr = np.linspace(self.RR[0, 0], self.RR[0, -1], self.nr)
            zz = np.linspace(self.ZZ[0, 0], self.RR[-1, 0], self.nr)
            self.RR, self.ZZ = np.meshgrid(rr, zz)
            self.base_RR, self.base_ZZ = data["RR_grid"], data["ZZ_grid"]

        self.sample_random_subgrids = partial(
            sample_random_subgrids,
            RR_min=self.RR.min(),
            RR_max=self.RR.max(),
            ZZ_min=self.ZZ.min(),
            ZZ_max=self.ZZ.max(),
            nr=self.RR.shape[0],
            nz=self.RR.shape[1],
            seed=RANDOM_SEED,
        )

    def get_scaler(self) -> Scaler:
        return self.scaler

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> List[Tensor]:
        inputs = self.inputs[idx, ...]
        flux = self.flux[idx, ...]
        if self.is_physics_informed:
            rhs = self.rhs[idx, ...]
        RR = self.RR
        ZZ = self.ZZ

        if flux.shape[1] != RR.shape[0] or flux.shape[1] != RR.shape[1]:
            flux = interp_fun(
                f=flux, RR=self.base_RR, ZZ=self.base_ZZ, rr=self.RR, zz=self.ZZ
            )
            if self.is_physics_informed:
                rhs = interp_fun(
                    f=rhs, RR=self.base_RR, ZZ=self.base_ZZ, rr=self.RR, zz=self.ZZ
                )

        if random.random() > 0.5 and self.do_super_resolution:
            # interpolate on a subgrid
            rr, zz = self.sample_random_subgrids()
            flux = interp_fun(f=flux, RR=self.RR, ZZ=self.ZZ, rr=rr, zz=zz)
            if self.is_physics_informed:
                rhs = interp_fun(
                    f=rhs,
                    RR=self.RR,
                    ZZ=self.ZZ,
                    rr=rr[1:-1, 1:-1],
                    zz=zz[1:-1, 1:-1],
                )
            else:
                rhs = np.zeros_like(flux[1:-1, 1:-1])
            RR = rr
            ZZ = zz
        else:
            if self.is_physics_informed:
                rhs = rhs[1:-1, 1:-1]
            else:
                rhs = np.zeros_like(flux[1:-1, 1:-1])

        if self.is_physics_informed:
            L_ker, Df_ker = compute_Grad_Shafranov_kernels(RR=RR, ZZ=ZZ)
        else:
            L_ker, Df_ker = np.zeros((3, 3)), np.zeros((3, 3))

        return _to_tensor(
            device=self.device,
            dtype=self.dtype,
            inputs=[inputs, flux, rhs, RR, ZZ, L_ker, Df_ker],
        )
