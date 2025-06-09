from __future__ import annotations
from sklearn.preprocessing import StandardScaler
from typing import Tuple, TypeAlias, List, Optional, Dict
import torch
from torch import Tensor, nn
import json
from pathlib import Path
import pickle
import numpy as np
from numpy.typing import NDArray
from functools import singledispatchmethod
from safetensors import safe_open

from scipy import signal

from .models.models import MODELS
from .config import PlaNetConfig
from .data import Scaler, compute_Grad_Shafranov_kernels
from .loss import Gauss_kernel_5x5, _compute_grad_shafranov_operator
from .types import _TypeNpFloat
from .constants import DTYPE
from .utils import get_device


def load_model_safetensors(
    model: nn.Module, path: str | Path, device: torch.device | str
) -> None:
    """Given a model, loads the layers from safetensors"""
    device_ = torch.device(device)
    if not isinstance(path, Path):
        path = Path(path)
    try:
        state_dict: Dict[str, Tensor] = {}
        with safe_open(path, framework="pt", device=str(device_)) as f:  # type: ignore[no-untyped-call]
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"load_model_safetensors failed with error:")
        print(e)


def validate_device(model: nn.Module, device: torch.device) -> None:
    """Checks if all the parameters are on the correct device"""
    need_fix = False
    for name, param in model.named_parameters():
        if param.device != device:
            need_fix = True
            break
    if need_fix:
        model.to(device)


# def jit_compile_model(
#     model: nn.Module, device: torch.device, dtype: torch.dtype
# ) -> nn.Module:
#     dummy_inputs = dummy_planet_input_tensor(
#         batch_size=32,
#         nr=model.config.nr,
#         nz=model.config.nz,
#         n_measures=model.config.n_measures,
#         device=device,
#     )
#     traced_model = torch.jit.trace(model, example_kwarg_inputs={"x": dummy_inputs})
#     traced_model.config = model.config
#     return traced_model


class PlaNet:
    fast_inference: bool = False
    compile_model: bool = False

    def __init__(
        self, model: nn.Module, scaler: Scaler, fast_inference: Optional[bool] = None
    ):
        if fast_inference is not None:
            self.fast_inference = fast_inference
        self.model: nn.Module = model
        self.get_model_device_and_dtype()
        self.model.eval()
        self.scaler: Scaler = scaler
        scaler.to(self.device)
        self.Gauss_kernel = torch.tensor(
            Gauss_kernel_5x5, device=self.device, dtype=self.dtype
        )

    def to(self, device: torch.device) -> None:
        self.device = device
        self.scaler.to(device)
        self.model.to(device)

    def get_model_device_and_dtype(self) -> None:
        _, param = next(iter(self.model.named_parameters()))
        self.device = param.device
        self.dtype = param.dtype

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        device: Optional[str] = None,
        fast_inference: Optional[bool] = None,
    ) -> PlaNet:
        print(f"Loading model from {path}")
        model_path = Path(path)
        device_ = get_device() if device is None else torch.device(device)

        # load scaler (already fitted during training)
        scaler = Scaler.from_config(f"{path}/scaler.json")
        scaler.to(device_)

        # load the core planet model
        config = PlaNetConfig(**json.load(open(model_path / Path("config.json"), "r")))
        model = MODELS[config.model_name](**config.to_dict())
        load_model_safetensors(
            model=model, path=model_path / Path("model.safetensors"), device=device_
        )
        validate_device(model=model, device=device_)
        return cls(
            model=model.to(device_), scaler=scaler, fast_inference=fast_inference
        )

    @staticmethod
    def _np_to_tensor(
        inputs_np: List[_TypeNpFloat], device: torch.device, dtype: torch.dtype
    ) -> List[Tensor]:
        # return list(map(lambda x: torch.tensor(x, device=device, dtype=dtype), inputs_np))
        return [torch.tensor(x, device=device, dtype=dtype) for x in inputs_np]
        # return [Tensor(x).to(device).to(dtype) for x in inputs_np]

    def __call__(
        self,
        measures: _TypeNpFloat | Tensor,
        rr: _TypeNpFloat | Tensor,
        zz: _TypeNpFloat | Tensor,
    ) -> _TypeNpFloat | Tensor:
        if isinstance(measures, np.ndarray):
            message = "all inputs must be of type np.ndarray"
            assert isinstance(rr, np.ndarray) and isinstance(zz, np.ndarray), message
            return self._call_numpy(measures, rr, zz)
        else:
            message = "all inputs must be of type torch.Tensor"
            assert isinstance(rr, Tensor) and isinstance(zz, Tensor), message
            return self._call_tensor(measures, rr, zz)

    def _call_tensor(
        self,
        measures: Tensor,
        rr: Tensor,
        zz: Tensor,
    ) -> Tensor:
        if measures.ndim == 1:
            measures = measures[None, :]
        if rr.ndim == 2:
            rr = torch.tile(rr[None, ...], (measures.shape[0], 1, 1))
        if zz.ndim == 2:
            zz = torch.tile(zz[None, ...], (measures.shape[0], 1, 1))

        scaled_inputs = self.scaler.transform(measures, inplace=self.fast_inference)
        with torch.inference_mode():
            flux = self.model((scaled_inputs, rr, zz))

        return flux

    def _call_numpy(
        self,
        measures: _TypeNpFloat,
        rr: _TypeNpFloat,
        zz: _TypeNpFloat,
    ) -> _TypeNpFloat:
        if measures.ndim == 1:
            measures = measures[None, :]
        if rr.ndim == 2:
            rr = np.tile(rr[None, ...], (measures.shape[0], 1, 1))
        if zz.ndim == 2:
            zz = np.tile(zz[None, ...], (measures.shape[0], 1, 1))

        # prepare the inputs [simulating batch size of 1]
        scaled_inputs = self.scaler.transform(measures, inplace=self.fast_inference)

        message = f"scaled_inputs must be of type np.ndarray, got {type(scaled_inputs)}"
        assert isinstance(scaled_inputs, np.ndarray), message

        # perfrom the forward pass
        inputs = self._np_to_tensor(
            [scaled_inputs, rr, zz],
            device=self.device,
            dtype=self.dtype,
        )
        with torch.inference_mode():
            flux = self.model(inputs)

        # go back to np array (with the correct dtype and device)
        if self.device != torch.device("cpu"):
            flux = flux.cpu()

        return flux.numpy().astype(measures.dtype)

    def compute_gs_operator(
        self,
        flux: _TypeNpFloat | Tensor,
        rr: _TypeNpFloat | Tensor,
        zz: _TypeNpFloat | Tensor,
    ) -> _TypeNpFloat | Tensor:
        if isinstance(flux, np.ndarray):
            message = "all inputs must be of type np.ndarray"
            assert isinstance(rr, np.ndarray) and isinstance(zz, np.ndarray), message
            return self.compute_gs_ope_numpy(flux, rr, zz)
        else:
            message = "all inputs must be of type torch.Tensor"
            assert isinstance(rr, Tensor) and isinstance(zz, Tensor), message
            return self.compute_gs_ope_tensor(flux, rr, zz)

    def compute_gs_ope_tensor(
        self,
        flux: Tensor,
        rr: Tensor,
        zz: Tensor,
    ) -> Tensor:
        assert (
            flux.ndim == 3
        ), f"For torch tensors, planet is compatible only with batched input. Expected 'flux.ndim=3', got {flux.ndim}"
        assert (
            rr.ndim == 3
        ), f"For torch tensors, planet is compatible only with batched input. Expected 'rr.ndim=3', got {rr.ndim}"
        assert (
            zz.ndim == 3
        ), f"For torch tensors, planet is compatible only with batched input. Expected 'zz.ndim=3', got {zz.ndim}"

        n_batch = flux.shape[0]
        L_ker = np.zeros(shape=(n_batch, 3, 3))
        Df_dr_ker = np.zeros(shape=(n_batch, 3, 3))
        for i_batch in range(rr.shape[0]):
            L_ker_batch, Df_dr_ker_batch = compute_Grad_Shafranov_kernels(
                rr[i_batch, ...], zz[i_batch, ...]
            )
            L_ker[i_batch, ...] = L_ker_batch
            Df_dr_ker[i_batch, ...] = Df_dr_ker_batch
        gs_ope = _compute_grad_shafranov_operator(
            flux,
            torch.tensor(L_ker, dtype=self.dtype, device=self.device),
            torch.tensor(Df_dr_ker, dtype=self.dtype, device=self.device),
            rr,
            zz,
            self.Gauss_kernel,
        )
        return gs_ope

    def compute_gs_ope_numpy(
        self,
        flux: _TypeNpFloat,
        rr: _TypeNpFloat,
        zz: _TypeNpFloat,
    ) -> _TypeNpFloat:
        squeeze_output = False
        if flux.ndim == 2:
            squeeze_output = True
            flux = flux[None, :]
        if rr.ndim == 2:
            rr = np.tile(rr[None, ...], (flux.shape[0], 1, 1))
        if zz.ndim == 2:
            zz = np.tile(zz[None, ...], (flux.shape[0], 1, 1))
        # use this one also if we have a batch of 1 input
        if squeeze_output:
            return self._compute_gs_ope_numpy_batch(flux, rr, zz).squeeze()
        else:
            return self._compute_gs_ope_numpy_batch(flux, rr, zz)

    def _compute_gs_ope_numpy(
        self, flux: _TypeNpFloat, rr: _TypeNpFloat, zz: _TypeNpFloat
    ) -> _TypeNpFloat:
        L_ker, Df_dr_ker = compute_Grad_Shafranov_kernels(rr, zz)
        hr = rr[1, 2] - rr[1, 1]
        hz = zz[2, 1] - zz[1, 1]
        Lpsi = signal.convolve2d(flux, L_ker, mode="valid")
        Dpsi_dr = signal.convolve2d(flux, Df_dr_ker, mode="valid")
        lhs_scipy = Lpsi - Dpsi_dr / rr[1:-1, 1:-1]
        alfa = -2 * (hr**2 + hz**2)
        beta = alfa / (hr**2 * hz**2)
        return signal.convolve(lhs_scipy * beta, Gauss_kernel_5x5, mode="same")

    def _compute_gs_ope_numpy_batch(
        self, flux: _TypeNpFloat, rr: _TypeNpFloat, zz: _TypeNpFloat
    ) -> _TypeNpFloat:
        gs_ope = np.zeros_like(flux[:, 1:-1, 1:-1])
        for i_batch in range(rr.shape[0]):
            gs_ope[i_batch, ...] = self._compute_gs_ope_numpy(
                flux[i_batch, ...], rr[i_batch, ...], zz[i_batch, ...]
            )
        return gs_ope
