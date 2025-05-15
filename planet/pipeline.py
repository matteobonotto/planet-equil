from __future__ import annotations
from sklearn.preprocessing import StandardScaler
from typing import Tuple, TypeAlias, List
import torch
from torch import Tensor
import json
from pathlib import Path
import pickle
import numpy as np
from numpy.typing import NDArray

from scipy import signal

from .model import PlaNetCore
from .config import PlaNetConfig
from .data import compute_Grda_Shafranov_kernels
from .loss import Gauss_kernel_5x5
from .types import _TypeNpFloat


class PlaNet:
    def __init__(self, model: PlaNetCore, scaler: StandardScaler):
        self.model: PlaNetCore = model
        self.set_device_and_dtype()
        self.model.eval()
        self.scaler: StandardScaler = scaler

    def set_device_and_dtype(self) -> None:
        _, param = next(iter(self.model.named_parameters()))
        self.device = param.device
        self.dtype = param.dtype

    @classmethod
    def from_pretrained(cls, path: str) -> PlaNet:
        model_path = Path(path)

        # load model config
        config = PlaNetConfig(**json.load(open(model_path / Path("config.json"), "r")))

        # load scaler (already fitted during training)
        scaler = pickle.load(open(model_path / Path("scaler.pkl"), "rb"))

        # load the core planet model
        model = PlaNetCore(**config.to_dict())
        model.load_state_dict(torch.load(model_path / Path("model.pt")))
        return cls(model, scaler)

    def _np_to_tensor(
        self, inputs_np: List[_TypeNpFloat], device: torch.device, dtype: torch.dtype
    ) -> List[Tensor]:
        return [Tensor(x).to(device).to(dtype) for x in inputs_np]

    def __call__(
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
        scaled_inputs = self.scaler.transform(measures)

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
        self, flux: _TypeNpFloat, rr: _TypeNpFloat, zz: _TypeNpFloat
    ) -> _TypeNpFloat:
        L_ker, Df_dr_ker = compute_Grda_Shafranov_kernels(rr, zz)
        hr = rr[1, 2] - rr[1, 1]
        hz = zz[2, 1] - zz[1, 1]
        Lpsi = signal.convolve2d(flux, L_ker, mode="valid")
        Dpsi_dr = signal.convolve2d(flux, Df_dr_ker, mode="valid")
        lhs_scipy = Lpsi - Dpsi_dr / rr[1:-1, 1:-1]
        alfa = -2 * (hr**2 + hz**2)
        beta = alfa / (hr**2 * hz**2)
        gs_ope_smooth = signal.convolve(lhs_scipy * beta, Gauss_kernel_5x5, mode="same")
        return gs_ope_smooth
