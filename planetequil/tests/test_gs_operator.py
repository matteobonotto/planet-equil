import os
import sys

sys.path.append(os.getcwd())

import torch
from scipy import signal
import numpy as np

from planetequil.loss import _compute_grad_shafranov_operator, Gauss_kernel_5x5
from planetequil.train import DataModule
from planetequil.config import PlaNetConfig
from planetequil.utils import load_config


def test_gs_operator():
    ###
    datamodule = DataModule(load_config("planetequil/tests/data/config.yml"))
    dataloader = datamodule.train_dataloader()
    meas, flux, RHS_in, RR, ZZ, Laplace_kernel, Df_dr_kernel = next(iter(dataloader))

    ###
    gauss_kernel = torch.tensor(Gauss_kernel_5x5, dtype=torch.float32)
    rhs_computed = _compute_grad_shafranov_operator(
        flux, Laplace_kernel, Df_dr_kernel, RR, ZZ, gauss_kernel
    )

    (RHS_in - rhs_computed).norm(dim=0).shape
    diff = RHS_in - rhs_computed  # shape [batch, 32, 32]

    norm_difference = torch.norm(diff.view(diff.shape[0], -1), dim=1)  # shape [batch]
    norm_rhs = torch.norm(RHS_in.view(RHS_in.shape[0], -1), dim=1)  # shape [batch]
    norm = 100 * norm_difference / norm_rhs

    assert (
        norm < 5
    ).all(), (
        "error with _compute_grad_shafranov_operator in at least one element is > 5%"
    )
