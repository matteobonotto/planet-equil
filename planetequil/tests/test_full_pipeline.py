import os
import shutil
import pytest
import torch
from copy import deepcopy
import numpy as np

from planetequil import PlaNet
from planetequil.constants import DTYPE
from planetequil.config import Config
from planetequil.train import main_train
from planetequil.utils import (
    dummy_planet_input_tensor,
    dummy_planet_input_np,
    get_device,
    read_h5_numpy,
)
from planetequil.plot import contourf


# @pytest.mark.slow
def test_full_pipe():
    config = Config()
    config.epochs = 1
    config.log_to_wandb = False
    config.dataset_path = "planetequil/tests/data/planet_sample_dataset.h5"
    print(config)

    if os.path.exists(config.save_path):
        shutil.rmtree(config.save_path)

    ### train and save model
    main_train(config=config)

    ### load pretrained model
    pipe = PlaNet.from_pretrained(config.save_path)

    ### perform inference (numpy)
    measures, rr, zz = dummy_planet_input_np()
    flux = pipe(measures, rr, zz)
    gs_ope = pipe.compute_gs_operator(flux, rr, zz)

    assert flux.shape == (measures.shape[0], *rr.shape[1:])
    assert gs_ope.shape == (measures.shape[0], *rr[0, 1:-1, 1:-1].shape)

    ### remove artifacts
    if os.path.exists(config.save_path):
        shutil.rmtree(config.save_path)


def test_pipeline_device():
    planet = PlaNet.from_pretrained("planetequil/tests/data/planet_slim_mlp")
    print(f"Model loaded on {planet.device} device")

    ### device: cpu
    device = torch.device("cpu")
    planet.to(device)
    inputs = dummy_planet_input_tensor(device=device)
    out = planet(*inputs)

    ### device: gpu (skip if gpu not available)
    if get_device() != torch.device("cpu"):
        device = get_device()
        planet.to(device)
        inputs = dummy_planet_input_tensor(device=device)
        out = planet(*inputs)


def test_pipeline_numpy_vs_torch():
    data = read_h5_numpy("planetequil/tests/data/planet_sample_dataset.h5")
    sample = 0
    planet = PlaNet.from_pretrained("planetequil/tests/data/planet_slim_mlp")

    ### numpy
    device = torch.device("cpu")
    planet.to(device)
    inputs = {
        "measures": deepcopy(data["measures"][sample, ...]),
        "rr": data["RR_grid"],
        "zz": data["ZZ_grid"],
    }
    flux_np = planet(**inputs)

    gs_ope_np = planet.compute_gs_operator(
        flux=flux_np,
        rr=inputs["rr"],
        zz=inputs["zz"],
    )

    ### torch
    device = torch.device("cpu")
    planet.to(device)
    inputs = {
        "measures": deepcopy(data["measures"][sample, ...]),
        "rr": data["RR_grid"],
        "zz": data["ZZ_grid"],
    }
    inputs_torch = {
        k: torch.tensor(v, device=planet.device, dtype=DTYPE) for k, v in inputs.items()
    }
    flux_torch = planet(**inputs_torch)

    gs_ope_torch = planet.compute_gs_operator(
        flux=flux_torch,
        rr=inputs_torch["rr"],
        zz=inputs_torch["zz"],
    )

    error_flux = np.linalg.norm(flux_np - flux_torch.numpy()) / np.linalg.norm(flux_np)
    error_gs_ope = np.linalg.norm(gs_ope_np - gs_ope_torch.numpy()) / np.linalg.norm(
        gs_ope_np
    )

    assert error_flux < 1e-6, f"error_flux > 1e-6 (got {error_flux})"
    assert error_gs_ope < 1e-4, f"error_gs_ope > 1e-4 (got {error_gs_ope})"
