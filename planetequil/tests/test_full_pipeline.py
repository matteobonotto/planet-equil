import os
import shutil
import pytest

from planetequil.config import Config
from planetequil.train import main_train
from planetequil import PlaNet
from planetequil.utils import dummy_planet_input


@pytest.mark.slow
def test_full_pipe():
    config = Config()
    config.epochs = 2
    config.log_to_wandb = False
    config.dataset_path = "planet/tests/data/iter_like_data_sample.h5"
    print(config)

    if os.path.exists(config.save_path):
        shutil.rmtree(config.save_path)

    ### train and save model
    main_train(config=config)

    ### load pretrained model
    pipe = PlaNet.from_pretrained(config.save_path)

    ### perform inference
    measures, rr, zz = dummy_planet_input()
    flux = pipe(measures, rr, zz)
    gs_ope = pipe.compute_gs_operator(flux, rr, zz)

    assert flux.shape == (measures.shape[0], *rr.shape[1:])
    assert gs_ope.shape == (measures.shape[0], *rr[0, 1:-1, 1:-1].shape)

    ### remove artifacts
    if os.path.exists(config.save_path):
        shutil.rmtree(config.save_path)
