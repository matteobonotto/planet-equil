# PlaNet: plasma equilibrium reconstruction using physics-informed neural operator.
This is the official repository if the `planet-equil` package. It is a PyTorch implementation of PlaNet (PLAsma equilibrium reconstruction NETwork), a convolutional physics-informed neural operator for performing plasma equilibrium reconstruction using magnetic and non-magnetic measurements.

For any kind of reference on the model architecture or the mathematical formulation, see [this paper](https://www.sciencedirect.com/science/article/abs/pii/S0920379624000474). The original work was developed using TensorFlow, as mentioned in the paper. This is a polished and optimized version, using modern PyTorch and PyTorch Lightning. 

# Installation
The package can be installing with pip by running:
```shell
pip install planet-equil
```

## Dev installation
Alternatively, if you want to develop your own stuff you can clone the repo and then install the package and all the dependencies using `poetry`
```shell
git clone git@github.com:matteobonotto/planet-equil.git
cd planet-equil
pip3 install poetry==1.8.3
poetry config virtualenvs.create false
poetry install
```

## Models
This repository contains 3 different models:
- `PlaNet` (see `PlaNetCore` in `planetequil/models/planet.py`), the original version of the model as described in [the paper](https://www.sciencedirect.com/science/article/abs/pii/S0920379624000474).
- `PlaNetSlim` (see `PlaNetCoreSlim` in `planetequil/models/planet_slim.py`), a slim and faster version of `PlaNet`, where the `TrunkNet` is a 1D convolution encoder (instead of a 2D like `PlaNet`). This results on a faster model if compared to the original `PlaNet`, but with the same reconstruction accuracy.
- `PlaNetSlimMLP` (see `PlaNetCoreSlimMLP` in `planetequil/models/planet_slim.py`). It is a further modification to the `PlaNetSlim` models: here the decoder is a simple MLP instead if a 2D transposed convolution one. This results in a faster model, at the cost of a minimal loss of reconstruction accuracy.

This models can be found in in the folder `model_hub/`, and have already been pretrained using [this dataset](https://github.com/matteobonotto/ITERlike-equilibrium-dataset). There is also a version of the original `PlaNet` trained without imposing the physics informed constraint in the loss function (see `model_hub/planet_no_physics_informed/`).

## Data
To train and test PlaNet you can use the dataset available at [this repo](https://github.com/matteobonotto/ITERlike_equilibrium_dataset.git), containing ~85k equilibria of an ITER-like devices. All the equilibria have been computed numerically using the free-boundary Grad-Shafranov solver FRIDA, which is publicly available [here](https://github.com/matteobonotto/frida).

The data can be downloaded and stored in a PlaNet-compatible format (see `1_dataset_creation.ipynb` for further info about the format) using the `planetequil/scripts/create_dataset.py` script. To download a sample dataset (~8.2K equilibria), run:
```shell
python planetequil/scripts/create_dataset.py
```
to download the full dataset, run:
```shell
python planetequil/scripts/create_dataset.py --full
```

## Tutorials
There are tutorial notebooks available to get started with `planet`:
- [1_dataset_creation.ipynb](notebooks/1_dataset_creation.ipynb): show how to create and format your data to perform trainign and inference using the PlaNet model.
- [2_model_training.ipynb](notebooks/2_model_training.ipynb): shows how to perfrom a full training of the PlaNet model using PyTorch Lightning`.
- [3_load_pretrained_and_prediction.ipynb](notebooks/3_load_pretrained_and_prediction.ipynb): shows how to load a pretrained model and how to use if to perform reconstruction of plasma equilibrium and to estimate the Grad-Shafranov operator.
- [4_compare_models.ipynb](notebooks/4_compare_models.ipynb): shows how to load the available models in the folder `model_hub/`, already pretrained using [this dataset](https://github.com/matteobonotto/ITERlike-equilibrium-dataset), and check their precision in terms of reconstructing the flux and the Grad-Shafranov operators.


### Citations
If you find the repository useful please consider citing [this paper](https://doi.org/10.1016/j.fusengdes.2024.114193).
















