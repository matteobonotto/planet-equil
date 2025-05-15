# PlaNet: plasma equilibrium reconstruction using physics-informed neural operator.
This is the official repository if the `planet` package. It is a PyTorch implementation of PlaNet (PLAsma equilibrium reconstruction NETwork), a convolutional physics-informed neural operator for performing plasma equilibrium reconstruction using magnetic and non-magnetic measurements.

For any kind of reference on the model architecture or the mathematical formulation, see [this paper](https://www.sciencedirect.com/science/article/abs/pii/S0920379624000474). The original work was developed using TensorFlow, as mentioned in the paper. This is a polished and optimized version, using modern PyTorch and PyTorch Lightning. 

# Installation
First, create a virtual environment using `venv`
```shell
python3.10 -m venv venv 
source venv/bin/activate
```
then install the package and all the dependencies using `poetry`
```shell
pip3 install poetry==1.8.3
poetry config virtualenvs.create false
poetry install
```

## Data
To train and test PlaNet you can use the dataset available at [this repo](https://github.com/matteobonotto/ITERlike_equilibrium_dataset.git), containing ~85k equilibria of an ITER-like devices. All the equilibria have been computed numerically using the free-boundary Grad-Shafranov solver FRIDA, which is publicly available [here](https://github.com/matteobonotto/frida).


## Tutorials
There are tutorial notebooks available to get started with `planet`:
- [1_dataset_creation.ipynb](notebooks/1_dataset_creation.ipynb): show how to create and format your data to perform trainign and inference using the PlaNet model.
- [2_model_training.ipynb](notebooks/2_model_training.ipynb): shows how to perfrom a full training of the PlaNet model using PyTorch Lightning`.
- [3_load_pretrained_and_prediction.ipynb](notebooks/3_load_pretrained_and_prediction.ipynb): shows how to load a pretrained model and how to use if to perform reconstruction of plasma equilibrium and to estimate the Grad-Shafranov operator.

















