from __future__ import annotations
import random
import numpy as np
from numpy import ndarray
from functools import partial
from typing import Optional, Tuple, Any, List, Dict, List, Tuple
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, asdict
import yaml
from torchinfo import summary

from multiprocessing import cpu_count
from torch import Tensor, nn, autograd
import torch.nn.functional as F

import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import WandbLogger


from torch.utils.data import DataLoader, Subset

from planet.utils import read_h5_numpy, get_accelerator, last_ckp_path, save_model_and_scaler
from planet.constants import RANDOM_SEED


random.seed(RANDOM_SEED)
DTYPE = torch.float32


@dataclass
class PlaNetMLPConfig:
    meas_dim: int = 302
    hidden_dim: int = 128
    n_layers: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Config:
    is_physics_informed: bool = True
    dataset_path: Optional[str] = None
    batch_size: int = 64
    epochs: int = 10
    # hidden_dim: int = 128
    # nr: int = 64
    # nz: int = 64
    # branch_in_dim: int = 302
    planet: Optional[PlaNetMLPConfig] = None
    log_to_wandb: bool = False
    wandb_project: Optional[str] = None
    save_checkpoints: bool = False
    save_path: str = "tmp/model.pt"
    resume_from_checkpoint: bool = False
    num_workers: int = 4
    do_super_resolution: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> Config:

        cls_instance = cls()
        for k, v in config_dict.items():
            if k in cls_instance.__dict__.keys():
                setattr(cls_instance, k, v)

        if hasattr(cls_instance, "planet"):
            cls_instance.planet = PlaNetMLPConfig(**cls_instance.planet)

        return cls_instance


def load_config(path: str) -> Config:
    config_dict = yaml.safe_load(open(path, "r"))
    return Config.from_dict(config_dict=config_dict)


class TrainableSwish(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x: Tensor) -> Tensor:
        return swish(x, self.beta)


def swish(x: Tensor, beta: float = 1.0) -> Tensor:
    return x * F.sigmoid(beta * x)


class LineardNormAct(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.norm = nn.BatchNorm1d(num_features=out_features)
        self.act = TrainableSwish()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.linear(x)))


class MLPStack(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, n_layers: int = 3, hidden_dim: int = 128
    ):
        super().__init__()

        layers = []
        for i in range(n_layers):
            if i == 0:
                in_features, out_features = in_dim, hidden_dim
                layers.append(LineardNormAct(in_features, out_features))
            elif i == n_layers - 1:
                in_features, out_features = hidden_dim, out_dim
                layers.append(nn.Linear(in_features, out_features)) 
            else:
                in_features, out_features = hidden_dim, hidden_dim
                layers.append(LineardNormAct(in_features, out_features))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class PlaNetCoreMLP(nn.Module):
    def __init__(
        self,
        meas_dim: int = 302,
        hidden_dim: int = 128,
        n_layers: int = 3,
    ):
        super().__init__()
        self.trunk = MLPStack(in_dim=2, out_dim=hidden_dim, n_layers=n_layers)
        self.branch = MLPStack(in_dim=meas_dim, out_dim=hidden_dim, n_layers=n_layers)
        self.decoder = MLPStack(in_dim=hidden_dim, out_dim=1, n_layers=n_layers)

    def forward(self, x: Tensor) -> Tensor:
        x_meas, rz = x
        out_branch = self.branch(x_meas)
        out_trunk = self.trunk(rz)
        return self.decoder(out_branch * out_trunk)


def compute_partial_derivative(f: Tensor, var: Tensor) -> Tensor:
    d = autograd.grad(f, var, grad_outputs=torch.ones_like(f), create_graph=True)[0]
    return d


def compute_gso(pred: Tensor, rz: Tensor) -> Tensor:
    # Create coordinates with requires_grad=True
    r = rz[..., -2]
    z = rz[..., -1]

    # First-order derivatives
    # df_dr = torch.autograd.grad(pred, r, grad_outputs=torch.ones_like(pred), create_graph=True)[0]
    # df_dz = torch.autograd.grad(pred, z, grad_outputs=torch.ones_like(pred), create_graph=True)[0]

    # # Second-order (for Laplacian)
    # d2f_dr2 = torch.autograd.grad(df_dr, r, grad_outputs=torch.ones_like(df_dr), create_graph=True)[0]
    # d2f_dz2 = torch.autograd.grad(df_dz, z, grad_outputs=torch.ones_like(df_dz), create_graph=True)[0]

    df_dr = compute_partial_derivative(pred, r)
    df_dz = compute_partial_derivative(pred, z)
    d2f_dr2 = compute_partial_derivative(df_dr, r)
    d2f_dz2 = compute_partial_derivative(df_dz, z)

    gso = d2f_dr2 + d2f_dz2 - df_dr / r
    return gso


class GSOperatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred: Tensor,
        rhs: Tensor,
        rz: Tensor,
    ) -> Tensor:

        gso = compute_gso(pred=pred, rz=rz)

        return self.mse(gso, rhs)


MAP_PDELOSS: Dict[str, nn.Module] = {"grad_shafranov_operator": GSOperatorLoss}


class PlaNetLoss(nn.Module):
    log_dict: Dict[str, float] = {}

    def __init__(
        self,
        is_physics_informed: bool = True,
        scale_mse: float = 1.0,
        scale_pde: float = 0.1,  # for better stability
        pde_loss_class: str = "grad_shafranov_operator",
    ):
        super().__init__()
        self.is_physics_informed = is_physics_informed
        self.loss_mse = nn.MSELoss()
        self.loss_pde = MAP_PDELOSS[pde_loss_class]()
        self.scale_mse = scale_mse
        self.scale_pde = scale_pde

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        rz: Tensor,
        rhs: Tensor,
    ) -> Tensor:
        mse_loss = self.scale_mse * self.loss_mse(input=pred, target=target)
        self.log_dict["mse_loss"] = mse_loss.item()
        if not self.is_physics_informed:
            return mse_loss
        else:
            pde_loss = self.scale_pde * self.loss_pde(
                pred=pred,
                rhs=rhs,
                rz=rz,
            )
            self.log_dict["pde_loss"] = pde_loss.item()
            return mse_loss + pde_loss


def _to_tensor(
    device: torch.device, inputs: Tuple[Any], dtype: torch.dtype
) -> Tuple[Tensor]:
    inputs_t: List[Tensor] = []
    for x in inputs:
        inputs_t.append(
            torch.tensor(
                x,
                dtype=dtype,
                # device=device,
            )
        )
    return tuple(inputs_t)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class PlaNetDataset:
    def __init__(
        self,
        path: str,
        dtype: torch.dtype = torch.float32,
        is_physics_informed: bool = True,
    ):
        self.dtype = dtype
        self.device = get_device()
        self.scaler_inputs = StandardScaler()
        self.scaler_rz = StandardScaler()
        self.is_physics_informed = is_physics_informed

        data = read_h5_numpy(path)
        n_sample, nr, nz = data["flux"].shape

        self.inputs = np.column_stack(
            (data["measures"], data["coils_current"], data["p_profile"])
        )
        self.inputs = self.scaler_inputs.fit_transform(self.inputs)

        self.map_equil_idx = (
            np.arange(n_sample)[:, None, None] * np.ones((nr, nz), dtype=int)
        ).ravel()
        self.flux = data["flux"].ravel()
        self.rhs = data["rhs"].ravel()

        self.rz = np.column_stack(
            [
                np.repeat(data["RR_grid"][None, ...], n_sample, axis=0).reshape(-1, 1),
                np.repeat(data["ZZ_grid"][None, ...], n_sample, axis=0).reshape(-1, 1),
            ]
        )
        self.rz = self.scaler_rz.fit_transform(self.rz)

    def get_scaler(self) -> StandardScaler:
        return self.scaler

    def __len__(self) -> int:
        return self.map_equil_idx.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor]:
        idx_equil = self.map_equil_idx[idx]

        inputs = self.inputs[idx_equil, ...]
        flux = self.flux[idx]
        rhs = self.rhs[idx]
        rz = self.rz[idx, :]

        return _to_tensor(
            device=self.device,
            dtype=self.dtype,
            inputs=(inputs, flux, rhs, rz),
        )


def collate_fun(batch: Tuple[Tuple[Tensor]]) -> Tuple[Tensor]:
    return (
        torch.stack([s[0] for s in batch], dim=0),  # inputs
        torch.stack([s[1] for s in batch], dim=0),  # flux
        torch.stack([s[2] for s in batch], dim=0),  # rhs
        torch.stack([s[3] for s in batch], dim=0),  # rz
    )


class DataModule(L.LightningDataModule):
    def __init__(self, config: PlaNetMLPConfig):
        super().__init__()
        self.dataset = PlaNetDataset(
            path=config.dataset_path,
            is_physics_informed=config.is_physics_informed,
        )
        self.batch_size = config.batch_size
        self.num_workers = (
            cpu_count() - 2 if config.num_workers == -1 else config.num_workers
        )
        self.split_dataset()

    def split_dataset(self, ratio: int = 0.1):
        idx = list(range(len(self.dataset)))
        idx_valid = random.sample(idx, k=int(ratio * len(idx)))
        idx_train = list(set(idx).difference(idx_valid))
        self.train_dataset = Subset(self.dataset, idx_train)
        self.val_dataset = Subset(self.dataset, idx_valid)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fun,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fun,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def setup(self, stage=None):
        pass


class LightningPlaNet(L.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.model = PlaNetCoreMLP(**config.planet.to_dict())
        self.loss_module = PlaNetLoss(is_physics_informed=config.is_physics_informed)

    def _compute_loss_batch(self, batch, batch_idx):
        inputs, flux, rhs, rz = batch
        pred = self((inputs, rz))
        loss = self.loss_module(
            pred=pred,
            target=flux,
            rhs=rhs,
            rz=rz,
        )
        return loss

    def forward(self, *args):
        return self.model(*args)

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss_batch(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        # self.logger.log_metrics({'train_'+k:v for k,v in self.loss_module.log_dict.items()})
        for k, v in self.loss_module.log_dict.items():
            self.log(f"train_{k}", v, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss_batch(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.001)


def main_train(config: Config):
    #
    save_dir = Path(config.save_path).parent
    save_dir.mkdir(exist_ok=True, parents=True)

    ### instantiate model and datamodule
    model = LightningPlaNet(config=config)
    datamodule = DataModule(config=config)

    ### define some callbacks
    callbacks: List[Callback] = []
    if config.save_checkpoints is not None:
        callbacks.append(
            ModelCheckpoint(
                dirpath=save_dir / Path("ckp"), save_top_k=2, monitor="val_loss"
            )
        )

    # get the logger
    wandb_logger: Optional[WandbLogger] = None
    if config.log_to_wandb:
        wandb_logger = WandbLogger(project=config.wandb_project)

    ### train the model
    trainer = Trainer(
        max_epochs=config.epochs,
        accelerator=get_accelerator(),
        devices="auto",
        callbacks=callbacks,
        logger=wandb_logger,
    )
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=(
            last_ckp_path(save_dir / Path("ckp"))
            if config.resume_from_checkpoint
            else None
        ),
    )

    ### save model + scaler for inference
    save_model_and_scaler(trainer, datamodule.dataset.scaler, config)


if __name__ == "__main__":

    # load config
    # args = parse_arguments()
    cfg_path = "config/config_mlp.yml"
    config = load_config(cfg_path)
    main_train(config=config)
