from typing import List, Tuple, Any, Optional, Dict, Type
import random
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from multiprocessing import cpu_count

from torch import Tensor, nn
from torchinfo import summary
from torch.optim.lr_scheduler import (
    ExponentialLR,
    LinearLR,
    ReduceLROnPlateau,
    LRScheduler,
)

import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, Callback, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger, Logger

from .config import Config
from .models.models import MODELS
from .loss import PlaNetLoss
from .data import PlaNetDataset, get_device
from .utils import (
    get_accelerator,
    last_ckp_path,
    save_model_and_scaler,
    dummy_planet_input,
)
from .types import _TypeBatch
from .constants import DTYPE


def collate_fun(batch: _TypeBatch) -> List[Tensor]:
    return [
        torch.stack([s[0] for s in batch], dim=0),  # measures
        torch.stack([s[1] for s in batch], dim=0),  # flux
        torch.stack([s[2] for s in batch], dim=0),  # rhs
        torch.stack([s[3] for s in batch], dim=0),  # RR
        torch.stack([s[4] for s in batch], dim=0),  # ZZ
        torch.stack([s[5] for s in batch], dim=0),  # L_ker
        torch.stack([s[6] for s in batch], dim=0),  # Dr_ker
    ]


class DataModule(L.LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        assert config.planet is not None, "must provide valid config.planet, got None"
        self.dataset = PlaNetDataset(
            path=config.dataset_path,
            is_physics_informed=config.is_physics_informed,
            nr=config.planet.nr,
            nz=config.planet.nz,
            do_super_resolution=config.do_super_resolution,
        )
        self.batch_size = config.batch_size
        self.num_workers = (
            cpu_count() - 2 if config.num_workers == -1 else config.num_workers
        )
        self.split_dataset()

    def split_dataset(self, ratio: float = 0.1) -> None:
        idx = list(range(len(self.dataset)))
        idx_valid = random.sample(idx, k=int(ratio * len(idx)))
        idx_train = list(set(idx).difference(idx_valid))
        self.train_dataset = Subset(self.dataset, idx_train)
        self.val_dataset = Subset(self.dataset, idx_valid)

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fun,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fun,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=True,
        )

    def setup(self, stage: Optional[Any] = None) -> None:
        pass


def get_scheduler(optimizer: torch.optim.Optimizer, trainer: Trainer) -> LRScheduler:
    scheduler = ExponentialLR(optimizer=optimizer, gamma=1 - 1.75e-3)
    # scheduler = LinearLR(
    #         optimizer,
    #         start_factor=1.0,
    #         end_factor=0.02,
    #         total_iters=(
    #             5 if trainer.max_epochs is None else trainer.max_epochs
    #         ),
    #     )
    # scheduler = ReduceLROnPlateau(optimizer)
    return scheduler


class LightningPlaNet(L.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        assert config.planet is not None, "must provide valid config.planet, got None"
        self.model = MODELS[config.model_name](**config.planet.to_dict())
        self.summary()
        self.loss_module = PlaNetLoss(is_physics_informed=config.is_physics_informed)

    def summary(self) -> None:
        input_data = tuple(
            [
                torch.tensor(x, device=self.device, dtype=DTYPE)
                for x in dummy_planet_input()
            ]
        )
        summary(self.model, input_data={"x": input_data})

    def forward(self, *args: Any) -> Tensor:
        return self.model(*args)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss = self._compute_loss_batch(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        # self.logger.log_metrics({'train_'+k:v for k,v in self.loss_module.log_dict.items()})
        for k, v in self.loss_module.log_dict.items():
            self.log(f"train_{k}", v, prog_bar=False)
        return loss

    def _compute_loss_batch(self, batch: Tensor, batch_idx: int) -> Tensor:
        measures, flux, rhs, RR, ZZ, L_ker, Df_ker = batch
        pred = self((measures, RR, ZZ))
        loss = self.loss_module(
            pred=pred,
            target=flux,
            rhs=rhs,
            Laplace_kernel=L_ker,
            Df_dr_kernel=Df_ker,
            RR=RR,
            ZZ=ZZ,
        )
        return loss

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        self._log_lr()

    def _log_lr(self) -> None:
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]["lr"]  # type: ignore
        self.log("lr", lr, on_step=True, on_epoch=False)

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss = self._compute_loss_batch(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = get_scheduler(optimizer, self.trainer)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


def get_best_model(config: Config) -> nn.Module:
    best_ckpt_path = last_ckp_path(config.save_path / Path("ckp"))
    print(f"Loading best model from checkpoint: {best_ckpt_path}")
    lightning_planet = LightningPlaNet.load_from_checkpoint(
        best_ckpt_path, config=config
    )
    model = lightning_planet.model
    model.to(torch.device("cpu"))
    return model


def main_train(config: Config) -> None:
    #
    save_dir = Path(config.save_path)
    save_dir.mkdir(exist_ok=True, parents=True)

    ### instantiate model and datamodule
    model = LightningPlaNet(config=config)
    datamodule = DataModule(config=config)

    ### define some callbacks
    callbacks: List[Callback] = [TQDMProgressBar(refresh_rate=10)]
    if config.save_checkpoints is not None:
        callbacks.append(
            ModelCheckpoint(
                dirpath=save_dir / Path("ckp"), save_top_k=2, monitor="val_loss"
            )
        )

    # get the logger
    logger: Optional[Logger] = None
    if config.log_to_wandb:
        logger = WandbLogger(project=config.wandb_project)

    ### train the model
    trainer = Trainer(
        max_epochs=config.epochs,
        accelerator=get_accelerator(),
        devices="auto",
        callbacks=callbacks,
        logger=logger,
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
    try:
        # get the model from the best saved checkpoint
        planet_model = get_best_model(config=config)
    except:
        # is something fails, get the model from the trainer
        planet_model = trainer.model.model.to(torch.device("cpu"))  # type: ignore
    save_model_and_scaler(planet_model, datamodule.dataset.scaler, config)
