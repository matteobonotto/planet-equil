from typing import List, Tuple, Any, Optional, Dict
import random
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from multiprocessing import cpu_count

from torch import Tensor
from torch.optim import Optimizer

import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import LRSchedulerConfig

from .config import Config
from .model import PlaNetCore
from .loss import PlaNetLoss
from .data import PlaNetDataset, get_device
from .utils import get_accelerator, last_ckp_path, save_model_and_scaler
from .types import _TypeBatch


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
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fun,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def setup(self, stage: Optional[Any] = None) -> None:
        pass


class LightningPlaNet(L.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        assert config.planet is not None, "must provide valid config.planet, got None"
        self.model = PlaNetCore(**config.planet.to_dict())
        self.loss_module = PlaNetLoss(is_physics_informed=config.is_physics_informed)

    def forward(self, *args: Any) -> Tensor:
        return self.model(*args)

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

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss = self._compute_loss_batch(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        # self.logger.log_metrics({'train_'+k:v for k,v in self.loss_module.log_dict.items()})
        for k, v in self.loss_module.log_dict.items():
            self.log(f"train_{k}", v, prog_bar=False)
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss = self._compute_loss_batch(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[LRSchedulerConfig]]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.02,
            total_iters=(
                5 if self.trainer.max_epochs is None else self.trainer.max_epochs
            ),
        )
        scheduler_config = LRSchedulerConfig(
            scheduler=scheduler,
            interval="epoch",
            frequency=1,
        )
        return [optimizer], [scheduler_config]


def main_train(config: Config) -> None:
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
