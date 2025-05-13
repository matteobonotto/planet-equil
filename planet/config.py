from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


@dataclass
class PlaNetConfig:
    hidden_dim: int = 128
    nr: int = 64
    nz: int = 64
    branch_in_dim: int = 302

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
    planet: Optional[PlaNetConfig] = None
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
            cls_instance.planet = PlaNetConfig(**cls_instance.planet)

        return cls_instance
