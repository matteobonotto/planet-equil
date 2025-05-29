from copy import deepcopy
from typing import Tuple, List

# import tensorflow as tf
import torch
from torch import Tensor, nn
from torchinfo import summary

from ..config import PlaNetConfig
from .layers import Conv2dNornAct, TrainableSwish, GatedMLP, GatedMLPStack, MLPStack


class TrunkNet_(nn.Module):
    def __init__(
        self,
        in_dim: int = 6,
        out_dim: int = 128,
        hidden_dim: int = 16,
        n_layers: int = 2,
    ):
        super().__init__()
        self.norm = nn.BatchNorm1d(num_features=in_dim)
        self.block = GatedMLPStack(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )

    # @staticmethod
    # def _compress_grid(grid: Tensor) -> Tensor:
    #     # compress a coordinate array coming from measgrid to 3 items: [stard, stop, step]
    #     if grid[0, 0, 0] != grid[0, 1, 0]:
    #         coordinates = grid[:, :, 0]
    #     else:
    #         coordinates = grid[:, 0, :]
    #     compressed = torch.stack(
    #         [
    #             torch.max(coordinates, dim=1).values,
    #             torch.min(coordinates, dim=1).values,
    #             coordinates[:, 1] - coordinates[:, 0],
    #         ],
    #         dim=-1,
    #     )
    #     return compressed

    @staticmethod
    def _compress_grid(grid: Tensor) -> Tensor:
        if grid[0, 0, 0] != grid[0, 1, 0]:
            return grid[:, :, 0]
        else:
            return grid[:, 0, :]

    def forward(self, x_r: Tensor, x_z: Tensor) -> Tensor:
        x = torch.cat([self._compress_grid(x_r), self._compress_grid(x_z)], dim=-1)
        x = self.norm(x)
        return self.block(x)


class Conv1dNornAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: str = "same",
    ):
        super().__init__()
        self.conv2d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.norm = nn.BatchNorm1d(num_features=out_channels)
        self.act = TrainableSwish()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv2d(x)))


class TrunkNet(nn.Module):
    def __init__(
        self,
        in_dim: int = 6,
        out_dim: int = 128,
        hidden_dim: int = 16,
        n_layers: int = 1,
        nr: int = 32,
        nz: int = 32,
    ):
        super().__init__()
        assert nr == nz, f"nr must be equal to nz, got {nr=}, {nz=}"
        self.norm_r = nn.BatchNorm1d(num_features=1)
        self.norm_z = nn.BatchNorm1d(num_features=1)
        self.trunk_r = nn.ModuleList()
        self.trunk_z = nn.ModuleList()
        channels: List[int] = [1, 2, 4]
        for i in range(len(channels) - 1):
            (in_channels, out_channels) = channels[i], channels[i + 1]
            self.trunk_r.append(
                nn.Sequential(
                    Conv1dNornAct(in_channels=in_channels, out_channels=out_channels),
                    nn.MaxPool1d(kernel_size=2),
                )
            )
            self.trunk_z.append(
                nn.Sequential(
                    Conv1dNornAct(in_channels=in_channels, out_channels=out_channels),
                    nn.MaxPool1d(kernel_size=2),
                )
            )
        self.flatten = nn.Flatten()
        self.block = MLPStack(
            in_dim=2 * channels[-1] * nr // 2 ** (len(channels) - 1),
            out_dim=out_dim,
            n_layers=n_layers,
        )

    @staticmethod
    def _compress_grid(grid: Tensor) -> Tensor:
        if grid[0, 0, 0] != grid[0, 1, 0]:
            return grid[:, :, 0]
        else:
            return grid[:, 0, :]

    def forward(self, x_r: Tensor, x_z: Tensor) -> Tensor:
        # branch for x_r
        x_r = self._compress_grid(x_r)
        x_r = self.norm_r(x_r.unsqueeze(1))
        for layer in self.trunk_r:
            x_r = layer(x_r)

        # branch for x_z
        x_z = self._compress_grid(x_z)
        x_z = self.norm_z(x_z.unsqueeze(1))
        for layer in self.trunk_z:
            x_z = layer(x_z)
            pass

        x = self.flatten(torch.cat([x_r, x_z], dim=-1))
        return self.block(x)


class PlaNetCoreSlim(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        nr: int = 64,
        nz: int = 64,
        n_measures: int = 302,
    ):
        super().__init__()
        self.config = PlaNetConfig(
            nr=nr, nz=nz, hidden_dim=hidden_dim, n_measures=n_measures
        )
        self.trunk = TrunkNet(
            in_dim=nr + nz,
            out_dim=hidden_dim,
            hidden_dim=hidden_dim,
            n_layers=1,
            nr=nr,
            nz=nz,
        )
        self.branch = MLPStack(
            in_dim=n_measures,
            out_dim=hidden_dim,
            n_layers=2,
        )
        self.decoder = MLPStack(
            in_dim=hidden_dim,
            out_dim=nr * nz,
            n_layers=3,
        )

    def forward(self, x: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        x_meas, x_r, x_z = x
        out_trunk = self.trunk(x_r, x_z)
        out_branch = self.branch(x_meas)
        out = self.decoder(out_branch * out_trunk)
        return out.view(-1, self.config.nr, self.config.nz)


class PlaNetCoreSlim_(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        nr: int = 64,
        nz: int = 64,
        n_measures: int = 302,
    ):
        super().__init__()
        self.config = PlaNetConfig(
            nr=nr, nz=nz, hidden_dim=hidden_dim, n_measures=n_measures
        )
        self.trunk = TrunkNet(
            in_dim=nr + nz,
            out_dim=hidden_dim,
            hidden_dim=hidden_dim,
            n_layers=3,
            nr=nr,
            nz=nz,
        )
        self.branch = GatedMLPStack(
            in_dim=n_measures,
            out_dim=hidden_dim,
            hidden_dim=hidden_dim,
            n_layers=4,
        )
        self.decoder = GatedMLPStack(
            in_dim=hidden_dim,
            out_dim=nr * nz,
            hidden_dim=hidden_dim,
            n_layers=4,
        )

    def forward(self, x: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        x_meas, x_r, x_z = x
        out_trunk = self.trunk(x_r, x_z)
        out_branch = self.branch(x_meas)
        out = self.decoder(out_branch * out_trunk)
        return out.view(-1, self.config.nr, self.config.nz)
