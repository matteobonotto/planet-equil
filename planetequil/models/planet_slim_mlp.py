from copy import deepcopy
from typing import Tuple, List

# import tensorflow as tf
import torch
from torch import Tensor, nn

from ..config import PlaNetConfig
from .layers import Conv2dNornAct, TrainableSwish


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


class TrunkNet_old(nn.Module):
    def __init__(self, hidden_dim: int = 128, nr: int = 32, nz: int = 32):
        super().__init__()
        assert nr % 2 == 0, f"nr must be a power of 2, got {nr}"
        assert nz % 2 == 0, f"nz must be a power of 2, got {nz}"
        self.norm_r = nn.BatchNorm2d(num_features=1)
        self.norm_z = nn.BatchNorm2d(num_features=1)
        self.trunk_r = nn.ModuleList()
        self.trunk_z = nn.ModuleList()
        channels: List[int] = [1, 8, 16, 32]
        for i in range(3):
            (in_channels, out_channels) = channels[i], channels[i + 1]
            self.trunk_r.append(
                nn.Sequential(
                    Conv2dNornAct(in_channels=in_channels, out_channels=out_channels),
                    nn.MaxPool2d(kernel_size=2),
                )
            )
            self.trunk_z.append(
                nn.Sequential(
                    Conv2dNornAct(in_channels=in_channels, out_channels=out_channels),
                    nn.MaxPool2d(kernel_size=2),
                )
            )
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(
            in_features=int(2 * channels[-1] * nr / 2**3 * nz / 2**3), out_features=128
        )
        self.act = TrainableSwish()
        self.linear_2 = nn.Linear(in_features=128, out_features=hidden_dim)

    def forward(self, x_r: Tensor, x_z: Tensor) -> Tensor:
        # branch for x_r
        x_r = self.norm_r(x_r.unsqueeze(1))
        for layer in self.trunk_r:
            x_r = layer(x_r)

        # branch for x_z
        x_z = self.norm_z(x_z.unsqueeze(1))
        for layer in self.trunk_z:
            x_z = layer(x_z)

        # concatenate branches and output
        x = torch.cat((x_r, x_z), dim=1)  # [batch, 32+32, 4, 4] -> 1024
        x = self.flatten(x)
        x = self.act(self.linear_1(x))
        x = self.linear_2(x)
        return x


class TrunkNet(nn.Module):
    def __init__(self, hidden_dim: int = 128, nr: int = 32, nz: int = 32):
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
        n_conv = len(channels)
        self.linear_1 = nn.Linear(
            in_features=int(2 * channels[-1] * nr / 2 ** (n_conv - 1)), out_features=128
        )
        self.act = TrainableSwish()
        self.linear_2 = nn.Linear(in_features=128, out_features=hidden_dim)

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

        # concatenate branches and output
        x = torch.cat((x_r, x_z), dim=1)
        x = self.flatten(x)
        x = self.act(self.linear_1(x))
        x = self.linear_2(x)
        return x


class BranchNet(nn.Module):
    def __init__(self, in_dim: int = 302, hidden_dim: int = 128):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_dim, out_features=hidden_dim)
        self.norm_1 = nn.BatchNorm1d(num_features=hidden_dim)
        self.act_1 = TrainableSwish(beta=1.0)
        self.linear_2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.norm_2 = nn.BatchNorm1d(num_features=hidden_dim)
        self.act_2 = TrainableSwish(beta=1.0)
        self.linear_3 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act_1(self.norm_1(self.linear_1(x)))
        x_skip = x
        x = self.act_2(self.norm_2(self.linear_2(x))) + x_skip
        x = self.linear_3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, nr: int = 32, nz: int = 32):
        super().__init__()
        assert nr % 2 == 0, f"nr must be a power of 2, got {nr}"
        assert nz % 2 == 0, f"nz must be a power of 2, got {nz}"
        self.nr = nr
        self.nz = nz
        # self.channels = [128, 32, 16, 8]
        # self.channels = [32, 16, 8, 4, 2]
        self.channels = [64, 32, 8, 4]  # (planet_slim_1) good choice
        # self.channels = [64, 32, 16, 8]  # planet_slim_2
        # self.channels = [64, 32, 16, 8, 1] # planet_slim_3
        # self.channels = [32, 16, 8, 4]  # planet_slim_4
        # self.channels = [64, 32, 16, 8, 4]  # planet_slim_5
        # channels = [32, 16, 8, 4, 1]
        n_conv = len(self.channels)
        self.linear = nn.Linear(
            in_features=hidden_dim,
            out_features=self.channels[0]
            * int(self.nr / 2 ** (n_conv - 1))
            * int(self.nz / 2 ** (n_conv - 1)),
        )
        self.act = TrainableSwish()
        self.decoder = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            in_channels, out_channels = self.channels[i], self.channels[i + 1]
            self.decoder.append(
                nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    Conv2dNornAct(in_channels=in_channels, out_channels=out_channels),
                )
            )
        self.conv = nn.Conv2d(
            in_channels=self.channels[-1],
            out_channels=1,
            kernel_size=(1, 1),
            padding="same",
        )

    def forward(self, x_trunk: Tensor, x_branch: Tensor) -> Tensor:
        n_conv = len(self.channels)
        batch = x_branch.shape[0]
        x = x_branch * x_trunk  # [batch, 64]
        x = self.act(self.linear(x))  # [batch, 2048]
        x = x.reshape(
            (
                batch,
                self.channels[0],
                int(self.nr / 2 ** (n_conv - 1)),
                int(self.nz / 2 ** (n_conv - 1)),
            )
        )

        for layer in self.decoder:
            x = layer(x)

        x = self.conv(x).squeeze()
        return x
    

class DecoderMLP(nn.Module):
    def __init__(self, hidden_dim: int = 128, nr: int = 32, nz: int = 32):
        super().__init__()
        self.nr = nr
        self.nz = nz
        self.linear_1 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.norm_1 = nn.BatchNorm1d(num_features=hidden_dim)
        self.act_1 = TrainableSwish(beta=1.0)
        self.linear_2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.norm_2 = nn.BatchNorm1d(num_features=hidden_dim)
        self.act_2 = TrainableSwish(beta=1.0)
        self.linear_3 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        # self.norm_3 = nn.BatchNorm1d(num_features=hidden_dim)
        # self.act_3 = TrainableSwish(beta=1.0)
        self.linear_4 = nn.Linear(in_features=hidden_dim, out_features=nr * nz)
        # self.conv = nn.Conv2d(
        #     in_channels=self.channels[-1],
        #     out_channels=1,
        #     kernel_size=(1, 1),
        #     padding="same",
        # )

    def forward(self, x_trunk: Tensor, x_branch: Tensor) -> Tensor:
        x = x_trunk * x_branch
        # x_skip = x
        x = self.act_1(self.norm_1(self.linear_1(x)))# + x_skip
        # x_skip = x
        x = self.act_2(self.norm_2(self.linear_2(x)))# + x_skip
        # x_skip = x
        # x = self.act_3(self.norm_3(self.linear_3(x))) + x_skip
        x = self.linear_4(x)
        # return self.conv(x.view(-1, self.nz, self.nz))
        return x.view(-1, self.nz, self.nz)


class PlaNetCoreSlimMLP(nn.Module):
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
        self.trunk = TrunkNet(hidden_dim=hidden_dim, nr=nr, nz=nz)
        self.branch = BranchNet(hidden_dim=hidden_dim, in_dim=n_measures)
        self.decoder = DecoderMLP(hidden_dim=hidden_dim, nr=nr, nz=nz)

    def forward(self, x: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        x_meas, x_r, x_z = x
        out_branch = self.branch(x_meas)
        out_trunk = self.trunk(x_r, x_z)
        return self.decoder(out_branch, out_trunk)
