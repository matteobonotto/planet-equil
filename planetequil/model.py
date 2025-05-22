from copy import deepcopy
from typing import Tuple, List

# import tensorflow as tf
import torch
from torch import Tensor, nn

from ..config import PlaNetConfig
from .models.layers import Conv2dNornAct, TrainableSwish


class TrunkNet(nn.Module):
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
        # [batch, 1, 32, 32]
        # [batch, 8, 16, 16]
        # [batch, 16, 8, 8]
        # [batch, 32, 4, 4] -> 512
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(
            in_features=int(2 * channels[-1] * nr / 2**3 * nz / 2**3), out_features=128
        )
        self.act = TrainableSwish()
        self.linear_2 = nn.Linear(in_features=128, out_features=hidden_dim)

    def forward(self, x_r: Tensor, x_z: Tensor) -> Tensor:
        """
        # x_r = input_query_RR
        x_r = layers.BatchNormalization()(input_query_RR)
        for i in range(3):
            x_r = conv2D_Norm_activation(x_r, filters=(i + 1) * 8, kernel_size=(3, 3))
            x_r = layers.MaxPooling2D(pool_size=(2, 2))(x_r)

        # x_z = input_query_ZZ
        x_z = layers.BatchNormalization()(input_query_ZZ)
        for i in range(3):
            x_z = conv2D_Norm_activation(x_z, filters=(i + 1) * 8, kernel_size=(3, 3))
            x_z = layers.MaxPooling2D(pool_size=(2, 2))(x_z)

        out_trunk = layers.Concatenate()([x_r, x_z])
        out_trunk = layers.Flatten()(out_trunk)
        out_trunk = layers.Dense(128, dtype=DTYPE)(out_trunk)
        out_trunk = Swish(beta=1.0, trainable=True)(out_trunk)

        for i in range(2):
            out_trunk = layers.Dense(64, dtype=DTYPE)(out_trunk)
        out_trunk = Swish(beta=1.0, trainable=True)(out_trunk)
        """

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


class BranchNet(nn.Module):
    def __init__(self, in_dim: int = 302, hidden_dim: int = 128):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_dim, out_features=256)
        self.norm_1 = nn.BatchNorm1d(num_features=256)
        self.act_1 = TrainableSwish(beta=1.0)
        self.linear_2 = nn.Linear(in_features=256, out_features=128)
        self.norm_2 = nn.BatchNorm1d(num_features=128)
        self.act_2 = TrainableSwish(beta=1.0)
        self.linear_3 = nn.Linear(in_features=128, out_features=hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        x = layers.Dense(256, dtype=DTYPE)(input_fun)
        x = Swish(beta=1.0, trainable=True)(x)

        x = layers.Dense(128, dtype=DTYPE)(x)
        x = Swish(beta=1.0, trainable=True)(x)

        x = layers.Dense(64, dtype=DTYPE)(x)
        out_branch = Swish(beta=1.0, trainable=True)(x)
        """
        x = self.act_1(self.norm_1(self.linear_1(x)))
        x = self.act_2(self.norm_2(self.linear_2(x)))
        x = self.linear_3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, nr: int = 32, nz: int = 32):
        super().__init__()
        assert nr % 2 == 0, f"nr must be a power of 2, got {nr}"
        assert nz % 2 == 0, f"nz must be a power of 2, got {nz}"
        self.nr = nr
        self.nz = nz
        self.linear = nn.Linear(
            in_features=hidden_dim,
            out_features=128 * int(self.nr / 2**3) * int(self.nz / 2**3),
        )
        self.act = TrainableSwish()
        self.decoder = nn.ModuleList()
        # channels = [32, 16, 8, 4, 1]
        channels = [128, 32, 16, 8]
        for i in range(len(channels) - 1):
            in_channels, out_channels = channels[i], channels[i + 1]
            self.decoder.append(
                nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    Conv2dNornAct(in_channels=in_channels, out_channels=out_channels),
                )
            )
        self.conv = nn.Conv2d(
            in_channels=channels[-1], out_channels=1, kernel_size=(1, 1), padding="same"
        )

    def forward(self, x_trunk: Tensor, x_branch: Tensor) -> Tensor:
        """
        # Multiply layer
        out_multiply = layers.Multiply(name="Multiply")([out_branch, out_trunk])

        # conv2d-based decoder
        x_dec = layers.Dense(
            neuron_FC,
            dtype=DTYPE,
        )(out_multiply)
        x_dec = Swish(beta=1.0, trainable=True)(x_dec)

        x_dec = layers.Reshape(target_shape=(n_w, n_h, n_c))(x_dec)

        x_dec = layers.UpSampling2D(size=(2, 2), interpolation=interpolation)(x_dec)
        x_dec = conv2D_Norm_activation(x_dec, filters=32, kernel_size=(3, 3))

        x_dec = layers.UpSampling2D(size=(2, 2), interpolation=interpolation)(x_dec)
        x_dec = conv2D_Norm_activation(x_dec, filters=16, kernel_size=(3, 3))

        x_dec = layers.UpSampling2D(size=(2, 2), interpolation=interpolation)(x_dec)
        x_dec = conv2D_Norm_activation(x_dec, filters=8, kernel_size=(3, 3))

        out_grid = layers.Conv2D(
            filters=1,
            kernel_size=(1, 1),
            strides=1,
            padding="same",
            activation="linear",
            kernel_initializer="he_normal",
            bias_initializer="zeros",
            dtype=DTYPE,
        )(x_dec)

        outputs = out_grid
        """
        # batch, n_hidden = x_branch.shape
        batch = x_branch.shape[0]
        x = x_branch * x_trunk  # [batch, 64]
        x = self.act(self.linear(x))  # [batch, 2048]
        # x = x.reshape((batch, 128, 4, 4))
        x = x.reshape((batch, 128, int(self.nr / 2**3), int(self.nz / 2**3)))

        for layer in self.decoder:
            x = layer(x)

        x = self.conv(x).squeeze()
        return x


class PlaNetCore(nn.Module):
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
        self.decoder = Decoder(hidden_dim=hidden_dim, nr=nr, nz=nz)

    def forward(self, x: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        x_meas, x_r, x_z = x
        out_branch = self.branch(x_meas)
        out_trunk = self.trunk(x_r, x_z)
        return self.decoder(out_branch, out_trunk)
