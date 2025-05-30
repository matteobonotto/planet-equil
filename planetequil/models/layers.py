from copy import deepcopy
from typing import Tuple, List, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TrainableSwish(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x: Tensor) -> Tensor:
        return swish(x, self.beta)


def swish(x: Tensor, beta: nn.Parameter) -> Tensor:
    return x * F.sigmoid(beta * x)


class Conv2dNornAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        padding: str = "same",
    ):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.act = TrainableSwish()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv2d(x)))


class GatedMLP(nn.Module):
    def __init__(
        self,
        in_features: int = 64,
        hidden_dim: int = 64,
        out_featues: Optional[int] = None,
    ):
        super().__init__()
        if out_featues is None:
            out_featues = in_features
        self.linear1 = nn.Linear(in_features=in_features, out_features=2 * hidden_dim)
        self.norm = nn.BatchNorm1d(num_features=2 * hidden_dim)
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=out_featues)
        self.act = TrainableSwish()

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.norm(x)
        u, v = x.chunk(chunks=2, dim=-1)
        gated = self.act(u)
        return self.linear2(gated * v)


class GatedMLPStack(nn.Module):
    def __init__(
        self,
        in_dim: int = 302,
        out_dim: int = 128,
        hidden_dim: int = 128,
        n_layers: int = 2,
    ):
        super().__init__()
        layers = []
        for i in range(n_layers):
            if i == 0:
                layer = GatedMLP(
                    in_features=in_dim, hidden_dim=hidden_dim, out_featues=hidden_dim
                )
            elif i == n_layers - 1:
                layer = GatedMLP(
                    in_features=hidden_dim, hidden_dim=hidden_dim, out_featues=out_dim
                )
            else:
                layer = GatedMLP(
                    in_features=hidden_dim,
                    hidden_dim=hidden_dim,
                    out_featues=hidden_dim,
                )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        for l in self.layers:
            x = l(x)
            pass
        return x


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int = 64,
        out_featues: Optional[int] = None,
        activation: bool = True,
    ):
        super().__init__()
        if out_featues is None:
            out_featues = in_features
        self.linear = nn.Linear(in_features=in_features, out_features=out_featues)
        self.norm = nn.BatchNorm1d(num_features=out_featues)
        self.act = TrainableSwish() if activation else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.linear(x)))


class MLPStack(nn.Module):
    def __init__(
        self,
        in_dim: int = 302,
        out_dim: int = 128,
        hidden_dim: int = 128,
        n_layers: int = 2,
    ):
        super().__init__()
        if n_layers == 1:
            self.layers = nn.ModuleList(
                [MLP(in_features=in_dim, out_featues=out_dim, activation=False)]
            )
        else:
            layers = []
            for i in range(n_layers):
                if i == 0:
                    layer = MLP(in_features=in_dim, out_featues=hidden_dim)
                elif i == n_layers - 1:
                    layer = MLP(
                        in_features=hidden_dim, out_featues=out_dim, activation=False
                    )
                else:
                    layer = MLP(
                        in_features=hidden_dim,
                        out_featues=hidden_dim,
                    )
                layers.append(layer)
            self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        for l in self.layers:
            x = l(x)
        return x


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
