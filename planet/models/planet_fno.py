from typing import Tuple
from torch import nn, Tensor
from torchinfo import summary


from ..utils import dummy_planet_input
from .layers import MLPStack


class SpectralConv2D(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class FNOBlock(nn.Module):
    def __init__(
        self, n_harm_in: int = 64, n_harm_out: int = 24, hidden_dim: int = 128
    ):
        super().__init__()
        self.spectral_conv = SpectralConv2D()
        # self.w = nn.Linear(in_features=)

    def forward(self, x):
        pass


class FNO(nn.Module):
    def __init__(self, n_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([FNOBlock() for _ in n_layers])

    def forward(self, x):
        # TODO: (SpectralConvolution + skip connections, nonlinearity)
        for l in self.layers:
            x = l(x)
        return x


class PlaNetFNO(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        nr: int = 64,
        nz: int = 64,
        n_measures: int = 302,
    ):
        super().__init__()

        self.sensor_lifter = MLPStack(
            in_dim=n_measures,
            out_dim=nr * nz,
            hidden_dim=hidden_dim,
            n_layers=2,
        )
        self.fno_decoder = None

    def forward(self, x: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        measures, x_r, x_z = x

        measures_lift = self.sensor_lifter(measures)
        out = self.fno_decoder(measures_lift, x_r, x_z)

        return out


if __name__ == "__main__":
    model = PlaNetFNO()
    inputs = dummy_planet_input()
    summary(model, input_data=inputs)
