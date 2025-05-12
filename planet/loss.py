from typing import Dict, List
import torch
from torch import Tensor, nn
import numpy as np
import torch.nn.functional as F


Gauss_kernel_3x3 = np.array(([1, 2, 1], [2, 4, 2], [1, 2, 1])) / (16)
Gauss_kernel_5x5 = np.array(
    (
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1],
    )
) / (273)


def _compute_grad_shafranov_operator(
    pred: Tensor,
    Laplace_kernel: Tensor,
    Df_dr_kernel: Tensor,
    RR: Tensor,
    ZZ: Tensor,
    Gauss_kernel: Tensor,
    smooth: bool = True,
) -> Tensor:
    """
    This implementation is taken from the paper https://doi.org/10.1016/j.fusengdes.2024.114193,
    cited as [1] in the code
    """

    batch_size = pred.shape[0]

    # Compute the Laplace and the Dpsi_dr operators
    Lpsi = F.conv2d(
        pred[:, None, ...].permute(1, 0, 2, 3),
        weight=Laplace_kernel[:, None, ...],
        groups=batch_size,
    ).permute(1, 0, 2, 3)

    # The '-' is necessary because the depthwise conv filters has to be transposed
    # to perform real convolution (here [+h 0 -h] -> [-h 0 +h])
    Dpsi_dr = -F.conv2d(
        pred[:, None, ...].permute(1, 0, 2, 3),
        weight=Df_dr_kernel[:, None, ...],
        groups=batch_size,
    ).permute(1, 0, 2, 3)
    Dpsi_dr = torch.div(Dpsi_dr, RR[:, None, 1:-1, 1:-1])

    # the LHS in equation 12 of [1]
    lhs = Lpsi - Dpsi_dr

    # GS operator (RHS of requation 12 of [1])
    hr = (RR[:, 1, 2] - RR[:, 1, 1])[:, None, None, None]
    hz = (ZZ[:, 2, 1] - ZZ[:, 1, 1])[:, None, None, None]
    alfa = -2 * (hr**2 + hz**2)
    beta = alfa / (hr**2 * hz**2)
    GS_ope = lhs * beta

    # convolve with the gaussian kernel to smooth the solution a bit
    if smooth:
        GS_ope = F.conv2d(
            GS_ope,
            weight=Gauss_kernel[None, None, ...],
            groups=1,
            padding="same",
        )

    return GS_ope.squeeze()


class GSOperatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.register_buffer(
            "Gauss_kernel", torch.tensor(Gauss_kernel_3x3, dtype=torch.float32)
        )

    def forward(
        self,
        pred: Tensor,
        rhs: Tensor,
        Laplace_kernel: Tensor,
        Df_dr_kernel: Tensor,
        RR: Tensor,
        ZZ: Tensor,
    ) -> Tensor:
        rhs_computed = _compute_grad_shafranov_operator(
            pred, Laplace_kernel, Df_dr_kernel, RR, ZZ, self.Gauss_kernel
        )
        return self.mse(rhs_computed, rhs)


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
        rhs: Tensor,
        Laplace_kernel: Tensor,
        Df_dr_kernel: Tensor,
        RR: Tensor,
        ZZ: Tensor,
    ) -> Tensor:
        mse_loss = self.scale_mse * self.loss_mse(input=pred, target=target)
        self.log_dict["mse_loss"] = mse_loss.item()
        if not self.is_physics_informed:
            return mse_loss
        else:
            pde_loss = self.scale_pde * self.loss_pde(
                pred=pred,
                rhs=rhs,
                Laplace_kernel=Laplace_kernel,
                Df_dr_kernel=Df_dr_kernel,
                RR=RR,
                ZZ=ZZ,
            )
            self.log_dict["pde_loss"] = pde_loss.item()
            return mse_loss + pde_loss
