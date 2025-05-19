import os
import sys

sys.path.append(os.getcwd())

import torch
from scipy import signal
import numpy as np

from planetequil.loss import _compute_grad_shafranov_operator, Gauss_kernel_5x5
from planetequil.train import DataModule
from planetequil.config import PlaNetConfig
from planetequil.utils import load_config


def test_gs_operator():
    ###
    datamodule = DataModule(load_config("planet/tests/data/config.yml"))
    dataloader = datamodule.train_dataloader()
    meas, flux, RHS_in, RR, ZZ, Laplace_kernel, Df_dr_kernel = next(iter(dataloader))

    ###
    gauss_kernel = torch.tensor(Gauss_kernel_5x5, dtype=torch.float32)
    rhs_computed = _compute_grad_shafranov_operator(
        flux, Laplace_kernel, Df_dr_kernel, RR, ZZ, gauss_kernel
    )

    (RHS_in - rhs_computed).norm(dim=0).shape
    diff = RHS_in - rhs_computed  # shape [batch, 32, 32]

    norm_difference = torch.norm(diff.view(diff.shape[0], -1), dim=1)  # shape [batch]
    norm_rhs = torch.norm(RHS_in.view(RHS_in.shape[0], -1), dim=1)  # shape [batch]
    norm = 100 * norm_difference / norm_rhs

    assert (
        norm < 5
    ).all(), (
        "error with _compute_grad_shafranov_operator in at least one element is > 5%"
    )


# def test_gs_operator_():
#     ###
#     # ds = PlaNetDataset(path="planet/tests/data/iter_like_data_sample.h5")
#     datamodule = DataModule(dataset_path="planet/tests/data/iter_like_data_sample.h5")
#     dataloader = datamodule.train_dataloader()
#     meas, flux, RHS_in, RR, ZZ, Laplace_kernel, Df_dr_kernel = next(iter(dataloader))

#     ###
#     gauss_kernel = torch.tensor(Gauss_kernel, dtype=torch.float32)
#     rhs_computed = _compute_grad_shafranov_operator(
#         flux, Laplace_kernel, Df_dr_kernel, RR, ZZ, gauss_kernel
#     )

#     ### this is the scipy implementation
#     # kr = np.array(([0, 0, 0], [1, -2, 1], [0, 0, 0]))*hz**2
#     # kz = np.transpose(np.array(([0, 0, 0], [1, -2, 1], [0, 0, 0])))*hr**2

#     i_case = 0
#     for i_case in range(10):
#         rhs_i = rhs_computed[i_case, ...].numpy()
#         L_ker = Laplace_kernel[i_case,...].numpy()
#         Df_dr_ker = Df_dr_kernel[i_case,...].numpy()
#         psi = flux[i_case,...].numpy()
#         rr_in = RR[i_case,1:-1, 1:-1].numpy()

#         hr = (RR[i_case,1,2] - RR[i_case,1,1]).cpu().detach().numpy()
#         hz = (ZZ[i_case,2,1] - ZZ[i_case,1,1]).cpu().detach().numpy()
#         Lpsi = signal.convolve2d(psi, L_ker, mode='valid')
#         Dpsi_dr = signal.convolve2d(psi, Df_dr_ker, mode='valid')
#         lhs_scipy = Lpsi - Dpsi_dr/rr_in

#         alfa = -2 * (hr**2 + hz**2)
#         beta = alfa / (hr**2 * hz**2)
#         GS_ope_scipy = lhs_scipy * beta

#         error_GS_ope = np.linalg.norm(GS_ope_scipy - rhs_i)/np.linalg.norm(rhs_i)
#         print(f'scipy: {error_GS_ope=}')

#         fun_contourf_sol(
#             GS_ope_scipy - rhs_i,
#             RR[i_case, 1:-1, 1:-1],
#             ZZ[i_case, 1:-1, 1:-1],
#             title='scipy',
#         )

#         # fun_contourf_sol(
#         #     rhs_computed[i_case, ...],
#         #     RR[i_case, 1:-1, 1:-1],
#         #     ZZ[i_case, 1:-1, 1:-1],
#         # )


#         import torch.nn.functional as F
#         batch_size = meas.shape[0]
#         pred = flux
#         # Compute the Laplace and the Dpsi_dr operators
#         Lpsi = F.conv2d(
#             pred[:, None, ...].permute(1, 0, 2, 3),
#             weight=Laplace_kernel[:, None, ...],
#             groups=batch_size,
#         ).permute(1, 0, 2, 3)

#         # The '-' is necessary because the depthwise conv filters has to be transposed
#         # to perform real convolution (here [+h 0 -h] -> [-h 0 +h])
#         Dpsi_dr = -F.conv2d(
#             pred[:, None, ...].permute(1, 0, 2, 3),
#             weight=Df_dr_kernel[:, None, ...],
#             groups=batch_size,
#         ).permute(1, 0, 2, 3)

#         # the LHS in equation 12 of [1]
#         lhs = Lpsi - torch.div(Dpsi_dr, RR[:, None, 1:-1, 1:-1])

#         # GS operator (RHS of requation 12 of [1])
#         hr = (RR[:, 1, 2] - RR[:, 1, 1])[:, None, None, None]
#         hz = (ZZ[:, 2, 1] - ZZ[:, 1, 1])[:, None, None, None]
#         alfa = -2 * (hr**2 + hz**2)
#         beta = alfa / (hr**2 * hz**2)
#         GS_ope = (lhs * beta)

#         GS_ope_smooth = F.conv2d(
#             GS_ope,
#             weight=gauss_kernel[None, None, ...],
#             groups=1,
#             padding="same",
#         ).squeeze()

#         # error_lhs = np.linalg.norm(lhs_scipy - lhs[i_case, ...].numpy())/np.linalg.norm(lhs_scipy)
#         # print(error_lhs)

#         error_GS_ope = np.linalg.norm(GS_ope_smooth[i_case, ...].numpy() - rhs_i)/np.linalg.norm(rhs_i)
#         print(error_GS_ope)

#         print(f'torch: {error_GS_ope=}')

#         fun_contourf_sol(
#             GS_ope_smooth[i_case, ...].numpy() - rhs_i,
#             RR[i_case, 1:-1, 1:-1],
#             ZZ[i_case, 1:-1, 1:-1],
#             title='torch',
#         )

#         if error_GS_ope > 1:
#             pass


#     pass

#     # Df_dr_kernel = np.array(([0, 0, 0], [+1, 0, -1], [0, 0, 0]))/(2*hr)

#     # # LHS_conv = Lpsi - Dpsi_dr/RR_in
#     # GS_ope = Lpsi - Dpsi_dr/RR_in

#     # # x = rhs_computed[0, ...]
#     # fun_contourf_sol(
#     #     rhs_computed[0, ...],
#     #     RR[0, 1:-1, 1:-1],
#     #     ZZ[0, 1:-1, 1:-1],
#     # )

#     # # fun_contourf_sol(
#     # #     rhs_computed[0, ...] - RHS_in[0, ...],
#     # #     RR[0, 1:-1, 1:-1],
#     # #     ZZ[0, 1:-1, 1:-1],
#     # # )

# test_gs_operator()
