import torch
from planet.models.planet_conv import TrunkNet, BranchNet, Decoder, PlaNetCore

DTYPE = torch.float32


def test_BranchNet():
    measures = torch.rand([8, 302], dtype=DTYPE)
    hidden_dim = [64, 128]
    for hd in hidden_dim:
        branch_net = BranchNet(in_dim=measures.shape[-1], hidden_dim=hd)
        out_branch = branch_net(measures)
        assert out_branch.shape == (8, hd)


def test_TrunkNet():
    RR, ZZ = (
        torch.rand([8, 64, 64], dtype=DTYPE),
        torch.rand([8, 64, 64], dtype=DTYPE),
    )
    trunk_net = TrunkNet(nr=64, nz=64)
    out_trunk = trunk_net(RR, ZZ)
    assert out_trunk.shape == (8, 128)

    RR, ZZ = (
        torch.rand([8, 32, 32], dtype=DTYPE),
        torch.rand([8, 32, 32], dtype=DTYPE),
    )
    trunk_net = TrunkNet()
    out_trunk = trunk_net(RR, ZZ)
    assert out_trunk.shape == (8, 128)


test_TrunkNet()


def test_PlaNetCore():
    measures = torch.rand([8, 302], dtype=DTYPE)

    hidden_dim = 128
    nr = nz = 64
    RR, ZZ = (
        torch.rand([8, nr, nz], dtype=DTYPE),
        torch.rand([8, nr, nz], dtype=DTYPE),
    )
    planet_core = PlaNetCore(hidden_dim=hidden_dim, nr=nr, nz=nz)
    out_planet = planet_core((measures, RR, ZZ))

    assert out_planet.shape == (8, nr, nz)

    hidden_dim = 64
    nr = nz = 32
    RR, ZZ = (
        torch.rand([8, nr, nz], dtype=DTYPE),
        torch.rand([8, nr, nz], dtype=DTYPE),
    )
    planet_core = PlaNetCore(hidden_dim=hidden_dim, nr=nr, nz=nz)
    out_planet = planet_core((measures, RR, ZZ))

    assert out_planet.shape == (8, nr, nz)


# if __name__ == "__main__":
#     train_ds_1 = tf.data.Dataset.load(
#         "./data/tf_Dataset_NeuralOpt_all_domain_only_32x32.data"
#     )
#     train_ds_2 = tf.data.Dataset.load(
#         "./data/tf_Dataset_NeuralOpt_super_res_only_32x32.data"
#     )
#     # train_ds_1 = tf.data.Dataset.load('./gdrive/MyDrive/Colab_Notebooks/tf_Datasets/tf_Dataset_NeuralOpt_all_domain_only_64x64.data')
#     # train_ds_2 = tf.data.Dataset.load('./gdrive/MyDrive/Colab_Notebooks/tf_Datasets/tf_Dataset_NeuralOpt_super_res_only_64x64.data')
#     dict_geo = scipy.io.loadmat("./data/data_geo_Dataset_NeuralOpt_super_res_32x32.mat")

#     train_ds = train_ds_1.concatenate(train_ds_2)
#     train_ds = train_ds.shuffle(42)
#     # train_ds = train_ds.batch(1024)

#     x_ds, y_ds, RHS_in_ds, RR_ds, ZZ_ds, Laplace_kernel_ds, Df_dr_kernel_ds = iter(
#         train_ds
#     ).next()

#     print(f"batch size: {x_ds.shape[0]}")
#     print(train_ds.cardinality().numpy() * x_ds.shape[0])
#     print(x_ds.shape)

#     n_grid = RR_ds.shape[1]
#     n_output = y_ds.shape[1]
#     n_input = x_ds.shape[1]

#     RR_pixels = dict_geo["RR_pixels"]
#     ZZ_pixels = dict_geo["ZZ_pixels"]

#     RHS_i = RHS_in_ds[-1, :, :].numpy()
#     psi_i = y_ds[-1, :, :].numpy()

#     inputs = [x_ds, RR_ds, ZZ_ds]

#     branch_net = BranchNet(in_dim=x_ds.shape[-1])
#     out_branch = branch_net(torch.tensor(x_ds.numpy()))

#     trunk_net = TrunkNet()
#     out_trunk = trunk_net(
#         torch.tensor(RR_ds.numpy()).unsqueeze(1),
#         torch.tensor(ZZ_ds.numpy()).unsqueeze(1),
#     )

#     decoder = Decoder()
#     out = decoder(out_trunk, out_branch)

#     planet = PlaNetCore()
#     summary(
#         planet,
#         input_data=(
#             torch.tensor(x_ds.numpy()),
#             torch.tensor(RR_ds.numpy()).unsqueeze(1),
#             torch.tensor(ZZ_ds.numpy()).unsqueeze(1),
#         ),
#     )

#     # loss functions
