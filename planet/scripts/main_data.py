from planet.utils import read_h5_numpy, write_h5

if __name__ == "__main__":

    from scipy import io

    mat = io.loadmat(
        "/Users/matte/Documents/RESEARCH/PlaNet_Equil_reconstruction/ITER_like_equilibrium_dataset.mat"
    )

    print(mat.keys())

    measures = mat["DB_meas_Bpickup_test_ConvNet"]
    flux = mat["DB_psi_pixel_test_ConvNet"]
    rhs = mat["DB_res_RHS_pixel_test_ConvNet"]
    # jpla          = mat['DB_Jpla_pixel_test_ConvNet']
    coils_current = mat["DB_coils_curr_test_ConvNet"]
    RR_grid = mat["RR_pixels"]
    ZZ_grid = mat["ZZ_pixels"]
    # separatrix_coordinates     = mat['DB_separatrix_200_test_ConvNet']
    # is_diverted                               = mat['XP_YN']
    # f_profile                   = mat['DB_f_test_ConvNet']
    p_profile = mat["DB_p_test_ConvNet"]

    data = {
        "measures": measures,
        "flux": flux,
        "rhs": rhs,
        "coils_current": coils_current,
        "RR_grid": RR_grid,
        "ZZ_grid": ZZ_grid,
        "p_profile": p_profile,
    }
    # data = {k: v.astype("float32") for k, v in data.items()}
    write_h5(data, filename="iter_like_data")

    data_sample = {
        k: v[:64, ...] for k, v in data.items() if "RR" not in k and "ZZ" not in k
    }
    data_sample["RR_grid"] = data["RR_grid"]
    data_sample["ZZ_grid"] = data["ZZ_grid"]
    write_h5(data_sample, filename="planet/tests/data/iter_like_data_sample")

    import pyarrow as pa

    pa.Table.from_pydict(data)

    write_h5(data, filename="iter_like_data")
    import time

    t0 = time.time()
    data = read_h5_numpy(filename="iter_like_data.h5")
    t1 = time.time() - t0

    # # create the dataset with the equilibria on the full grid
    # import tensorflow as tf

    # ds = tf.data.Dataset.load(
    #     "/Users/matte/Library/CloudStorage/GoogleDrive-matteobonotto90@gmail.com/My Drive/Colab_Notebooks/PlaNet/tf_Datasets/tf_Dataset_NeuralOpt_all_domain_only_32x32.data"
    # )

    # measures, flux, rhs, RR, ZZ, L_ker, Df_ker = next(iter(ds))
    # batch_size = measures.shape[0]

    # RR = RR[0, ...].numpy()
    # ZZ = ZZ[0, ...].numpy()

    # measures = np.zeros((len(ds)*measures.shape[0], *measures.shape[1:]))
    # flux = np.zeros((len(ds)*flux.shape[0], *flux.shape[1:]))
    # rhs = np.zeros((len(ds)*rhs.shape[0], *rhs.shape[1:]))

    # for i, batch in enumerate(ds):
    #    idx = slice(i*batch_size, (i+1)*batch_size)
    #    if batch[0].shape[0] == batch_size:
    #         measures[idx, ...] = batch[0]
    #         flux[idx, ...] = batch[1]
    #         rhs[idx, ...] = batch[2]

    # measures_l = []
    # flux_l = []
    # rhs_l = []
    # for i, batch in enumerate(ds):
    #    if batch[0].shape[0] == batch_size:
    #         measures_l.append(batch[0])
    #         flux_l.append(batch[1])
    #         rhs_l.append(batch[2])
    #         # if i >= 3:
    #         #     break
    # measures = np.concat(measures_l, axis=0)
    # flux = np.concat(flux_l, axis=0)
    # rhs = np.concat(rhs_l, axis=0)

    # write_h5(
    #    {
    #       "measures": measures,
    #       "flux": flux,
    #       "rhs": rhs,
    #       "RR": RR,
    #       "ZZ": ZZ,
    #    },
    #    filename="planet_data"
    # )
    # data = read_h5_numpy(filename="planet_data.h5")
