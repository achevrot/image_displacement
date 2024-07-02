# %%
import numpy as np
from tinygrad import Tensor

# %%


def load():
    ds = np.load("dataset.npy")
    n_split = int(ds.shape[0] * 0.8)
    np.random.shuffle(ds)

    ds_train, ds_test = ds[:n_split, :, :], ds[n_split:, :, :]

    # reminder : dim 1 = ref, mod, Ux, Uy

    X_train, Y_train = (
        Tensor(ds_train[:, :2, :, :]),
        Tensor(ds_train[:, 2:, :, :]),
    )
    X_test, Y_test = (
        Tensor(ds_test[:, :2, :, :]),
        Tensor(ds_test[:, 2:, :, :]),
    )

    return X_train, Y_train, X_test, Y_test


# %%
