# %%
import numpy as np
from tinygrad import Tensor

# %%


def load():
    ds = np.load("dataset.npy")
    n_split = int(ds.shape[0] * 0.8)
    np.random.shuffle(ds)

    ds_train, ds_test = ds[:n_split, :], ds[n_split:, :]

    X_train, Y_train = (
        Tensor(ds_train[:, 0, np.newaxis, :, :]),
        Tensor(ds_train[:, 1, np.newaxis, :, :]),
    )
    X_test, Y_test = (
        Tensor(ds_test[:, 0, np.newaxis, :, :]),
        Tensor(ds_test[:, 1, np.newaxis, :, :]),
    )

    return X_train, Y_train, X_test, Y_test


# %%
