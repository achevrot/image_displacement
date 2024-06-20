# %%

from typing import List, Callable
from tinygrad import Tensor, TinyJit, nn, GlobalCounters
from tinygrad.helpers import getenv, colored
from tqdm import trange
from dataloader import load
from model import Encoder, Decoder
from helpers import img_2_tensor, plot_image
from tinygrad.nn.state import get_parameters
import numpy as np

# %%

X_train, _, _, _ = load()

c_hid = 32
latent_dim = 64

# %%


class RefAutoEncoder:
    def __init__(self, latent_dim=64):
        self.ref_encoder = Encoder(latent_dim)
        self.ref_decoder = Decoder(latent_dim)

    def __call__(self, x: Tensor) -> Tensor:

        return x.sequential([self.ref_encoder, self.ref_decoder])

    def get_latent_rep(self, x: Tensor) -> Tensor:

        return self.ref_encoder(x)

    def save(self, filename):
        with open(filename + ".npy", "wb") as f:
            for par in get_parameters(self):
                # if par.requires_grad:
                np.save(f, par.numpy())

    def train_step(self, opt, input, output) -> Tensor:
        with Tensor.train():
            opt.zero_grad()
            samples = Tensor.randint(1, high=input.shape[0])
            # TODO: this "gather" of samples is very slow. will be under 5s when this is fixed
            loss = MSEloss(
                self.__call__(input[samples]), output[samples]
            ).backward()
            opt.step()
            return loss


class AutoEncoder:
    def __init__(self, x_ref, latent_dim=64):
        self.ref_lat = rae.get_latent_rep(x_ref)
        self.layers = [
            Encoder(latent_dim),
            lambda x: x.add(self.ref_lat),
            Decoder(latent_dim),
        ]
        self.dec = []

    def __call__(self, x: Tensor) -> Tensor:

        return x.sequential(self.layers)

    def train_step(self, opt, input, output) -> Tensor:
        with Tensor.train():
            opt.zero_grad()
            samples = Tensor.randint(1, high=input.shape[0])
            # TODO: this "gather" of samples is very slow. will be under 5s when this is fixed
            loss = MSEloss(
                self.__call__(input[samples]), output[samples]
            ).backward()
            opt.step()
            return loss

    def save(self, filename):
        with open(filename + ".npy", "wb") as f:
            for par in get_parameters(self):
                # if par.requires_grad:
                np.save(f, par.numpy())


# %%


def MSEloss(y_hat, y):
    return ((y_hat - y) ** 2).sum(axis=[1, 2, 3]).mean(axis=[0])


# X_train, Y_train, X_test, Y_test = load()
X_ref = img_2_tensor("Dataset/source/ref.tif").expand((1, -1, -1, -1))
# TODO: remove this when HIP is fixed
# X_train, X_test = X_train.float(), X_test.float()
# samples = Tensor.randint(1, high=X_train.shape[0])

epochs = 15
rae = RefAutoEncoder(128)
optimizer = nn.optim.Adam(nn.state.get_parameters(rae))


def get_test_acc() -> Tensor:
    return MSEloss(rae(X_ref), X_ref)


for epoch in range(1, epochs + 1):
    print(f"epoch {epoch}/{epochs}")
    for i in (t := trange(70)):
        GlobalCounters.reset()  # NOTE: this makes it nice for DEBUG=2 timing
        loss = rae.train_step(optimizer, X_ref, X_ref)
        t.set_description(f"loss: {loss.item():6.5f}")
    if i % 10 == 9:
        test_acc = get_test_acc().item()

    # verify eval acc
    if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
        if test_acc >= target and test_acc != 100.0:
            print(colored(f"{test_acc=} >= {target}", "green"))
        else:
            raise ValueError(colored(f"{test_acc=} < {target}", "red"))

    print(f"test_accuracy: {test_acc:5.5f}")

plot_image(X_ref[0][0], rae(X_ref)[0][0])

# %%

X_train, Y_train, X_test, Y_test = load()
X_ref = img_2_tensor("Dataset/source/ref.tif").expand((1, -1, -1, -1))
# TODO: remove this when HIP is fixed
X_train, X_test = X_train.float(), X_test.float()
# samples = Tensor.randint(1, high=X_train.shape[0])

epochs = 20
model = AutoEncoder(X_ref, 128)
optimizer = nn.optim.Adam(nn.state.get_parameters(model))


def get_test_acc(model, Y, Y_hat) -> Tensor:
    return MSEloss(model(Y_hat), Y)


for epoch in range(1, epochs + 1):
    print(f"epoch {epoch}/{epochs}")
    for i in (t := trange(70)):
        GlobalCounters.reset()  # NOTE: this makes it nice for DEBUG=2 timing
        samples = Tensor.randint(1, high=X_train.shape[0])
        loss = model.train_step(optimizer, X_train[samples], Y_train[samples])
        t.set_description(f"loss: {loss.item():6.5f}")
    if i % 10 == 9:
        test_acc = get_test_acc(model, X_test, Y_test).item()

    # verify eval acc
    if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
        if test_acc >= target and test_acc != 100.0:
            print(colored(f"{test_acc=} >= {target}", "green"))
        else:
            raise ValueError(colored(f"{test_acc=} < {target}", "red"))

    print(f"test_accuracy: {test_acc:5.5f}")

plot_image(X_ref[0][0], rae(X_ref)[0][0])

AutoEncoder
