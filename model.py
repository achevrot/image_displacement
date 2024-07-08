# %%
# model based off https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
from typing import List, Callable
from tinygrad import Tensor, TinyJit, nn, GlobalCounters
from tinygrad.helpers import getenv, colored
from tqdm import trange
from dataloader import load
from helpers import plot_image
import numpy as np
from tinygrad.nn.state import get_parameters

QUICK = getenv("QUICK")


class Encoder:

    def __init__(self, latent_dim, base_channel_size=32):

        c_hid = base_channel_size
        self.layers: List[Callable[[Tensor], Tensor]] = [
            nn.Conv2d(1, c_hid, 5, padding=4, stride=2),
            Tensor.relu,
            nn.Conv2d(c_hid, c_hid, 5, padding=4),
            Tensor.relu,
            nn.Conv2d(c_hid, 2 * c_hid, 5, padding=4, stride=2),  # -> 64x64
            Tensor.relu,
            nn.Conv2d(2 * c_hid, 2 * c_hid, 5, padding=4),
            Tensor.relu,
            nn.Conv2d(2 * c_hid, 2 * c_hid, 5, padding=4, stride=2),
            Tensor.relu,
            nn.Conv2d(2 * c_hid, 2 * c_hid, 5, padding=4),
            Tensor.relu,
            nn.Conv2d(2 * c_hid, 4 * c_hid, 5, padding=4, stride=2),
            Tensor.relu,
            nn.Conv2d(4 * c_hid, 4 * c_hid, 5, padding=4),
            Tensor.relu,
            nn.Conv2d(4 * c_hid, 4 * c_hid, 5, padding=4, stride=2),
            Tensor.relu,
            lambda x: x.flatten(1),  # noqa: E731
            nn.Linear(4 * c_hid * 16 * 16, latent_dim),
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)


class Decoder:
    def __init__(self, latent_dim, base_channel_size=32):

        c_hid = base_channel_size
        self.linear: List[Callable[[Tensor], Tensor]] = [nn.Linear(latent_dim, 4 * 16 * 16 * c_hid)]
        self.net = [
            nn.ConvTranspose2d(
                4 * c_hid,
                4 * c_hid,
                kernel_size=5,
                output_padding=2,
                padding=2,
                stride=2,
            ),
            Tensor.relu,
            nn.Conv2d(4 * c_hid, 2 * c_hid, kernel_size=5, padding=2),
            Tensor.relu,
            nn.ConvTranspose2d(
                2 * c_hid,
                2 * c_hid,
                kernel_size=5,
                output_padding=2,
                padding=3,
                stride=2,
            ),  # 8x8 => 16x16
            Tensor.relu,
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=5, padding=3),
            Tensor.relu,
            nn.ConvTranspose2d(
                2 * c_hid,
                c_hid,
                kernel_size=3,
                output_padding=2,
                padding=3,
                stride=2,
            ),
            Tensor.relu,
            nn.Conv2d(c_hid, c_hid, kernel_size=5, padding=1),
            Tensor.relu,
            nn.ConvTranspose2d(c_hid, 1, kernel_size=5, output_padding=2, padding=3, stride=2),
        ]

    def __call__(self, x: Tensor) -> Tensor:
        x = x.sequential(self.linear)
        x = x.reshape(x.shape[0], -1, 16, 16)
        x = x.sequential(self.net)
        return x


class AutoEncoder:
    def __init__(self, latent_dim=64):
        self.ref_enc = [
            Encoder(latent_dim),
            Decoder(latent_dim),
            # lambda x: x.sigmoid(),
        ]

    def __call__(self, x: Tensor) -> Tensor:

        return x.sequential(self.layers)

    def save(self, filename):
        with open(filename + ".npy", "wb") as f:
            for par in get_parameters(self):
                # if par.requires_grad:
                np.save(f, par.numpy())


def MSEloss(y_hat, y):
    return ((y_hat - y) ** 2).sum(axis=[1, 2, 3]).mean(axis=[0])


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load()
    # TODO: remove this when HIP is fixed
    X_train, X_test = X_train.float(), X_test.float()
    model = AutoEncoder(128)

    def train_step(opt) -> Tensor:
        with Tensor.train():
            opt.zero_grad()
            samples = Tensor.randint(1, high=X_train.shape[0])
            # TODO: this "gather" of samples is very slow. will be under 5s when this is fixed
            loss = MSEloss(model(X_train[samples]), Y_train[samples]).backward()
            opt.step()
            return loss

    def get_test_acc() -> Tensor:
        return MSEloss(model(X_test), Y_test)

    lrs = [1e-4, 1e-5] if QUICK else [1e-3, 1e-4, 1e-5, 1e-5]
    epochss = [2, 1] if QUICK else [100, 50, 50, 50]

    for lr, epochs in zip(lrs, epochss):
        optimizer = nn.optim.Adam(nn.state.get_parameters(model), lr=lr)
        for epoch in range(1, epochs + 1):
            print(f"epoch {epoch}/{epochs}")
            test_acc = float("nan")
            for i in (t := trange(70)):
                GlobalCounters.reset()  # NOTE: this makes it nice for DEBUG=2 timing
                loss = train_step(optimizer)
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
            model.save(f"checkpoints/checkpoint{test_acc * 1e6:.0f}_{epoch}")

    plot_image(X_test[1][0], model(X_test)[1][0])
# %%
