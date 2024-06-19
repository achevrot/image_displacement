# %%
# model based off https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
from typing import List, Callable
from tinygrad import Tensor, TinyJit, nn, GlobalCounters
from tinygrad.helpers import getenv, colored
from tqdm import trange
from dataloader import load


class Encoder:

    def __init__(self, latent_dim=64, base_channel_size=32):

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
    def __init__(self, latent_dim=64, base_channel_size=32):

        c_hid = base_channel_size
        self.linear: List[Callable[[Tensor], Tensor]] = [
            nn.Linear(latent_dim, 4 * 16 * 16 * c_hid)
        ]
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
            nn.ConvTranspose2d(
                c_hid, 1, kernel_size=5, output_padding=2, padding=3, stride=2
            ),
        ]

    def __call__(self, x: Tensor) -> Tensor:
        x = x.sequential(self.linear)
        x = x.reshape(x.shape[0], -1, 16, 16)
        x = x.sequential(self.net)
        return x


class AutoEncoder:
    def __init__(self):
        self.layers = [Encoder(), Decoder()]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)


def MSEloss(y_hat, y):
    return ((y_hat - y) ** 2).mean()


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load()
    # TODO: remove this when HIP is fixed
    X_train, X_test = X_train.float(), X_test.float()
    model = AutoEncoder()
    opt = nn.optim.Adam(nn.state.get_parameters(model))
    samples = Tensor.randint(1, high=X_train.shape[0])

    def train_step() -> Tensor:
        with Tensor.train():
            opt.zero_grad()
            samples = Tensor.randint(1, high=X_train.shape[0])
            # TODO: this "gather" of samples is very slow. will be under 5s when this is fixed
            loss = MSEloss(model(X_train[samples]), X_train[samples]).backward()
            opt.step()
            return loss

    def get_test_acc() -> Tensor:
        return (model(X_test).argmax(axis=1) == X_test).mean() * 100

    test_acc = float("nan")
    for i in (t := trange(70)):
        GlobalCounters.reset()  # NOTE: this makes it nice for DEBUG=2 timing
        loss = train_step()
        if i % 10 == 9:
            test_acc = get_test_acc().item()
        t.set_description(
            f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%"
        )

    # verify eval acc
    if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
        if test_acc >= target and test_acc != 100.0:
            print(colored(f"{test_acc=} >= {target}", "green"))
        else:
            raise ValueError(colored(f"{test_acc=} < {target}", "red"))

# %%
