# %%
from typing import List, Callable
from tinygrad import Tensor, TinyJit, nn, GlobalCounters
from tinygrad.nn import optim
from tinygrad.helpers import getenv, colored
from tqdm import trange
from dataloader import load
from helpers import plot_image
import numpy as np
from tinygrad.nn.state import get_parameters
from itertools import chain

QUICK = False

import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.helpers import CI
from tinygrad.engine.jit import TinyJit


def train(
    model,
    X_train,
    Y_train,
    optim,
    steps,
    BS=128,
    lossfn=lambda out, y: out.sparse_categorical_crossentropy(y),
    transform=lambda x: x,
    target_transform=lambda x: x,
    noloss=False,
    allow_jit=True,
):

    def train_step(x, y):
        # network
        out = model.forward(x) if hasattr(model, "forward") else model(x)
        loss = lossfn(out, y)
        optim.zero_grad()
        loss.backward()
        if noloss:
            del loss
        optim.step()
        if noloss:
            return (None, None)
        return loss.realize()

    if allow_jit:
        train_step = TinyJit(train_step)

    with Tensor.train():
        losses, accuracies = [], []
        for i in (t := trange(steps, disable=CI)):
            samp = Tensor.randint(BS, high=X_train.shape[0])
            x = X_train[samp]
            y = Y_train[samp]
            loss = train_step(x, y)
            # printing
            if not noloss:
                loss = loss.numpy()
                losses.append(loss)
                t.set_description("loss %.2f" % (loss))
    return [losses]


class conv:

    def __init__(
        self,
        batchNorm,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=None,
        stride=1,
    ):
        if padding:
            pad = padding
        else:
            pad = (kernel_size - 1) // 2
        if batchNorm:
            self.layers = [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=pad,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                Tensor.leakyrelu,
            ]
        else:
            self.layers = [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=pad,
                    bias=True,
                ),
                Tensor.leakyrelu,
            ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


class deconv:

    def __init__(self, in_planes, out_planes):
        self.layers = [
            nn.ConvTranspose2d(
                in_planes, out_planes, kernel_size=5, stride=2, padding=2, bias=True
            ),
            Tensor.leakyrelu,
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)


class Upsample:
    def __init__(self, scale_factor: int, mode: str = "nearest") -> None:
        assert mode == "nearest"  # only mode supported for now
        self.mode = mode
        self.scale_factor = scale_factor

    def __call__(self, x: Tensor) -> Tensor:
        assert len(x.shape) > 2 and len(x.shape) <= 5
        (b, c), _lens = x.shape[:2], len(x.shape[2:])
        tmp = x.reshape([b, c, -1] + [1] * _lens) * Tensor.ones(
            *[1, 1, 1] + [self.scale_factor] * _lens
        )
        return (
            tmp.reshape(list(x.shape) + [self.scale_factor] * _lens)
            .permute(
                [0, 1]
                + list(
                    chain.from_iterable([[y + 2, y + 2 + _lens] for y in range(_lens)])
                )
            )
            .reshape([b, c] + [x * self.scale_factor for x in x.shape[2:]])
        )


class Upsample2:
    def __init__(self, channels):
        self.conv = conv(False, channels, channels, 4, padding=0)

    def __call__(self, x):
        bs, c, py, px = x.shape
        x = (
            x.reshape(bs, c, py, 1, px, 1)
            .expand(bs, c, py, 2, px, 2)
            .reshape(bs, c, py * 2, px * 2)
        )
        return self.conv(x)


class FlowNetS:
    expansion = 1

    def __init__(self, batchNorm=True, input_channels=2, training=True):

        self.training = training
        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, input_channels, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=1)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 5, 2, 2, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 5, 2, 2, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 5, 2, 2, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 5, 2, 2, bias=False)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         if m.bias is not None:
        #             init.uniform_(m.bias)
        #         init.xavier_uniform_(m.weight)

        #     if isinstance(m, nn.ConvTranspose2d):
        #         if m.bias is not None:
        #             init.uniform_(m.bias)
        #         init.xavier_uniform_(m.weight)
        #         # init_deconv_bilinear(m.weight)
        self.upsample1 = Upsample2(2)

    def __call__(self, x: Tensor) -> Tensor:

        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = out_conv5.cat(out_deconv5, flow6_up, dim=1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)

        concat4 = out_conv4.cat(out_deconv4, flow5_up, dim=1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)

        concat3 = out_conv3.cat(out_deconv3, flow4_up, dim=1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = out_conv2.cat(out_deconv2, flow3_up, dim=1)
        flow2 = self.predict_flow2(concat2)

        if not self.training:
            return self.upsample1(flow2), flow3, flow4, flow5, flow6
        else:
            return self.upsample1(flow2)

    def save(self, filename):
        with open(filename + ".npy", "wb") as f:
            for par in get_parameters(self):
                # if par.requires_grad:
                np.save(f, par.numpy())


def MSEloss(y_hat, y):
    return ((y_hat - y) ** 2).sum().mean()


if __name__ == "__main__":

    X_train, Y_train, X_test, Y_test = load()
    # TODO: remove this when HIP is fixed
    X_train, X_test = X_train.float(), X_test.float()
    samples = Tensor.randint(1, high=X_train.shape[0])
    model = FlowNetS(input_channels=2)

    BS = 1
    steps = len(X_train) // BS

    def train_step(opt) -> Tensor:
        with Tensor.train():
            opt.zero_grad()
            samples = Tensor.randint(10, high=X_train.shape[0])
            # TODO: this "gather" of samples is very slow. will be under 5s when this is fixed
            loss = MSEloss(model(X_train[samples])[0], Y_train[samples]).backward()
            opt.step()
            return loss

    def get_test_acc() -> Tensor:
        return MSEloss(model(X_test), Y_test)

    params = get_parameters(model)
    [x.gpu() for x in params]
    lrs = [1e-4, 1e-5] if QUICK else [1e-3, 1e-4, 1e-5, 1e-5]
    epochss = [2, 1] if QUICK else [13, 3, 3, 1]

    for lr, epochs in zip(lrs, epochss):
        optimizer = optim.Adam(nn.state.get_parameters(model), lr=lr)
        for epoch in range(1, epochs + 1):
            # first epoch without augmentation
            train(
                model,
                X_train,
                Y_train,
                optimizer,
                steps=steps,
                lossfn=MSEloss,
                BS=BS,
            )
            accuracy = get_test_acc().numpy()
            print(f"accuracy : {accuracy:.0f}")
            model.save(f"checkpoints/checkpoint{accuracy:.0f}")

    # for lr, epochs in zip(lrs, epochss):
    #     optimizer = nn.optim.Adam(nn.state.get_parameters(model), lr=lr)
    #     for epoch in range(1, epochs + 1):
    #         print(f"epoch {epoch}/{epochs}")
    #         test_acc = float("nan")
    #         for i in (t := trange(70)):
    #             GlobalCounters.reset()  # NOTE: this makes it nice for DEBUG=2 timing
    #             loss = train_step(optimizer)
    #             t.set_description(f"loss: {loss.item():6.5f}")
    #             if i % 10 == 9:
    #                 test_acc = get_test_acc().item()

    #             # verify eval acc
    #             if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
    #                 if test_acc >= target and test_acc != 100.0:
    #                     print(colored(f"{test_acc=} >= {target}", "green"))
    #                 else:
    #                     raise ValueError(colored(f"{test_acc=} < {target}", "red"))

    #         print(f"test_accuracy: {test_acc:5.5f}")
    #         model.save(f"checkpoints/checkpoint{test_acc * 1e6:.0f}")

    # plot_image(X_train[samples], model(X_train[samples]))

# %%
