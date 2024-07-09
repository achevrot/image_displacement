# %%
from tinygrad import Tensor, nn, GlobalCounters
from tinygrad.nn import optim
from tinygrad.helpers import getenv
from tqdm import trange
from dataloader import load
import numpy as np
from tinygrad.nn.state import get_parameters
from tinygrad.nn.state import safe_save, get_state_dict
from itertools import chain
from conv_modules import *

GPU = getenv("GPU")
QUICK = getenv("QUICK")

BS = 1
Tensor.manual_seed(42)


class Upsample:
    def __init__(self, scale_factor: int, mode: str = "nearest") -> None:
        assert mode == "nearest"  # only mode supported for now
        self.mode = mode
        self.scale_factor = scale_factor

    def __call__(self, x: Tensor) -> Tensor:
        assert len(x.shape) > 2 and len(x.shape) <= 5
        (b, c), _lens = x.shape[:2], len(x.shape[2:])
        tmp = x.reshape([b, c, -1] + [1] * _lens) * Tensor.ones(*[1, 1, 1] + [self.scale_factor] * _lens)
        return (
            tmp.reshape(list(x.shape) + [self.scale_factor] * _lens)
            .permute([0, 1] + list(chain.from_iterable([[y + 2, y + 2 + _lens] for y in range(_lens)])))
            .reshape([b, c] + [x * self.scale_factor for x in x.shape[2:]])
        )[:, :, :-3, :-3]


class Upsample2:
    def __init__(self, scale_factor=2):
        self.scale_factor = scale_factor

    def __call__(self, x):
        bs, c, py, px = x.shape
        x = (
            x.reshape(bs, c, py, 1, px, 1)
            .expand(bs, c, py, self.scale_factor, px, self.scale_factor)
            .reshape(bs, c, py * self.scale_factor, px * self.scale_factor)
        )
        if self.scale_factor == 1:
            return x
        else:
            return x[:, :, :-1, :-1]


class FlowNetS:
    expansion = 1

    def __init__(self, batchNorm=True, input_channels=2, training=True):

        self.training = training
        self.batchNorm = batchNorm
        self.conv0 = conv(self.batchNorm, input_channels, 64)
        self.conv1 = conv(self.batchNorm, 64, 64, stride=1)
        self.conv1_1 = conv(self.batchNorm, 64, 128)
        self.conv2 = conv(self.batchNorm, 128, 128, stride=2)
        self.conv2_1 = conv(self.batchNorm, 128, 128)
        self.conv3 = conv(self.batchNorm, 128, 256, stride=2)
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

        self.inter_conv5 = i_conv(self.batchNorm, 1026, 512)
        self.inter_conv4 = i_conv(self.batchNorm, 770, 256)
        self.inter_conv3 = i_conv(self.batchNorm, 386, 128)
        self.inter_conv2 = i_conv(self.batchNorm, 194, 64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(512)
        self.predict_flow4 = predict_flow(256)
        self.predict_flow3 = predict_flow(128)
        self.predict_flow2 = predict_flow(64)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 5, 2, 2, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 5, 2, 2, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 5, 2, 2, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 5, 2, 2, bias=False)
        # self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 5, 2, 2, bias=False)
        # self.upsampled_flow1_to_out = nn.ConvTranspose2d(2, 2, 5, 2, 2, bias=False)

        # weights are by default initialized with kaiming normals

        self.upsample1 = Upsample2(2)

    def __call__(self, x: Tensor) -> Tensor:

        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))

        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = out_conv5.cat(out_deconv5, flow6_up, dim=1)
        out_interconv5 = self.inter_conv5(concat5)
        flow5 = self.predict_flow5(out_interconv5)

        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)

        concat4 = out_conv4.cat(out_deconv4, flow5_up, dim=1)
        out_interconv4 = self.inter_conv4(concat4)
        flow4 = self.predict_flow4(out_interconv4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)

        concat3 = out_conv3.cat(out_deconv3, flow4_up, dim=1)
        out_interconv3 = self.inter_conv3(concat3)
        flow3 = self.predict_flow3(out_interconv3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = out_conv2.cat(out_deconv2, flow3_up, dim=1)
        out_interconv2 = self.inter_conv2(concat2)
        flow2 = self.predict_flow2(out_interconv2)
        # flow2 = self.upsampled_flow2_to_1(flow2)
        # flow2 = self.upsampled_flow1_to_out(flow2)

        if self.training:
            return self.upsample1(flow2), flow3, flow4, flow5, flow6
        else:
            return self.upsample1(flow2)

    def save(self, filename):
        with open(filename + ".npy", "wb") as f:
            for par in get_parameters(self):
                # if par.requires_grad:
                np.save(f, par.numpy())

    def load(self, filename):
        with open(filename + ".npy", "rb") as f:
            for par in get_parameters(self):
                # if par.requires_grad:
                try:
                    par.numpy()[:] = np.load(f)
                    par.gpu()
                except:
                    print("Could not load parameter")


def MSEloss(y_hat, y):
    # if mean:
    return ((y_hat - y) ** 2).mean()


# else:
#     batch_size = y_hat.shape[0]
#     return ((y_hat - y) ** 2).sum() / batch_size


def MSEloss2(y_hat, y, mean=False, BS=1):
    if mean:
        return y_hat.sub(y).pow(2).mean()
    else:
        return y_hat.sub(y).pow(2).sum() / BS


def multiscaleEPE(network_output, target_flow, weights=None):

    startScale = 4
    numScales = len(network_output)
    loss_weights = Tensor([(0.32 / 2**scale) for scale in range(numScales)])
    assert len(loss_weights) == numScales
    multiScales = [
        target_flow.avg_pool2d(startScale * (2**scale), startScale * (2**scale)).pad((None, None, (0, 1), (0, 1)))
        for scale in range(numScales)
    ]

    for i, output in enumerate(network_output):
        if i == 0:
            mse = MSEloss(output, target_flow)
            loss = mse.mul(loss_weights[i])
        else:
            mse = MSEloss(output, multiScales[i - 1])
            loss = loss.add(mse.mul(loss_weights[i]))

    return loss


if __name__ == "__main__":

    X_train, Y_train, X_test, Y_test = load()
    # TODO: remove this when HIP is fixed
    # X_train, X_test = X_train.float(), X_test.float()
    # X_train_0 = X_train[:, 0, ...].stack(X_train[:, 0, ...], X_train[:, 0, ...], dim=1)
    # X_train_1 = X_train[:, 1, ...].stack(X_train[:, 1, ...], X_train[:, 1, ...], dim=1)
    # X_train = X_train_0.cat(X_train_1, dim=1)
    # model = FlowNetS(input_channels=6)
    model = FlowNetS(input_channels=2)
    # samples = Tensor.randint(BS, high=X_train.shape[0])
    # TODO: this "gather" of samples is very slow. will be under 5s when this is fixed

    steps = len(X_train) // BS

    def train_step(opt) -> Tensor:
        with Tensor.train():
            opt.zero_grad()
            samples = Tensor.randint(BS, high=X_train.shape[0])
            # TODO: this "gather" of samples is very slow. will be under 5s when this is fixed
            loss = multiscaleEPE(model(X_train[samples]), Y_train[samples]).backward()
            # loss = MSEloss2(model(X_train[samples])[0], Y_train[samples]).backward()
            opt.step()
            return loss

    def get_test_acc() -> Tensor:
        return ((model(X_test)[0][:, 0, ...] - Y_test[:, 0, ...]) ** 2).mean() ** (1 / 2)

    params = get_parameters(model)
    [x.gpu() for x in params]

    lrs = [1e-4, 1e-5] if QUICK else [1e-3, 1e-4, 1e-5, 1e-5]
    epochss = [2, 1] if QUICK else [30, 15, 15, 15]
    total_ep = 0
    for lr, epochs in zip(lrs, epochss):
        optimizer = optim.Adam(nn.state.get_parameters(model), lr=lr)
        for epoch in range(1, epochs + 1):
            total_ep += 1
            print(f"epoch {epoch}/{epochs}")
            test_acc = float("nan")
            for i in (t := trange(steps)):
                GlobalCounters.reset()  # NOTE: this makes it nice for DEBUG=2 timing
                loss = train_step(optimizer)
                t.set_description(f"loss: {loss.item():6.5f}")

            accuracy = get_test_acc().numpy()
            print(f"accuracy : {accuracy:.4f}")
            state_dict = get_state_dict(model)
            safe_save(state_dict, f"checkpoints/checkpoint{accuracy:.4f}_{total_ep}.safetensors")

# %%

# get_test_acc().numpy()

# from helpers import img_2_tensor, plot_image

# samples = Tensor.randint(BS, high=X_test.shape[0])

# plot_image(
#     model(X_test[samples])[0][0, 0, ...],
#     model(X_test[samples])[0][0, 1, ...],
#     figsize=(20, 20),
# )
# # %%

# plot_image(Y_train[samples][0, 0, ...], Y_train[samples][0, 1, ...], figsize=(20, 20))
# # %%

# from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict

# state_dict = get_state_dict(model)
# safe_save(state_dict, "model.safetensors")

# %%
