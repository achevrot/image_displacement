from tinygrad import nn, Tensor


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


class i_conv:
    def __init__(self, batchNorm, in_planes, out_planes, kernel_size=3, stride=1, bias=True) -> None:
        if batchNorm:
            self.layers = [
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    bias=bias,
                ),
                nn.BatchNorm2d(out_planes),
            ]
        else:
            self.layers = [
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    bias=bias,
                ),
            ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)


class deconv:

    def __init__(self, in_planes, out_planes, stride=2, padding=2):
        self.layers = [
            nn.ConvTranspose2d(
                in_planes,
                out_planes,
                kernel_size=5,
                stride=stride,
                padding=padding,
                bias=True,
            ),
            Tensor.leakyrelu,
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)
