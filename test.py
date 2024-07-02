from typing import Optional
from tinygrad import nn, Tensor
from tinygrad.helpers import make_pair


def corr_2d(
    img_ref: Tensor,
    img_mod: Tensor,
    weights,
    bias: Optional[Tensor] = None,
    kernel_size=1,
    patch_size=1,
    groups=1,
    stride=1,
    dilation=1,
    padding=0,
) -> Tensor:
    (bs, cin_), (cout, cin) = (
        img_ref.shape[:2],
        img_mod.shape[:2],
    )

    HW = make_pair(kernel_size)
    assert groups * cin == cin_ and len(img_ref.shape) == len(
        img_mod.shape
    ), f"Input Tensor shape {img_ref.shape} does not match the shape of the img_mods {img_mod.shape}. ({groups*cin} vs. {cin_})"
    if isinstance(padding, (tuple, list)):
        assert len(padding) == 2 * len(HW) or len(padding) == len(
            HW
        ), f"Expected padding of length {2*len(HW)} or {len(HW)}, but got {len(padding)} for tensor of shape {img_ref.shape}"
    padding_ = (
        [padding] * 2 * len(HW)
        if isinstance(padding, int)
        else (
            padding
            if len(padding) == 2 * len(HW)
            else [p for p in padding for _ in range(2)][::-1]
        )
    )
    # conv2d is a pooling op (with padding)
    x = img_ref.pad2d(padding_)._pool(
        HW, stride, dilation
    )  # (bs, groups*cin, oy, ox, H, W)
    rcout, oyx = cout // groups, x.shape[2 : -len(HW)]
    x = (
        x.reshape(bs, groups, cin, 1, *oyx, *HW)
        .expand(bs, groups, cin, rcout, *oyx, *HW)
        .permute(
            0,
            1,
            3,
            *[4 + i for i in range(len(oyx))],
            2,
            *[4 + len(oyx) + i for i in range(len(HW))],
        )
    )
    y = img_mod.pad2d(padding_)._pool(
        HW, stride, dilation
    )  # (bs, groups*cin, oy, ox, H, W)

    y = (
        y.reshape(bs, groups, cin, 1, *oyx, *HW)
        .expand(bs, groups, cin, rcout, *oyx, *HW)
        .permute(
            0,
            1,
            3,
            *[4 + i for i in range(len(oyx))],
            2,
            *[4 + len(oyx) + i for i in range(len(HW))],
        )
    )
    print("x : ", x)
    print("y : ", y)
    print(x * y)
    ret = (
        (x * y)
        .sum([-1 - i for i in range(1 + len(oyx))], keepdim=True)
        .reshape(bs, cout, *oyx)
    )

    ret = (
        (x * weight.reshape(1, groups, rcout, *[1 for _ in range(len(oyx))], cin, *HW))
        .sum([-1 - i for i in range(1 + len(oyx))], keepdim=True)
        .reshape(bs, cout, *oyx)
    )

    return (
        ret
        if bias is None
        else ret.add(bias.reshape(1, -1, *[1 for _ in range(len(HW))]))
    )


x = Tensor.arange(16).reshape(1, 1, 4, 4)
y = Tensor.arange(16).reshape(1, 1, 4, 4)
w = Tensor.arange(16).reshape(1, 1, 2, 2)

corr_2d(, w, 2)
