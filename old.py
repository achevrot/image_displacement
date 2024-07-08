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
from flow_model import FlowNetS

# %%


model = FlowNetS(training=True)
model.load("checkpoints/checkpoint0.1531_3")

# %%

X_train, Y_train, X_test, Y_test = load()
samples = Tensor.randint(1, high=X_test.shape[0])
X_test[samples][0, 0, ...]

# %%

plot_image(
    model(X_test[samples])[2][0, 0, ...],
    model(X_test[samples])[2][0, 1, ...],
    figsize=(20, 20),
)
# %%
plot_image(Y_test[samples][0, 0, ...], Y_test[samples][0, 1, ...], figsize=(20, 20))


# %%

X_train, Y_train, X_test, Y_test = load()
plot_image(Y_test[samples][0, 0, ...], Y_test[samples][0, 1, ...], figsize=(20, 20))
# %%

t = Y_test.numpy()
# %%

X_train, Y_train, X_test, Y_test = load()
samples = Tensor.randint(1, high=X_train.shape[0])

startScale = 4
numScales = 4
loss_weights = Tensor([(0.32 / 2**scale) for scale in range(numScales)])

div_flow = 0.05
assert len(loss_weights) == numScales

multiScales = [
    X_train[samples].avg_pool2d(startScale * (2**scale), startScale * (2**scale)).pad((None, None, (0, 1), (0, 1)))
    for scale in range(numScales)
]

multiScales
# %%
X_train[samples]
# %%
model(X_train[samples])

# %%
