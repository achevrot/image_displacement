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


model = FlowNetS()
# model.load("checkpoints/checkpoint0.1669")

# %%

X_train, Y_train, X_test, Y_test = load()
samples = Tensor.randint(1, high=X_train.shape[0])
X_test[samples][0, 0, ...]

# %%

plot_image(
    model(X_train[samples])[0, 0, ...],
    model(X_train[samples])[0, 1, ...],
    figsize=(20, 20),
)
plot_image(Y_train[samples][0, 1, ...], Y_test[samples][0, 0, ...], figsize=(20, 20))


# %%
