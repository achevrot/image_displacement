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
from flow_model_SD import FlowNetS, MSEloss
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict

# %%


model = FlowNetS(input_channels=2)
state_dict = safe_load("checkpoints/checkpoint0.0397_71.safetensors")
load_state_dict(model, state_dict)

# %%

X_train, Y_train, X_test, Y_test = load()
samples = Tensor.randint(1, high=X_train.shape[0])
X_train[samples][0, 0, ...]

# %%

plot_image(
    model(X_train[samples])[0][0, 0, ...],
    model(X_train[samples])[0][0, 1, ...],
    figsize=(20, 20),
)
# %%
plot_image(Y_train[samples][0, 0, ...], Y_train[samples][0, 1, ...], figsize=(20, 20))


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

(((model(X_test)[0][:, 0, ...] - Y_test[:, 0, ...]) ** 2).mean() ** (1 / 2)).numpy()

# %%
MSEloss(model(X_test)[0][:, 0, ...], Y_test[:, 0, ...]).numpy()

# %%
