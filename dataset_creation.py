# %%
from PIL import Image
from tinygrad import Tensor
import numpy as np
import pandas as pd
import glob


def img_2_tensor(filename):
    img = Image.open(filename)
    aspect_ratio = img.size[0] / img.size[1]
    img = img.resize(
        (
            int(257 * max(aspect_ratio, 1.0)),
            int(257 * max(1.0 / aspect_ratio, 1.0)),
        )
    )
    img = np.array(img)
    img = img.astype(np.float32).reshape(1, 257, 257)
    img /= 255.0

    return Tensor(img)


def displacement_2_tensor(filename):
    return Tensor(pd.read_csv(filename, sep=",", header=None).values)


def rotate_vectors(x_matrix, y_matrix, angle_degrees):

    angle_radians = np.radians(angle_degrees)

    # Create the rotation matrix
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    rotation_matrix = np.array(
        [[cos_theta, -sin_theta], [sin_theta, cos_theta]]
    )

    # Stack the x and y matrices vertically to create a matrix of shape (2, 256, 256)
    vectors = np.stack((x_matrix, y_matrix))

    # Reshape the vectors matrix to shape (2, 256*256)
    vectors_reshaped = vectors.reshape(2, -1)

    # Apply the rotation matrix to the reshaped vectors matrix
    rotated_vectors = np.dot(rotation_matrix, vectors_reshaped)

    # Reshape the rotated vectors back to shape (2, 256, 256)
    rotated_vectors_reshaped = rotated_vectors.reshape(2, 256, 256)

    # Extract the new x and y matrices from the rotated vectors
    new_x_matrix = rotated_vectors_reshaped[0]
    new_y_matrix = rotated_vectors_reshaped[1]

    return new_x_matrix, new_y_matrix


# %%


n_exemple = sum(
    1 for _ in glob.glob("Dataset/images/simple_rotation_img_*.tif")
)
t = Tensor.empty(n_exemple, 1, 257, 257)

for i, file in enumerate(glob.glob("Dataset/images/simple_rotation_img_*.tif")):
    t[i] = img_2_tensor(file)

np.save("X_dataset", t.numpy())
# %%

n_exemple = sum(1 for _ in glob.glob("Dataset/labels_x/*.csv"))

t = Tensor.empty(n_exemple, 1, 256, 256)

for i, file in enumerate(glob.glob("Dataset/labels_x/*.csv")):
    t[i] = displacement_2_tensor(file)

np.save("Y_dataset", t.numpy())
# %%

from tinygrad.nn.datasets import mnist

X_train, Y_train = Tensor(np.load("X_dataset.npy")), Tensor(
    np.load("Y_dataset.npy")
).reshape(-1, 1, 256, 256)
print(X_train.shape, X_train.dtype, Y_train.shape, Y_train.dtype)
# %%
