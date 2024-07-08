from PIL import Image
from tinygrad import Tensor, dtypes
import numpy as np
import pandas as pd
from scipy.ndimage import rotate
import matplotlib.pyplot as plt


def img_2_tensor(filename, norm=True):
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
    return Tensor((pd.read_csv(filename, sep=",", header=None).values + 1) / 2)


def rotate_vectors(x_matrix, y_matrix, angle_degrees):
    angle_radians = np.radians(angle_degrees)

    # Create the rotation matrix
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

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


def rotate_image(image, angle):
    # Rotate the image using scipy's rotate function
    rotated_image = rotate(image, angle, reshape=False)
    return rotated_image


def plot_image(origin_tensor, output_tensor, figsize=(10, 10)):
    f = plt.figure(figsize=figsize)
    ax = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    img = (origin_tensor * 255).cast(dtypes.int32).numpy()
    ax.imshow(img, cmap="gray")
    ax.set_title("Original Image")

    plt.subplot(1, 2, 2)
    img = (output_tensor * 255).cast(dtypes.int32).numpy()
    ax2.imshow(img, cmap="gray")
    ax2.set_title("Output Image")
    plt.show()


def plot_displacement(disp_tensor, figsize=(10, 10)):
    f = plt.figure(figsize=figsize)
    ax = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    img = disp_tensor
    ax.imshow(img, cmap="gray")
    ax.set_title("Original Image")
