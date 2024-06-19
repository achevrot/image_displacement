# %%
import glob
from helpers import rotate_image, displacement_2_tensor, img_2_tensor
from tinygrad import Tensor
import numpy as np

# %%
n = str(0).zfill(3)
x_path = "Dataset/source/images/simple_rotation_img_{}.tif"
y_path = "Dataset/source/displacement_x/Image_{}_Ux.csv"

n_exemple = sum(1 for _ in glob.glob("Dataset/source/images/*"))
ds = Tensor.empty(n_exemple, 2, 257, 257)

for file_num in range(1, n_exemple):

    x_i = img_2_tensor(x_path.format(str(file_num).zfill(3)))
    # --> (1, 1, 257, 257)
    y_i = displacement_2_tensor(y_path.format(str(file_num).zfill(3)))
    # --> (1, 257, 257)
    ds[file_num] = x_i.stack(y_i.expand(1, -1, -1), dim=1)[0]
    # --> (1, 1, 257, 257) + (1, 257, 257).expend ==> (1, 2, 257, 257)

# save to disk a (n_exemple, 2, 257, 257) tensor
np.save("dataset", ds.numpy())
# %%
