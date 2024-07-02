# %%
import glob
from helpers import displacement_2_tensor, img_2_tensor
from tinygrad import Tensor
import numpy as np

# %%
n = str(0).zfill(3)
ref_path = "dataset/Images/ref/simple_rotation_img_{}.tif"
mod_path = "dataset/Images/mod/simple_rotation_img_{}.tif"
x_disp_path = "dataset/Displacement_data/Ux/Image_{}_Ux.csv"
y_disp_path = "dataset/Displacement_data/Uy/Image_{}_Uy.csv"

n_exemple = sum(1 for _ in glob.glob("dataset/Images/mod/*"))
ds = Tensor.empty(n_exemple, 4, 257, 257)

ref = img_2_tensor(ref_path.format(str(0).zfill(3)))

for file_num in range(1, n_exemple):

    mod = img_2_tensor(mod_path.format(str(file_num).zfill(3)))
    # --> (1, 1, 257, 257)
    x_disp = displacement_2_tensor(x_disp_path.format(str(file_num).zfill(3)))
    y_disp = displacement_2_tensor(y_disp_path.format(str(file_num).zfill(3)))
    # --> (1, 257, 257)
    ds[file_num - 1] = ref.stack(
        mod.expand(1, -1, -1),
        x_disp.expand(1, -1, -1),
        y_disp.expand(1, -1, -1),
        dim=1,
    )[0]
    # --> (1, 1, 257, 257) + (1, 257, 257).expend ==> (1, 2, 257, 257)

# save to disk a (n_exemple, 2, 257, 257) tensor
np.save("dataset", ds.numpy())
# %%
