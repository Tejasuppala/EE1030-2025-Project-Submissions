import numpy as np
from PIL import Image
data_dir = "../hybrid_c_python/data/"
# --- read shapes ---
with open(data_dir + "shape.txt", "r") as f:
    dims = [list(map(int, line.split())) for line in f]  # shape: [[m1,n1],[m2,n2],[m3,n3]]


files = [data_dir + "a_out.bin", data_dir + "b_out.bin", data_dir + "c_out.bin"]
channels = []

for idx, (m, n) in enumerate(dims):
    raw = np.fromfile(files[idx], dtype=np.float32, count=m*n)
    arr = raw.reshape((m, n))
    channels.append(arr)

# stack into RGB
img = np.stack(channels, axis=2)     # shape (m, n, 3)

# clamp + convert to uint8
img = np.clip(img, 0, 255).astype(np.uint8)

save_dir = "../../figs/"
# save
Image.fromarray(img).save(save_dir + "x.png")
print("Saved x.png")
