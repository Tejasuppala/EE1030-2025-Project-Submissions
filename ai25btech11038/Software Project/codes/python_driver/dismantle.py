from PIL import Image
import numpy as np

#input image ->
original = Image.open('../../figs/einstein.jpg')

#creating an np.array of the image
pixel_array = np.asarray(original) 

R, G, B = pixel_array[:,:,0], pixel_array[:,:,1], pixel_array[:,:,2]

#getting the truncated no.of singular values
k = int(input("To how many singular values should the matrix be truncated to: "))

out_dir = "../hybrid_c_python/data/"

with open(out_dir + "shape.txt", "w") as f:
    for i in [R, G, B]:
        f.write(f"{i.shape[0]} {i.shape[1]}\n")
  
R.astype(np.float32).tofile(out_dir + "a.bin")
G.astype(np.float32).tofile(out_dir + "b.bin")
B.astype(np.float32).tofile(out_dir +  "c.bin")
with open(out_dir + "k.txt", "w") as g:
    g.write(f"{k}")