import numpy as np

data_dir = "../hybrid_c_python/data/"
# files
in_files  = [data_dir + "a.bin", data_dir + "b.bin", data_dir + "c.bin"]
out_files = [data_dir + "a_out.bin", data_dir + "b_out.bin", data_dir + "c_out.bin"]

# read shape.txt (m n for each channel)
shape = []
with open("shape.txt", "r") as f:
    for _ in range(3):
        m, n = map(int, f.readline().split())
        shape.append((m, n))

def frobenius_error(orig, approx):
    return np.linalg.norm(orig - approx, 'fro')

for idx, ((m, n), infile, outfile) in enumerate(zip(shape, in_files, out_files), start=1):
    A  = np.fromfile(infile, dtype=np.float32, count=m*n).reshape(m, n)
    Ak = np.fromfile(outfile, dtype=np.float32, count=m*n).reshape(m, n)

    err = frobenius_error(A, Ak)

    normA = np.linalg.norm(A)

    print(f"Channel {idx}: shape {m}x{n}")
    print(f"  ||A - A_k||_F = {err:.4f}")
    print(f" ||A|| = {normA}")
