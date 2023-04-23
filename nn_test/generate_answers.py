import numpy as np

# Demo numpy reshaping
print(np.reshape(np.array([1, 0, 0, 1]), [2, 2]))
print(np.array([1, 2]).__matmul__(np.reshape(np.array([1, 0, 0, 1]), [2, 2], order="F")))

f = np.genfromtxt("VDMEM.txt", dtype=np.int32)

# Do fully connected layer calculation
input_vector = f[:256]
weight_matrix = np.reshape(f[256:], [256,256], order="F")

answer = input_vector.__matmul__(weight_matrix)
print(answer.shape, answer.dtype)
print(answer)

print("first")
print(f[256::256])
print(np.array([79]*256).__mul__(f[256::256]))
