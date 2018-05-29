import numpy as np


arr = np.array([[1/2, 1/2], [1/4, 3/4], [1/8, 7/8]])

result = np.zeros([3,3])

for i in range(arr.shape[0]):
    p = arr[i]
    for j in range(arr.shape[0]):
        q = arr[j]
        result[i, j] = np.sum(p * (np.log2(p / q)))



for i in range(arr.shape[0]):
    print('%7.4f%7.4f%7.4f'%(result[i,0], result[i, 1], result[i, 2]))