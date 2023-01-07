import numpy as np
import matplotlib.pyplot as plt

from question5 import gray_puppies

# this function convolves the input image with the proper matrices then squares
# and sums the resulting matrices
def gradient_magnitude(img):

    gx_mat = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gx = convolve(img, gx_mat)

    gy_mat = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gy = convolve(img, gy_mat)

    return np.sqrt(gx ** 2 + gy **2)


# this function performs convolution where mat1 is a matrix of any size
# and mat2 is a 3x3 matrix
def convolve(mat1, mat2):
    num_row, num_col = mat1.shape[:-1]
    result = []

    for i in range(1, num_row - 1):
        for j in range(1, num_col - 1):
            result.append(np.sum(mat1[i-1: i+2, j-1 : j+2] * mat2))      # we want this slice of mat1 to be the same size as mat2, i.e. 3x3

    result = np.array(result)
    result = result.reshape(num_row-2, num_col-2)
    return result


grad = gradient_magnitude(gray_puppies)

plt.imshow(grad)
