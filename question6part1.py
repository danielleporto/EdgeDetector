import numpy as np
import matplotlib.pyplot as plt


def get_gaussian_matrix(size, sigma):
    # first we make a vector with equally spaced entries
    mat = np.linspace(-1 * sigma, sigma, size)
    # then we use the Gaussian function on each entry
    mat2 = np.exp(-0.5 * np.square(mat) / np.square(sigma)) / (2 * np.pi * (sigma ** 2))
    # then we join the resulting vector with itself to create the Gaussian matrix
    gaussian = np.outer(mat2, mat2)
    return gaussian


mat1 = get_gaussian_matrix(10, 10)

colormap1 = plt.imshow(mat1, cmap='winter')
plt.colorbar(colormap1)

mat2 = get_gaussian_matrix(10, 1)

colormap2 = plt.imshow(mat2, cmap='winter')
plt.colorbar(colormap2)
