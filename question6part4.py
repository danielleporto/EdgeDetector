import numpy
import matplotlib as plt
import cv2

from question5 import gray_puppies, gray_window
from question6part1 import get_gaussian_matrix
from question6part2 import gradient_magnitude
from question6part3 import detect_edges


# this code tests the code from parts 1 through 3
gauss = get_gaussian_matrix(10, 1)
blurred1 = cv2.filter2D(src=gray_puppies, ddepth=-1, kernel=gauss)

gradient1 = gradient_magnitude(blurred1)

final_edges1 = detect_edges(gradient1)

plt.imshow(final_edges1, cmap='gray')

blurred2 = cv2.filter2D(src=gray_window, ddepth=-1, kernel=gauss)

gradient2 = gradient_magnitude(blurred2)

final_edges2 = detect_edges(gradient2)

plt.imshow(final_edges2, cmap='gray')
