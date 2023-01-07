import numpy as np
import matplotlib.pyplot as plt

from question6part2 import grad

# this function implements the algorithm described in part 3
def detect_edges(gxy):
    num_row, num_col = gxy.shape
    taus = []
    taus.append(np.sum(gxy) / (num_row * num_col)) # tau_0 is assigned the average

    # tau_1 is assigned the value computed using the method described in the question
    taus.append(update_gradient(gxy, 1, taus))

    i = 1
    epsilon = 1 / (10 ** 10)
    # we then continue to compute a new tau until the below condition is satisfied
    while abs(taus[i] - taus[i-1]) >= epsilon:
        taus.append(update_gradient(gxy, i+1, taus))
        i += 1
    # the last computed tau is the final tau
    tau = taus[-1]

    edge_image = []
    # we must compare every entry to tau, and assgin it white or black accordingly
    for i in range(num_row):
        for j in range(num_col):
            if gxy[i, j] <= tau:
                edge_image.append(0)
            else:
                edge_image.append(255)

    edge_image = np.array(edge_image)
    edge_image = edge_image.reshape(num_row, num_col)

    return edge_image


def update_gradient(gxy, i, taus):
    # this function works by splitting up entries to those above the tau and
    # those below then computes the mean of each group and returns the mean of
    # those two values
    num_row, num_col = gxy.shape
    low_class = []
    high_class = []
    for j in range(num_row):
        for k in range(num_col):
            if gxy[j,k] < taus[i-1]:
                low_class.append(gxy[j,k])
            else:
                high_class.append(gxy[j,k])

    mL = np.sum(low_class) / len(low_class)
    mH = np.sum(high_class) / len(high_class)
    return (mL + mH) / 2


edges = detect_edges(grad)

plt.imshow(edges, cmap='gray')
