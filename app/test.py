from matplotlib.patches import Circle  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from scipy import signal  # type: ignore
import cv2  # type: ignore

from .config import SEARCHING_PX_PER_M
from .utils.debug import export_image
from .utils.math import Vec2, rbf_kernel


def to_position(coords):
    (x, y) = coords

    px = (x + 0.5) / SEARCHING_PX_PER_M + 3.5
    py = (y + 0.5) / SEARCHING_PX_PER_M

    return Vec2(px, py)


fig, ax = plt.subplots(3)

kernel = rbf_kernel(31, 6.0) - 0.5 * rbf_kernel(31, 3.5)
ax[0].imshow(kernel)
export_image("kernel_detect", kernel)

data = np.load("probability_map.npy")
print(f"Map size is {data.shape}")

ax[1].imshow(data)

conv = cv2.filter2D(data, -1, kernel)

max = np.argmax(conv, axis=None)
(x, y) = np.unravel_index(max, conv.shape)
position = to_position((int(x), int(y)))

print(f"Found max index at {(x, y)}, position {position}")

ax[2].imshow(conv)
circ = Circle((float(y), float(x)), 2, color="red")
ax[2].add_patch(circ)

plt.show()

# == Other stuff == #

# kernel = laplace(np.shape(100, 100), 1)

# export_image("detect_kernel", kernel, flip=False)

# # with open("app/output/plot_7.json", "r+") as f:
# #     data1 = json.load(f)
# # with open("app/output/plot_78.json", "r+") as f:
# #     data2 = json.load(f)

# # # cwt_peaks = signal.find_peaks_cwt(data, np.arange(1, 10))
# # # print(len(data1[30:]))
# # map = np.outer(data1[100:], data2[100:])
# # plt.imshow(map)
# # # plt.plot(data1[0:])
# # plt.show()
# # for peak in cwt_peaks:
# #     if data[peak] > 100:
# #         plt.axvline(x=peak, color="r", linestyle="--")
# # plt.show()


# # x = [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
# # y = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
# # tab = np.outer(x, y)
# # print(tab)
# XXX = 1
# YYY = 1
# list = []
# # for i in range(11):
# #     for j in range(11):
# #         if j % 2 == 0:
# #             a = 1
# #         else:
# #             a = -1
# #         list.append((XXX + (j - 5) * 0.125, YYY + (i - 5) * 0.125))

# # a = 1
# # for j in range(11):
# #     TEST2 = (2 * ((j + 1) % 2) - 1) * ((j + 1) // 2)
# #     # print(TEST2)
# #     for i in range(11):
# #         TEST = a * (i - 5) * 0.125
# #         print(XXX + TEST, YYY + TEST2 * 0.125)
# #         if i == 10:
# #             a = -a


# # for j in range(11):
# #     test2 = (2 * ((j + 1) % 2) - 1) * ((j + 1) // 2)

# import random

# # Création d'une matrice 6x6 avec des chiffres aléatoires entre 1 et 9
# matrice = [[random.randint(1, 9) for _ in range(6)] for _ in range(6)]


# a = matrice
# b = 3 * a
# threshold = np.median(a)
# print(threshold)
# c = (a > threshold) * a
# print(c)


gauss = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

signal.convolve2d(laplacian, gauss)
