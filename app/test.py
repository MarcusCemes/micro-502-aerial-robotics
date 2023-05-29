import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal.windows import gaussian
import cv2
# from scipy.ndimage.filters import laplace
# from scipy.ndimage import gaussian_laplace

# from utils.math import rbf_kernel


def rbf_kernel(size: int, sigma: float, integer=True):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = gaussian(size, std=sigma).reshape(size, 1)
    gkern2d = np.outer(gkern1d, gkern1d)

    if integer:
        max_value = np.max(gkern2d)
        gkern2d = gkern2d / max_value * 16
        gkern2d = gkern2d.astype(np.int32)

    return gkern2d

kernel = rbf_kernel(21, 3) - rbf_kernel(21, 2)

plt.imsave("kernel_detect.png", kernel)

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



gauss = np.array([[1,2,1],[2,4,2],[1,2,1]])
laplacian = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])

signal.convolve2d(laplacian, gauss)

