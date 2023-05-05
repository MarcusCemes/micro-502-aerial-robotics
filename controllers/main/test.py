import numpy as np
import cv2

import matplotlib.image
from scipy.signal.windows import gaussian


# def gaussian_kernel(l: int, sig: float):
#     ax = np.linspace(-(l - 1) / 2.0, (l - 1) / 2.0, l)
#     gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
#     kernel = np.outer(gauss, gauss)
#     return kernel / np.sum(kernel)


def gaussian_kernel(kernlen: int = 21, std: float = 3.0):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)

    # Convert the kernel kernel to a uint8 array
    max_value = np.max(gkern2d)
    gkern2d = gkern2d / max_value * 16
    gkern2d = gkern2d.astype(np.uint8)
    return gkern2d


field1 = np.zeros((10, 10), dtype=np.uint8)
field1[5, 5] = 1

kernel = gaussian_kernel()
print(kernel, kernel.shape)

field2 = cv2.filter2D(field1, -1, kernel)


matplotlib.image.imsave("output/kernel1.png", kernel, cmap="gray")
matplotlib.image.imsave("output/field1.png", field1, cmap="gray")
matplotlib.image.imsave("output/field2.png", field2, cmap="gray")
