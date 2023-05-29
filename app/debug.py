from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, Final

import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
import cv2

# from .utils.math import Vec2
MAP_PX_PER_M: Final = 25


# class Debug:
#     def __init__(self) -> None:
#         self.map_offset = 3.5
#         self.size = (int(float(MAP_PX_PER_M) * 1.5), int(float(MAP_PX_PER_M) * 3.0))
#         self.probability_map = np.zeros(self.size)
#         cv2.circle(self.probability_map, (70, 0), 5, 1, -1)
#         # plt.imshow(self.probability_map)
#         # plt.show()

#     def to_coords(self, position):
#         px_x, px_y = self.size
#         size_x, size_y = 1.5, 3.0

#         cx = int((position.x - self.map_offset) * px_x / size_x)
#         cy = int(position.y * px_y / size_y)

#         return (cx, cy)

#     def to_position(self, coords):
#         (y, x) = coords

#         return (1.5 - (x + 0.5) / MAP_PX_PER_M, 3.0 - (y + 0.5) / MAP_PX_PER_M)

#     def find_mean_position(self):
#         x_coords, y_coords = np.meshgrid(
#             range(self.probability_map.shape[1]),
#             range(self.probability_map.shape[0]),
#         )

#         mean_x = np.sum(x_coords * self.probability_map) / np.sum(self.probability_map)
#         mean_y = np.sum(y_coords * self.probability_map) / np.sum(self.probability_map)

#         return self.to_position((mean_x, mean_y))


# d = Debug()

# print(d.find_mean_position())

img = plt.imread("probability_map.png")[:, :, 0]
print(img.shape)
threshold = np.median(img[img > 0])
img = img[img > threshold]

plt.imshow(img)
plt.show()
