from enum import Enum
from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt
from common import Context
from config import MAP_PX_PER_M, MAP_SIZE, RANGE_THRESHOLD
from log import Logger
from utils import Coords, Vec2, clip, raytrace

DTYPE = np.float32

Vector2 = Annotated[npt.NDArray[DTYPE], Literal[2]]
Vector3 = Annotated[npt.NDArray[DTYPE], Literal[3]]

Matrix1x4 = Annotated[npt.NDArray[DTYPE], Literal[1, 4]]
Matrix2x2 = Annotated[npt.NDArray[DTYPE], Literal[2, 2]]
Matrix2x4 = Annotated[npt.NDArray[DTYPE], Literal[2, 4]]

Map = npt.NDArray[np.int8]
Field = npt.NDArray[np.uint8]

MAP_DTYPE = np.int8
MAP_MIN = -127
MAP_MAX = 127

UNIT_SENSOR_VECTORS: Matrix2x4 = np.array([[1, 0, -1, 0], [0, 1, 0, -1]], dtype=DTYPE)


class Sensor(Enum):
    Front = 0
    Left = 1
    Back = 2
    Right = 3


class Navigation(Logger):
    def __init__(self, ctx: Context):
        self.ctx = ctx

        MAP_X, MAP_Y = MAP_SIZE
        size = (int(MAP_PX_PER_M * MAP_X), int(MAP_PX_PER_M * MAP_Y))

        self.info(f"Initialising map with size {size}")
        self.map = np.zeros(size, dtype=MAP_DTYPE)
        self.map_dirty = False
        self.size = size

    def update(self) -> None:
        loc_detections = np.multiply(self.read_range_readings(), UNIT_SENSOR_VECTORS)
        loc_proj_detections = np.multiply(self.reduction_factors(), loc_detections)
        relative_detections = np.dot(self.yaw_rotation_matrix(), loc_proj_detections)

        self.paint_relative_detections(relative_detections)

    def paint_relative_detections(self, detections: Matrix2x4) -> None:
        position = self.global_position()

        for detection in detections.T:
            if not np.any(detection):
                continue

            self.map_dirty = True

            out_of_range = np.linalg.norm(detection) > RANGE_THRESHOLD
            detection = position + Vec2(*detection)
            self.paint_detection(position, detection, not out_of_range)

        if self.map_dirty:
            self.map_dirty = False
            self.paint_border()
            self.ctx.outlet.broadcast({"type": "map", "data": self.map.tolist()})

    def paint_detection(self, origin: Vec2, detection: Vec2, detected: bool) -> None:
        coords_origin = self.to_coords(origin)
        coords_detection = self.to_coords(detection)

        for coords in raytrace(coords_origin, coords_detection):
            if coords != coords_detection and self.coords_in_range(coords):
                self.update_pixel(coords, False)

        if detected and self.coords_in_range(coords_detection):
            self.update_pixel(coords_detection, True)

    def update_pixel(self, coords: Coords, occupation: bool) -> None:
        value = self.map[coords]
        offset = 64 if occupation else -8
        self.map[coords] = clip(value + offset, MAP_MIN, MAP_MAX)

    def read_range_readings(self) -> Matrix1x4:
        s = self.ctx.sensors

        return np.array(
            [
                s.range_front,
                s.range_left,
                s.range_back,
                s.range_right,
            ],
            dtype=DTYPE,
        )

    def reduction_factors(self) -> Matrix1x4:
        s = self.ctx.sensors
        return np.repeat(np.cos(np.array([s.pitch, s.roll], DTYPE)), 2)

    def yaw_rotation_matrix(self) -> Matrix2x2:
        yaw = self.ctx.sensors.yaw
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c, -s], [s, c]], dtype=DTYPE)

    def global_position(self) -> Vec2:
        s = self.ctx.sensors
        return Vec2(s.x_global, s.y_global)

    def to_coords(self, position: Vec2) -> Coords:
        px_x, px_y = self.size
        size_x, size_y = MAP_SIZE

        cx = int(position.x * px_x / size_x)
        cy = int(position.y * px_y / size_y)

        return (cx, cy)

    def coords_in_range(self, coords: Coords) -> bool:
        x, y = coords
        px_x, px_y = self.size

        return x >= 0 and x < px_x and y >= 0 and y < px_y

    def paint_border(self):
        self.map[0, :] = MAP_MAX
        self.map[-1, :] = MAP_MAX
        self.map[:, 0] = MAP_MAX
        self.map[:, -1] = MAP_MAX

    def potential_map(self) -> Field:
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=DTYPE)
        return convolve2D(self.map, kernel, padding=1)  # type: ignore


def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros(
            (image.shape[0] + padding * 2, image.shape[1] + padding * 2)
        )
        imagePadded[
            int(padding) : int(-1 * padding), int(padding) : int(-1 * padding)
        ] = image
        print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (
                            kernel * imagePadded[x : x + xKernShape, y : y + yKernShape]
                        ).sum()
                except:
                    break

    return output
