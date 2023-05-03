from enum import Enum
from multiprocessing import Process, Queue
from queue import Empty, Full
from time import time
from typing import Annotated, Literal

import matplotlib.image
import numpy as np
import numpy.typing as npt
from common import Context
from config import DEBUG_FILES, MAP_PX_PER_M, MAP_SIZE, RANGE_THRESHOLD
from convolution import conv2d
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

KERNEL_SIZE = 15
KERNEL_SIGMA = 3.0


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

        self.field_computer = PotentialComputer(self.ctx)

    def update(self) -> None:
        field = self.field_computer.recv()

        if field is not None:
            img = np.flip(np.flip(field, 1), 0)
            matplotlib.image.imsave("output/field.png", img, cmap="gray")

        loc_detections = np.multiply(self.read_range_readings(), UNIT_SENSOR_VECTORS)
        loc_proj_detections = np.multiply(self.reduction_factors(), loc_detections)
        relative_detections = np.dot(self.yaw_rotation_matrix(), loc_proj_detections)

        self.paint_relative_detections(relative_detections)

        self.field_computer.send(self.map)

    def paint_relative_detections(self, detections: Matrix2x4) -> None:
        position = self.global_position()

        for detection in detections.T:
            if not np.any(detection):
                continue

            out_of_range = np.linalg.norm(detection) > RANGE_THRESHOLD
            detection = position + Vec2(*detection)
            self.paint_detection(position, detection, not out_of_range)

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


class FieldGenerator:
    def __init__(self):
        super().__init__()
        self.kernel = self.generate_kernel(KERNEL_SIZE, KERNEL_SIGMA)

    def next(self, map: Map) -> Field:
        print(f"Applying convolution to map of shape {map.shape}")
        field = np.zeros(map.shape, dtype=np.uint8)
        field[map > 0] = 1

        return conv2d(field, self.kernel)

    def generate_kernel(self, size: int, sigma: float) -> npt.NDArray:
        x = np.linspace(-size, size, 2 * size + 1)
        kernel = np.exp(-(x**2) / (2 * sigma**2))
        kernel /= np.sum(kernel)
        return kernel


class PotentialComputer:
    def __init__(self, ctx: Context):
        super().__init__()
        self.ctx = ctx

        self.maps: Queue[Map] = Queue(1)
        self.fields: Queue[Field] = Queue(1)

        self.processing = False

        process = Process(target=run, args=(self.maps, self.fields))
        process.start()

        self.generator = FieldGenerator()

    def send(self, map: Map) -> None:
        try:
            self.maps.put_nowait(map)
        except Full:
            pass

    def recv(self) -> Field | None:
        try:
            return self.fields.get_nowait()
        except Empty:
            return None


def run(maps, fields) -> None:
    kernel = generate_kernel(KERNEL_SIZE, KERNEL_SIGMA)

    if DEBUG_FILES:
        matplotlib.image.imsave("output/kernel.png", kernel)

    while True:
        map = maps.get()

        field = np.zeros(map.shape, dtype=np.uint8)
        field[map > 0] = 1
        field = conv2d(field, kernel)

        fields.put(field)


def generate_kernel(size: int, sigma: float) -> npt.NDArray:
    ax = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)
