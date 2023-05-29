from enum import Enum
from math import sqrt
from typing import Annotated, Literal

import cv2  # type: ignore
import numpy as np
import numpy.typing as npt
from loguru import logger
from matplotlib import pyplot as plt  # type: ignore

from .common import Context
from .config import MAP_PX_PER_M, MAP_SIZE, OPTIMISE_PATH, RANGE_THRESHOLD
from .path_finding.dijkstra import Dijkstra
from .path_finding.grid_graph import GridGraph
from .types import Coords
from .utils.debug import export_image
from .utils.math import Vec2, clip, raytrace, rbf_kernel, circular_kernel

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
OCCUPATION_THRESHOLD = 3

UNIT_SENSOR_VECTORS: Matrix2x4 = np.array([[1, 0, -1, 0], [0, 1, 0, -1]], dtype=DTYPE)

KERNEL_SIZE = 15
KERNEL_SIGMA = 1.5


class Sensor(Enum):
    Front = 0
    Left = 1
    Back = 2
    Right = 3


class Navigation:
    def __init__(self, ctx: Context):
        self._ctx = ctx

        MAP_X, MAP_Y = MAP_SIZE
        size = (int(float(MAP_PX_PER_M) * MAP_X), int(float(MAP_PX_PER_M) * MAP_Y))

        logger.info(f"Initialising map with size {size}")
        self.map = np.zeros(size, dtype=MAP_DTYPE)
        self.size = size

        self.field_gen = FieldGenerator()
        self.field = self.field_gen.next(self.map)

        self.high_sensitivity = False

    def update(self):
        loc_detections = np.multiply(self.read_range_readings(), UNIT_SENSOR_VECTORS)
        loc_proj_detections = np.multiply(self.reduction_factors(), loc_detections)
        relative_detections = np.dot(self.yaw_rotation_matrix(), loc_proj_detections)

        self.paint_relative_detections(relative_detections)
        self._ctx.outlet.broadcast({"type": "map", "data": self.map.tolist()})

        if self._ctx.debug_tick:
            export_image("map", self.map, cmap="RdYlGn_r")

    def save(self) -> Map:
        return self.map.copy()

    def restore(self, map: Map) -> None:
        self.map = map
        self.field = self.field_gen.next(self.map)

    def compute_path(self, start: Coords, end: Coords) -> list[Coords] | None:
        self.field = self.field_gen.next(self.map)

        if self._ctx.debug_tick:
            export_image("field", self.field, cmap="gray")

        graph = GridGraph(self.field)
        algo = Dijkstra(graph, optimise=OPTIMISE_PATH)

        path = algo.find_path(start, end)

        if self._ctx.debug_tick and path is not None:
            self.plot_and_save_path(path)

        self._ctx.outlet.broadcast({"type": "path", "data": path})

        return path

    def paint_relative_detections(self, detections: Matrix2x4) -> None:
        position = self.global_position()

        for detection in detections.T:
            if not np.any(detection):
                continue

            out_of_range = np.linalg.norm(detection) > RANGE_THRESHOLD
            detection = position + Vec2(*detection)
            self.paint_detection(position, detection, not out_of_range)

        # self.paint_border()

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

        if occupation:
            offset = 255 if self.high_sensitivity else 64
        else:
            offset = -8

        self.map[coords] = clip(value + offset, MAP_MIN, MAP_MAX)

    def read_range_readings(self) -> Matrix1x4:
        return np.array(
            [
                clip(self._ctx.sensors.front, 0.0, RANGE_THRESHOLD + 0.01),
                clip(self._ctx.sensors.left, 0.0, RANGE_THRESHOLD + 0.01),
                clip(self._ctx.sensors.back, 0.0, RANGE_THRESHOLD + 0.01),
                clip(self._ctx.sensors.right, 0.0, RANGE_THRESHOLD + 0.01),
            ],
            dtype=DTYPE,
        )

    def reduction_factors(self) -> Matrix1x4:
        s = self._ctx.sensors
        return np.repeat(np.cos(np.array([s.pitch, s.roll], DTYPE)), 2)

    def yaw_rotation_matrix(self) -> Matrix2x2:
        yaw = self._ctx.sensors.yaw
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c, -s], [s, c]], dtype=DTYPE)

    def global_position(self) -> Vec2:
        s = self._ctx.sensors
        return Vec2(s.x, s.y)

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

    def to_position(self, coords: Coords) -> Vec2:
        (x, y) = coords
        # print(" waypoint coords", coords)
        # print("next no mag ", (x+.5)/MAP_PX_PER_M, (y+.5)/MAP_PX_PER_M)
        return Vec2((x + 0.5) / MAP_PX_PER_M, (y + 0.5) / MAP_PX_PER_M)

    def paint_border(self):
        self.map[0, :] = MAP_MAX
        self.map[-1, :] = MAP_MAX
        self.map[:, 0] = MAP_MAX
        self.map[:, -1] = MAP_MAX

    def is_visitable(self, coords: Coords) -> bool:
        return self.field[coords] < OCCUPATION_THRESHOLD

    def distance_to_obstacle(self, coords: Coords) -> float:
        distance = float("inf")
        (x, y) = coords
        indices = np.transpose(np.where(self.map >= OCCUPATION_THRESHOLD))

        for i, j in indices:
            distance = min(distance, sqrt((x - i) ** 2 + (y - j) ** 2))

        return distance / MAP_PX_PER_M

    def plot_and_save_path(self, path: list[Coords]) -> None:
        plt.figure("path")
        plt.xlim(0, self.map.shape[0])
        plt.ylim(0, self.map.shape[1])

        for i, j in path:
            plt.plot(
                i,
                j,
                marker="o",
                markersize=2,
                markeredgecolor="red",
            )

        plt.savefig("output/path.png")


class FieldGenerator:
    def __init__(self):
        super().__init__()

        self.kernel = circular_kernel(KERNEL_SIZE)
        export_image("kernel", self.kernel, cmap="gray")

    def next(self, map: Map) -> Field:
        field = np.zeros(map.shape, dtype=np.int32)
        field[map > 0] = 1
        return cv2.filter2D(field, -1, self.kernel)
