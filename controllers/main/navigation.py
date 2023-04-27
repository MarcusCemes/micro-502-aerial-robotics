from enum import Enum
from typing import Annotated, Literal, Optional

import numpy as np
import numpy.typing as npt

from common import Context
from config import MAP_PX_PER_M, MAP_SIZE, RANGE_THRESHOLD
from log import Logger

DType = np.float32
Vector2 = Annotated[npt.NDArray[DType], Literal[2]]
Vector3 = Annotated[npt.NDArray[DType], Literal[3]]

Matrix1x4 = Annotated[npt.NDArray[DType], Literal[1, 4]]
Matrix2x2 = Annotated[npt.NDArray[DType], Literal[2, 2]]
Matrix2x4 = Annotated[npt.NDArray[DType], Literal[2, 4]]


UNIT_SENSOR_VECTORS: Matrix2x4 = np.array([[1, 0, -1, 0], [0, 1, 0, -1]], dtype=DType)


class Sensor(Enum):
    Front = 0
    Left = 1
    Back = 2
    Right = 3


class Navigation(Logger):
    def __init__(self, ctx: Context):
        self.ctx = ctx

        size = tuple(int(MAP_PX_PER_M * v) for v in MAP_SIZE)

        self.info(f"Initialising map with size {size}")
        self.map = np.zeros(size, dtype=np.int8)
        self.size = size

    def update(self) -> None:
        loc_detections = np.multiply(self.read_range_readings(), UNIT_SENSOR_VECTORS)
        # loc_proj_detections = np.multiply(self.reduction_factors(), loc_detections)
        relative_detections = np.dot(self.yaw_rotation_matrix(), loc_detections)

        self.update_relative_detections(relative_detections)

    def update_relative_detections(self, detections: Matrix2x4) -> None:
        position = self.global_position()

        for detection in detections.T:
            if not np.any(detection):
                continue

            detection = position + detection
            self.update_detection(detection)

    def update_detection(self, detection: Vector2) -> None:
        if coords := self.to_coords(detection):
            self.map[coords] = 255

    def read_range_readings(self) -> Matrix1x4:
        s = self.ctx.sensors

        readings = np.array(
            [
                s.range_front,
                s.range_left,
                s.range_back,
                s.range_right,
            ],
            dtype=DType,
        )

        return np.where(readings < RANGE_THRESHOLD, readings, 0)

    def reduction_factors(self) -> Matrix1x4:
        """
        TODO: Correct this function and use it
        """
        s = self.ctx.sensors
        return np.repeat(np.cos(np.abs(np.array([s.pitch, s.yaw], DType))), 2)

    def yaw_rotation_matrix(self) -> Matrix2x2:
        yaw = self.ctx.sensors.yaw
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c, -s], [s, c]], dtype=DType)

    def global_position(self) -> Vector2:
        s = self.ctx.sensors
        return np.array([s.x_global, s.y_global], dtype=DType)

    def to_coords(self, position: Vector2) -> Optional[tuple[int, int]]:
        [x, y] = position
        px_x, px_y = self.size
        size_x, size_y = MAP_SIZE
        cx, cy = (int(x * px_x / size_x), int(y * px_y / size_y))

        if cx < 0 or cx >= px_x or cy < 0 or cy >= px_y:
            return None

        return (cx, cy)
