from math import pi
from random import random

import numpy as np

from common import Context
from config import MAP_PX_PER_M, MAP_SIZE, RANGE_THRESHOLD
from log import Logger


class Navigation(Logger):
    def __init__(self, ctx: Context):
        self.ctx = ctx

        size = tuple(int(MAP_PX_PER_M * v) for v in MAP_SIZE)

        self.info(f"Initialising map with size {size}")
        self.map = np.zeros(size, dtype=np.int8)
        self.size = size

    def update(self) -> None:
        sensors = self.ctx.sensors

        self.update_direction(sensors.range_front, 0)
        # self.update_direction(sensors.range_left, 0.5 * pi)
        # self.update_direction(sensors.range_left, pi)
        # self.update_direction(sensors.range_left, 1.5 * pi)

    def update_direction(self, distance: float, direction: float) -> None:
        """
        TODO: Improve this to take roll into account!
        """

        if distance >= RANGE_THRESHOLD:
            return

        sensors = self.ctx.sensors
        x_global, y_global = sensors.x_global, sensors.y_global
        pitch, roll, yaw = sensors.pitch, sensors.roll, sensors.yaw

        # x = int(x_global + distance * np.cos(yaw + direction) * np.cos(pitch))
        # y = int(y_global + distance * np.sin(yaw + direction) * np.cos(pitch))

        x = x_global + distance
        y = y_global

        if self.ctx.debug_tick:
            print(
                "x: {:2f}, y: {:2f}, d: {:2f}, d_x: {:2f}, d_y: {:2f}".format(
                    x_global, y_global, distance, x, y
                )
            )

        self.set_occupancy((x, y), True)

    def get_occupancy(self, x: float, y: float) -> bool:
        x, y = int(x), int(y)
        return self.map[x, y] != 0

    def set_occupancy(self, position: tuple[float, float], occupied: bool) -> None:
        coords = self.to_coords(position)
        self.map[coords] = 255 if occupied else 0

    def to_coords(self, position: tuple[float, float]) -> tuple[int, int]:
        x, y = position
        px_x, px_y = self.size
        size_x, size_y = MAP_SIZE
        return (int(x * px_x / size_x), int(y * px_y / size_y))
