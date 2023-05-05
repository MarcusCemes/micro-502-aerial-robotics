from __future__ import annotations


from dataclasses import dataclass
from math import atan2, cos, pi, sin, sqrt
from typing import Generator, Tuple, TypeVar, overload

import numpy as np
import numpy.typing as npt
from scipy.signal.windows import gaussian

from common import Context

TWO_PI = 2 * pi

T = TypeVar("T")

Coords = Tuple[int, int]


class Timer:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.timer_ticks = ctx.ticks

    def reset(self) -> None:
        self.timer_ticks = self.ctx.ticks

    def elapsed_ticks(self, ticks: int) -> bool:
        return self.ctx.ticks - self.timer_ticks >= ticks


@dataclass
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def __add__(self, rhs: Vec2) -> Vec2:
        return Vec2(self.x + rhs.x, self.y + rhs.y)

    def __sub__(self, rhs: Vec2) -> Vec2:
        return Vec2(self.x - rhs.x, self.y - rhs.y)

    @overload
    def __mul__(self, rhs: float) -> Vec2:
        ...

    @overload
    def __mul__(self, rhs: Vec2) -> float:
        ...

    def __mul__(self, rhs: float | Vec2) -> float | Vec2:
        if isinstance(rhs, Vec2):
            return self.x * rhs.x + self.y * rhs.y
        else:
            return Vec2(self.x * rhs, self.y * rhs)

    def __rmul__(self, lhs: float) -> Vec2:
        return self * lhs

    def __str__(self) -> str:
        return "Vec2({:.2f}, {:.2f})".format(self.x, self.y)

    def angle(self) -> float:
        return atan2(self.y, self.x)

    @overload
    def clip(self, bound: float, /) -> Vec2:
        ...

    @overload
    def clip(self, min_value: float, max_value: float, /) -> Vec2:
        ...

    def clip(self, min_value: float, max_value: float | None = None) -> Vec2:
        if max_value is None:
            max_value = abs(min_value)
            min_value = -max_value

        return Vec2(
            clip(self.x, min_value, max_value), clip(self.y, min_value, max_value)
        )

    def limit_mag(self, mag: float) -> Vec2:
        if self.mag() >= mag:
            return self.set_mag(mag)

        return self

    def to_list(self) -> list[float]:
        return [self.x, self.y]

    def mag(self) -> float:
        return sqrt(self * self)

    def rotate(self, angle: float) -> Vec2:
        s, c = sin(angle), cos(angle)
        return Vec2(self.x * c - self.y * s, self.x * s + self.y * c)

    def set_mag(self, mag: float) -> Vec2:
        if self.x == 0 and self.y == 0:
            return self

        return (mag / self.mag()) * self


def clip(value: float, min_value: float, max_value: float) -> float:
    return min(max(value, min_value), max_value)


def normalise_angle(angle: float) -> float:
    angle = ((angle % TWO_PI) + TWO_PI) % TWO_PI

    if angle > pi:
        angle -= TWO_PI

    return angle


def raytrace(a: Coords, b: Coords) -> Generator[Coords, None, None]:
    """
    An integer-only implementation of a supercover line algorithm.
    Enumerates all grid cells that intersect with a line segment.
    See https://playtechs.blogspot.com/2007/03/raytracing-on-grid.html
    """

    (x1, y1), (x2, y2) = a, b

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    n = 1 + dx + dy

    x = x1
    y = y1

    x_inc = 1 if x2 > x1 else -1
    y_inc = 1 if y2 > y1 else -1

    error = dx - dy
    dx *= 2
    dy *= 2

    for _ in range(n, 0, -1):
        yield x, y

        if error > 0:
            x += x_inc
            error -= dy

        else:
            y += y_inc
            error += dx


def rbf_kernel(size: int, sigma: float, unsigned=True) -> npt.NDArray:
    """Returns a 2D Gaussian kernel array."""
    gkern1d = gaussian(size, std=sigma).reshape(size, 1)
    gkern2d = np.outer(gkern1d, gkern1d)

    if unsigned:
        max_value = np.max(gkern2d)
        gkern2d = gkern2d / max_value * 16
        gkern2d = gkern2d.astype(np.uint8)

    return gkern2d
