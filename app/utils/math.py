from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, isclose, pi, sin, sqrt
from typing import Generator, overload

import numpy as np
import numpy.typing as npt
from scipy.signal.windows import gaussian

from ..types import Coords

TWO_PI = 2 * pi

EQ_TOLERANCE = 1e-6


class Vec2:
    x: float = 0.0
    y: float = 0.0

    def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
        self.x = float(x)
        self.y = float(y)

    def __eq__(self, rhs: Vec2) -> bool:
        return isclose(self.x, rhs.x, abs_tol=EQ_TOLERANCE) and isclose(
            self.y, rhs.y, abs_tol=EQ_TOLERANCE
        )

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
        return f"Vec2({self.x:.2f}, {self.y:.2f})"

    def __repr__(self) -> str:
        return str(self)

    def abs(self) -> float:
        return sqrt(self * self)

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

    def limit(self, mag: float) -> Vec2:
        if self.abs() >= mag:
            return self.set_mag(mag)

        return self

    def to_list(self) -> list[float]:
        return [self.x, self.y]

    def mag2(self) -> float:
        return self * self

    def rotate(self, angle: float) -> Vec2:
        s, c = sin(angle), cos(angle)
        return Vec2(self.x * c - self.y * s, self.x * s + self.y * c)

    def set_mag(self, mag: float) -> Vec2:
        if self.x == 0.0 and self.y == 0.0:
            return self

        return (mag / self.abs()) * self


def clip(value: float, min_value: float, max_value: float) -> float:
    return min(max(value, min_value), max_value)


def normalise_angle(angle: float) -> float:
    """Normalises an angle to the range [0, 2*PI]."""
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


def rbf_kernel(size: int, sigma: float, integer=True) -> npt.NDArray:
    """Returns a 2D Gaussian kernel array."""
    gkern1d = gaussian(size, std=sigma).reshape(size, 1)
    gkern2d = np.outer(gkern1d, gkern1d)

    if integer:
        max_value = np.max(gkern2d)
        gkern2d = gkern2d / max_value * 16
        gkern2d = gkern2d.astype(np.int32)

    return gkern2d


def deg_to_rad(deg: float) -> float:
    return deg * pi / 180


def rad_to_deg(rad: float) -> float:
    return rad * 180 / pi


def mm_to_m(mm: float) -> float:
    return mm / 1000
