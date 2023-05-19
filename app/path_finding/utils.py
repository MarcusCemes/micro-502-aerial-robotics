from heapq import heappop, heappush
from typing import Generic, TypeVar

from .types import Location

T = TypeVar("T")


class PriorityQueue(Generic[T]):
    """A basic priority queue implementation using Python's heapq."""

    def __init__(self):
        self._elements: list[tuple[float, T]] = []

    def empty(self) -> bool:
        return not self._elements

    def put(self, item: T, priority: float):
        heappush(self._elements, (priority, item))

    def get(self) -> T:
        return heappop(self._elements)[1]


def in_bounds(location: Location, size: tuple[int, int]) -> bool:
    """
    Returns true if a location is within the bounds of a map,
    preventing out-of-bounds errors
    """

    (x, y), (w, h) = location, size
    return 0 <= x < w and 0 <= y < h
