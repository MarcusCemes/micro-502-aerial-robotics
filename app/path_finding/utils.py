from heapq import heappop, heappush
from typing import Generic, TypeVar

from .types import Location

T = TypeVar("T")


class PriorityQueue(Generic[T]):
    """A basic priority queue implementation using Python's heapq."""

    def __init__(self):
        self.elements: list[tuple[float, T]] = []

    def empty(self) -> bool:
        return not self.elements

    def put(self, item: T, priority: float):
        heappush(self.elements, (priority, item))

    def get(self) -> T:
        return heappop(self.elements)[1]


def in_bounds(location: Location, size: tuple[int, int]) -> bool:
    """
    Returns true if a location is within the bounds of a map,
    preventing out-of-bounds errors
    """

    (x, y), (w, h) = location, size
    return 0 <= x < w and 0 <= y < h
