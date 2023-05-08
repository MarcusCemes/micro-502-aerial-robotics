from typing import Generator

from .types import Location, Map, WeightedGraph
from .utils import in_bounds

DIST_ADJC = 1.0
DIST_DIAG = 1.41421356237
OBSTACLE_THRESHOLD = 64


class GridGraph(WeightedGraph):
    """
    Standard grid graph implementation for Dijkstra. Splits the
    map into a grid of nodes, and connects them if they are directly adjacent
    (vertically, horizontally, or diagonally). The cost between directly
    adjacent nodes is 1, and between diagonally adjacent nodes is sqrt(2).
    """

    def __init__(self, map: Map):
        self.map = map
        self.size = map.shape
        self.nodes = []

    def neighbors(self, location: Location, _) -> Generator[Location, None, None]:
        (x, y) = location

        for dx, dy in self._offsets():
            nx = x + dx
            ny = y + dy

            # if in_bounds((nx, ny), self.size) and self.map[nx, ny] < OBSTACLE_THRESHOLD:
            if in_bounds((nx, ny), self.size):
                yield (nx, ny)

    def cost(self, a: Location, b: Location) -> float:
        x1, y1 = a
        x2, y2 = b

        base_cost = DIST_DIAG if x1 != x2 and y1 != y2 else DIST_ADJC
        potential_cost = max(1, self.map[b])

        return base_cost * potential_cost

    def _offsets(self) -> Generator[Location, None, None]:
        """Iterates over all index offsets adjacent to a node."""

        for i in range(-1, 2):
            for j in range(-1, 2):
                if i != 0 or j != 0:
                    yield (i, j)
