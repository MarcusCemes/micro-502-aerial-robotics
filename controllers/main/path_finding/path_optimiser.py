from typing import Generator

from .types import Location, Map
from .utils import in_bounds

OCCUPATION_THRESHOLD = 64


class PathOptimiser:
    """
    Attempts to simplify a path found by a path-finding algorithm
    by removing unnecessary nodes. This is a relatively cheap
    post-processing step that can provide more optimal line-of-sight paths.
    """

    def __init__(self, map: Map):
        self.map = map
        self.size = map.shape

    def optimise(self, path: list[Location]) -> list[Location]:
        """
        Optimise a path by removing unnecessary nodes based on a
        line-of-sight algorithm.
        """

        # At least three nodes are needed
        if len(path) <= 2:
            return path

        i = 1

        # Keep trying to remove nodes until we can't remove any more
        while i != len(path) - 1:
            if self.free_path(path[i - 1], path[i + 1]):
                path.pop(i)

                # The previous node may now be removable
                i = i - 1 if i > 1 else 1

            else:
                i += 1

        return path

    def free_path(self, a: Location, b: Location) -> bool:
        """Returns true if there is a line-of-sight between two nodes."""

        for x, y in self.intermediate_nodes(a, b):
            if self.map[x, y] >= OCCUPATION_THRESHOLD:
                return False

        return True

    def intermediate_nodes(
        self, a: Location, b: Location
    ) -> Generator[Location, None, None]:
        """Returns all nodes between two nodes using a linecover algorithm."""

        for loc in raytrace(a, b):
            if in_bounds(loc, self.size):
                yield loc


def raytrace(a: Location, b: Location) -> Generator[Location, None, None]:
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
