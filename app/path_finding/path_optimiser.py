from typing import Generator

from ..utils import raytrace
from .types import Location, Map
from .utils import in_bounds

OCCUPATION_THRESHOLD = 2


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
