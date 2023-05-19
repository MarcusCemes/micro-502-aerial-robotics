from typing import Generator, Protocol

from numpy import uint8
from numpy.typing import NDArray

Location = tuple[int, int]
Map = NDArray[uint8]


class WeightedGraph(Protocol):
    """Abstract base class for weighted graphs."""

    map: Map
    nodes: list[Location]

    def neighbors(
        self,
        location: Location,
        end: Location,
    ) -> Generator[Location, None, None]:
        """Returns a generator of all neighbors of a given node."""
        raise NotImplementedError()

    def cost(self, origin: Location, target: Location) -> float:
        """Returns the cost of moving between two particular nodes."""
        raise NotImplementedError()


class Algorithm(Protocol):
    """Abstract class for a path-finding algorithm."""

    def find_path(
        self,
        start: Location,
        end: Location,
    ) -> list[Location]:
        """Finds the shortest path between two nodes."""
        raise NotImplementedError()
