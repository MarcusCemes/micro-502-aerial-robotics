from .path_optimiser import PathOptimiser
from .types import Algorithm, Location, WeightedGraph
from .utils import PriorityQueue

# Dictionary that holds explored paths and their costs
Threads = dict[Location, tuple[Location | None, float]]


INF = float("inf")


class Dijkstra(Algorithm):
    """
    Implementation of the Dijkstra algorithm, which finds the shortest path
    between two points on a graph. This implementation uses a `heapq`-based
    priority queue to maintain the list of interesting nodes to visit.
    """

    def __init__(self, graph: WeightedGraph, optimise=True):
        self.graph = graph
        self.optimise = optimise

    def find_path(self, start: Location, end: Location) -> list[Location] | None:
        """Finds the shortest path between two points on the configured graph."""

        # Initialise the main data structures
        frontier = PriorityQueue()

        # Like when exploring a labyrinth, we keep track of the paths we've explored
        threads: Threads = {}

        # Add the starting node to the frontier
        frontier.put(start, 0)
        threads[start] = (None, 0)

        # While there are still nodes to explore, explore them...
        while not frontier.empty():
            current = frontier.get()

            # ...until we reach the end node
            if current == end:
                break

            (_, current_cost) = threads[current]

            # Explore the neighbours of the current node
            for next in self.graph.neighbors(current, end):
                new_cost = current_cost + self.graph.cost(current, next)

                (_, old_cost) = threads.get(next, (None, INF))

                # If the new path is shorter than a previous path, update the cost
                if new_cost < old_cost:
                    threads[next] = (current, new_cost)
                    frontier.put(next, new_cost)

        # Reconstruct the path from the cost dictionary
        path = self._reconstruct_path(threads, end)

        if path and self.optimise:
            path = PathOptimiser(self.graph.map).optimise(path)

        return path

    def _reconstruct_path(
        self, threads: Threads, end: Location
    ) -> list[Location] | None:
        """Reconstructs the path from the threads dictionary."""

        if end not in threads:
            return None

        (parent, _) = threads[end]
        path = [end]

        while parent != None:
            path.append(parent)
            (parent, _) = threads[parent]

        path.reverse()
        return path
