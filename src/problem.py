import random
from typing import List, Tuple, Optional, Generator, Callable
from abc import ABC
import math
import copy
import itertools
import functools


class Solution(ABC):

    def solutionDiffDistance(self, other) -> float:
        pass

    @property
    def check(self) -> bool:
        pass

    @property
    def objectiveValue(self) -> float:
        pass

    def neighbors(self) -> Generator['Solution', None, None]:
        pass

    def equiv(self, other):
        return self.solutionDiffDistance(other) == 0

    def normalize(self): # -> Solution
        return self


class Problem(ABC):
    pass


class Node:
    def __init__(self, nodeId: int, demand: int, xPos: float, yPos: float):
        self.id = nodeId
        self.demand: int = demand
        self.x: float = xPos
        self.y: float = yPos

    def __eq__(self, otherNode) -> bool:
        return self.demand == otherNode.demand and self.x == otherNode.x and self.y == otherNode.y

    def __hash__(self) -> int:
        return hash(self.id)

    def __str__(self):
        return str(self.id)

    def __repr__(self):
        return f"(Node{self.id} <{self.x}, {self.y}>, {self.demand})"

    def distance(self, otherNode) -> float:
        return math.sqrt((self.x - otherNode.x) ** 2 + (self.y - otherNode.y) ** 2)

    def name(self):
        return f'{self.id:03}'


class Route:
    def __init__(self, stops: List[Node], depot: Node):
        # note: stops should never include the depot stop at the start or end
        self.stops: List[Node] = stops
        self.depot: Node = depot

        self._dist: Optional[float] = None

    def __str__(self) -> str:
        nodeIds: List[str] = [str(n) for n in self.stops]
        if nodeIds:
            return f"0 {' '.join(nodeIds)} 0"
        else:
            return "0 0"

    def __repr__(self) -> str:
        return f"(Route {str(self.stops)})"

    def __contains__(self, node: Node) -> bool:
        return node in self.stops

    def __len__(self):
        return len(self.stops)

    def addStop(self, node: Node) -> None:
        self.stops.append(node)

    def insertStop(self, node: Node, idx: int) -> None:
        self.stops.insert(idx, node)

    def removeStop(self, node: Node) -> None:
        self.stops.remove(node)

    @property
    def distance(self) -> float:
        if self._dist is not None: # memoize
            return self._dist
        else:
            dist: float = 0
            prevNode: Node = self.depot

            for nextNode in self.stops + [self.depot]:
                dist += prevNode.distance(nextNode)

            return dist

    def adjacents(self, i: int) -> Tuple[Node, Node]:
        if i == 0:
            return (self.depot, self.stops[1]) if self.stops else (self.depot, self.depot)
        elif i == len(self.stops) - 1:
            return (self.stops[i - 1], self.depot)
        elif 0 < i < len(self.stops) - 1:
            return (self.stops[i - 1], self.stops[i + 1])
        else:
            raise IndexError(f"Provided node index {i} is out of bounds for route of length {len(self.stops)}")

    @property
    def demand(self):
        return sum([stop.demand for stop in self.stops])

    def normalize(self):
        """Mutate to be equivalent canonical representation"""
        if self.stops and self.stops[0].id > self.stops[-1].id:
            self.stops.reverse()

        return self


class VRPProblem(Problem):
    def __init__(self, numCustomers: int, numTrucks: int, truckCapacity: int, depot: Node, file: str = None):
        assert depot.demand == 0

        self.file = file

        self.numCustomers: int = numCustomers
        self.numTrucks: int = numTrucks
        self.truckCapacity: int = truckCapacity

        self.depot: Node = depot
        self.nodes: List[Node] = []

    def __repr__(self) -> str:
        return f"(Problem <#custs {self.numCustomers}>, <#trucks {self.numTrucks}>, <truckCap {self.truckCapacity}>)"

    def addNode(self, node: Node) -> None:
        self.nodes.append(node)


class VRPSolution(Solution):
    def __init__(self, problem: VRPProblem, routes: List[Route]):
        assert len(routes) == problem.numTrucks  # every truck has a route

        self.problem: VRPProblem = problem
        self.routes: List[Route] = routes
        self._objVal: Optional[float] = None

    def __str__(self) -> str:
        return ' | '.join('-'.join(stop.name() for stop in route.stops) for route in self.routes)

    def check(self) -> bool:
        pass

    @property
    def depot(self) -> Node:
        return self.problem.depot

    @property
    def totalDistance(self) -> float:
        return sum(r.distance for r in self.routes)

    @property
    def objectiveValue(self) -> float:
        if self._objVal is not None: # memoize
            return self._objVal
        else:
            self._objVal = self.totalDistance + self.infeasibilityPenalty
            # distance plus infeasibility penalty
            return self._objVal

    @property
    def infeasibilityPenalty(self) -> float:
        return 5 * self.capacityOverflow

    def neighbors(self) -> Generator['VRPSolution', None, None]:
        # generates neighbors by moving stops to a new index in the unioned routing
        swapFromIdxs: List[Tuple[int, int]] = [] # format: [(routeIdx, stopIdxInRoute), ...]
        swapToIdxs: List[Tuple[int, int]] = []
        for rIdx in range(len(self.routes)):
            swapFromIdxs += [(rIdx, nIdx) for nIdx in range(len(self.routes[rIdx]))]
            # allow appending to end of route - add 1 extra index for the possibility
            swapToIdxs += [(rIdx, nIdx) for nIdx in range(len(self.routes[rIdx]) + 1)]
        random.shuffle(swapFromIdxs)

        radius: int = 100
        for fromIdx in swapFromIdxs:
            # randomize order of offsets
            offsets: List[int] = list(range(-radius, radius + 1))
            random.shuffle(offsets)

            for offset in offsets:
                # get the place to move the stop to by adding the offset in the ordered route list
                toIdx: Tuple[int, int] = swapToIdxs[(swapToIdxs.index(fromIdx) + offset) % len(swapToIdxs)]

                # create new routes with the changes
                newRoutes: List[Route] = copy.deepcopy(self.routes)
                newRoutes[toIdx[0]].insertStop(self.routes[fromIdx[0]].stops[fromIdx[1]], toIdx[1])
                # remove stop (relies on stop uniqueness
                newRoutes[fromIdx[0]].removeStop(self.routes[fromIdx[0]].stops[fromIdx[1]])

                yield VRPSolution(self.problem, newRoutes)


    @property
    def distance(self) -> float:
        return sum([route.distance for route in self.routes])

    def getAdjacentNodes(self, node: Node) -> Tuple[Node, Node]:
        """Get pair of nodes before + after a node"""
        for route in self.routes:
            i: int = route.stops.index(node)
            if i >= 0:
                return route.adjacents(i)

        raise Exception(f"Node not found: {node}")

    def solutionDiffDistance(self, other) -> float:
        """Get _dist between 2 solutions"""
        # FIXME: buggy
        total: float = 0
        for (i, route) in enumerate(self.routes):
            for stop in route.stops:
                (pre1, suc1) = route.adjacents(i, self.depot)
                (pre2, suc2) = other.adjacents(stop)
                # choose the min of both orders to maintain invariance under route reversal:
                total += min(pre1.distance() + suc1.distance(),  # same order
                             pre1.distance() + suc1.distance()) # swapped
        return total

    @property
    def capacityOverflow(self) -> float:
        capOverflow: float = 0
        for rte in self.routes:
            demandSupplyDiff = rte.demand - self.problem.truckCapacity
            capOverflow += demandSupplyDiff if demandSupplyDiff > 0 else 0

        return capOverflow

    def isFeasible(self) -> bool:
        return (self.capacityOverflow == 0)

    @property
    def isOptimal(self) -> bool:
        # TODO: fill in
        return False

    def getFullRoute(self) -> List[Node]:
        fullRoute: List[Node] = [self.depot]
        for route in self.routes:
            assert route.depot == self.depot

            fullRoute.extend(route.stops)
            fullRoute.append(route.depot)

        return fullRoute

    def normalize(self):
        """Mutate to be equivalent canonical representation. Use for debugging, not solving."""
        for route in self.routes:
            route.normalize()
        self.routes.sort(key = lambda r: (len(r.stops), r.stops[0].id if r.stops else -1))

        return self
