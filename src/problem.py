import random
from typing import List, Tuple, Optional, Generator, Callable, TypeVar, cast
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

    def becomeNeighbor(self, pred: Callable[[float, float], bool]) -> None:
        pass


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
        return math.hypot(self.x - otherNode.x, self.y - otherNode.y)

    def name(self):
        return f'{self.id:03}'


def getClosestNode(nodeSet: List[Node], closestTo: Node) -> Tuple[Node, int]:
    assert nodeSet

    closestIdx: int = 0
    minDist: float = nodeSet[closestIdx].distance(closestTo)

    for n in range(len(nodeSet)):
        nodeDist: float = nodeSet[n].distance(closestTo)
        if nodeDist < minDist:
            closestIdx = n
            minDist = nodeDist

    return nodeSet[closestIdx], closestIdx


T = TypeVar("T")
def find(target: T, myList: List[T]) -> Generator[int, None, None]:
    for i in range(len(myList)):
        if myList[i] == target:
            yield i


class Route:
    def __init__(self, stops: List[Node], depot: Node):
        # note: stops should never include the depot stop at the start or end
        self.stops: List[Node] = stops
        self.depot: Node = depot

        self._dist: Optional[float] = None
        self._demand: Optional[int] = None

    def __str__(self) -> str:
        nodeIds: List[str] = [str(n) for n in self.stops]
        if nodeIds:
            return f"0 {' '.join(nodeIds)} 0"
        else:
            return "0 0"

    def __repr__(self) -> str:
        return f"(Route {str(self.stops)})"

    def __eq__(self, other: object) -> bool:
        other = cast(Route, other)

        return type(other) == Route and self.depot == other.depot and self.stops == other.stops

    def __contains__(self, node: Node) -> bool:
        return node in self.stops

    def __len__(self):
        return len(self.stops)

    def isFeasible(self, truckCap: int) -> bool:
        demand: int = sum(node.demand for node in self.stops)
        return demand <= truckCap

    def addStop(self, node: Node) -> None:
        self.stops.append(node)

    def insertStop(self, node: Node, idx: int) -> None:
        self.stops.insert(idx, node)

    def removeStop(self, node: Node) -> None:
        self.stops.remove(node)

    def greedyReorder(self) -> 'Route':
        newOrder: List[Node] = []

        stops: List[Node] = self.stops[:]
        if self.stops:
            node: Node
            idx: int
            node, idx = getClosestNode(stops, self.depot)
            stops.pop(idx)
            newOrder.append(node)

            while stops:
                node, idx = getClosestNode(stops, newOrder[-1])
                stops.pop(idx)
                newOrder.append(node)

            assert set(newOrder) == set(self.stops)
            self.stops = newOrder

        return self

    @property
    def distance(self) -> float:
        if not self._dist:
            self._dist: float = 0
            prevNode: Node = self.depot

            for nextNode in self.stops + [self.depot]:
                self._dist += prevNode.distance(nextNode)
                prevNode = nextNode

        return self._dist

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
    def demand(self) -> int:
        if not self._demand:
            self._demand = sum([stop.demand for stop in self.stops])
        return self._demand

    def normalize(self) -> 'Route':
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

        self._greedyDist: Optional[float] = None

    def __repr__(self) -> str:
        return f"(Problem <#custs {self.numCustomers}>, <#trucks {self.numTrucks}>, <truckCap {self.truckCapacity}>)"

    def __eq__(self, other: object) -> bool:
        other = cast(VRPProblem, other)

        return type(other) == VRPProblem and\
               self.numCustomers == other.numCustomers and\
               self.numTrucks == other.numTrucks and\
               self.truckCapacity == other.truckCapacity and\
               self.depot == other.depot

    def addNode(self, node: Node) -> None:
        self.nodes.append(node)

    @property
    def totalDemand(self) -> int:
        return sum(node.demand for  node in self.nodes)

    @property
    def greedyDistance(self) -> float:
        if not self._greedyDist:
            route = Route(self.nodes, self.depot)
            route.greedyReorder()
            self._greedyDist = route.distance
        return self._greedyDist


class VRPSolution(Solution):
    def __init__(self, problem: VRPProblem, routes: List[Route]):
        assert len(routes) == problem.numTrucks  # every truck has a route

        self.problem: VRPProblem = problem
        self.routes: List[Route] = routes

        self._singletonDists: Optional[float] = None

    def __str__(self) -> str:
        return ' | '.join('-'.join(stop.name() for stop in route.stops) for route in self.routes)

    def __eq__(self, other: object) -> bool:
        other = cast(VRPSolution, other)

        return type(other) == VRPSolution and self.problem == other.problem and self.routes == other.routes

    def check(self) -> bool:
        pass

    @property
    def depot(self) -> Node:
        return self.problem.depot

    @property
    def objectiveValue(self) -> float:
        # distance plus infeasibility penalty
        return self.totalDistance + self.infeasibilityPenalty

    @property
    def infeasibilityPenalty(self) -> float:
        if not self._singletonDists:
            self._singletonDists = 2.0 * sum(self.problem.depot.distance(node) for node in self.problem.nodes)

        multiplier: float
        if self.capacityOverflow >= 0.01 * self.problem.totalDemand:
            multiplier = 0.1 * self._singletonDists
        else:
            multiplier = 0.2 * self._singletonDists

        return multiplier * self.capacityOverflow

    def neighbors(self) -> Generator['VRPSolution', None, None]:
        # generates neighbors by moving stops to a new index in the unioned routing
        swapFromIdxs: List[Tuple[int, int]] = [] # format: [(routeIdx, stopIdxInRoute), ...]
        swapToIdxs: List[Tuple[int, int]] = []
        for rIdx in range(len(self.routes)):
            swapFromIdxs += [(rIdx, nIdx) for nIdx in range(len(self.routes[rIdx]))]
            # allow appending to end of route - add 1 extra index for the possibility
            swapToIdxs += [(rIdx, nIdx) for nIdx in range(len(self.routes[rIdx]) + 1)]
        random.shuffle(swapFromIdxs)

        radius: int = len(swapToIdxs) // 2
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

    def randomNeighbor(self) -> 'VRPSolution':
        return self.neighbors().__next__()

    @property
    def totalDistance(self) -> float:
        return sum([route.distance for route in self.routes])

    def getAdjacentNodes(self, node: Node) -> Tuple[Node, Node]:
        """Get pair of nodes before + after a node"""
        for route in self.routes:
            i: int = route.stops.index(node)
            if i >= 0:
                return route.adjacents(i)

        raise Exception(f"Node not found: {node}")

    # def solutionDiffDistance(self, other) -> float:
    #     """Get _dist between 2 solutions"""
    #     # FIXME: buggy
    #     total: float = 0
    #     for (i, route) in enumerate(self.routes):
    #         for stop in route.stops:
    #             (pre1, suc1) = route.adjacents(i, self.depot)
    #             (pre2, suc2) = other.adjacents(stop)
    #             # choose the min of both orders to maintain invariance under route reversal:
    #             total += min(pre1.distance() + suc1.distance(),  # same order
    #                          pre1.distance() + suc1.distance()) # swapped
    #     return total

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
        # returns full route without the starting depot stop
        fullRoute: List[Node] = []
        for route in self.routes:
            assert route.depot == self.depot

            fullRoute.extend(route.stops)
            fullRoute.append(route.depot)

        return fullRoute

    def normalize(self) -> 'VRPSolution':
        """Mutate to be equivalent canonical representation. Use for debugging, not solving."""
        for route in self.routes:
            route.normalize()
        self.routes.sort(key = lambda r: (len(r.stops), r.stops[0].id if r.stops else -1))

        return self

    def immutNormalize(self) -> 'VRPSolution':
        newRoutes: List[Route] = []
        for route in self.routes:
            newRoute: Route = copy.deepcopy(route)
            newRoute.normalize()
            newRoutes.append(newRoute)
        newRoutes.sort(key = lambda r: (len(r.stops), r.stops[0].id if r.stops else -1))

        return VRPSolution(self.problem, newRoutes)

    def greedyRouteReorder(self) -> 'VRPSolution':
        # greedily reorder routes in place without swapping nodes between routes
        for route in self.routes:
            route.greedyReorder()

        return self


class VRPSolution2Op(VRPSolution):

    def neighbors(self) -> Generator['VRPSolution2Op', None, None]:
        #TODO: implement 2-op neighborhoods
        fullRoute: List[Node] = self.getFullRoute()
        maxEdgeIdx: int = self.problem.numCustomers + self.problem.numTrucks

        edge1Idxs: List[int] = list(range(1, maxEdgeIdx + 1))
        random.shuffle(edge1Idxs)

        for edge1Idx in edge1Idxs:
            edge2Idxs: List[int] = list(range(edge1Idx, maxEdgeIdx + 1))
            for edge2Idx in edge2Idxs:
                yield VRPSolution2Op(self.problem,
                                     self.parseRoutes(VRPSolution2Op.swapEdges(fullRoute, edge1Idx, edge2Idx)))

    # noinspection DuplicatedCode
    def randomNeighbor(self) -> 'VRPSolution2Op':

        # TODO: keep track of dist/demand on either side of nodes to avoid bad cuts

        routes = self.routes
        idx1: int = random.randrange(len(routes))      # not weighted
        idx2: int = random.randrange(len(routes))      # not weighted
        if idx1 == idx2:
            route: Route = routes[idx1]
            cut1 = random.randrange(len(route.stops)+1)
            cut2 = random.randrange(len(route.stops)+1)
            if cut1 > cut2: cut1, cut2 = cut2, cut1
            L, M, R = route.stops[:cut1], route.stops[cut1:cut2], route.stops[cut2:]
            new = Route(L+M[::1]+R, self.depot)
            return VRPSolution2Op(self.problem, routes[:idx1]+[new]+routes[idx1+1:])
        else:
            if idx1 > idx2: idx1, idx2 = idx2, idx1
            route1: Route = routes[idx1]
            route2: Route = routes[idx2]
            cut1 = random.randrange(len(route1.stops)+1)
            cut2 = random.randrange(len(route2.stops)+1)
            L1, R1 = route1.stops[:cut1], route1.stops[cut1:]
            L2, R2 = route2.stops[:cut2], route2.stops[cut2:]
            new1: Route
            new2: Route
            if random.randrange(2):
                new1 = Route(L1+R2, self.depot)
                new2 = Route(L2+R1, self.depot)
            else:
                new1 = Route(L1+L2[::-1], self.depot)
                new2 = Route(R1+R2[::-1], self.depot)
            return VRPSolution2Op(self.problem, routes[:idx1]+[new1]+routes[idx1+1:idx2]+[new2]+routes[idx2+1:])

    # noinspection DuplicatedCode
    def becomeNeighbor(self, pred: Callable[[float, float], bool]) -> None:
        routes = self.routes
        idx1: int = random.randrange(len(routes))      # not weighted
        idx2: int = random.randrange(len(routes))      # not weighted
        dobj: float     # difference in objective
        dpen: float     # differnece in penalty
        if idx1 == idx2:
            route: Route = routes[idx1]
            cut1 = random.randrange(len(route.stops)+1)
            cut2 = random.randrange(len(route.stops)+1)
            if cut1 > cut2: cut1, cut2 = cut2, cut1
            L, M, R = route.stops[:cut1], route.stops[cut1:cut2], route.stops[cut2:]
            new = Route(L+M[::1]+R, self.depot)
            dobj = new.distance - route.distance
            dpen = new.isFeasible(self.problem.truckCapacity) - route.isFeasible(self.problem.truckCapacity)
            if pred(dobj, dpen): routes[idx1] = new
        else:
            if idx1 > idx2: idx1, idx2 = idx2, idx1
            route1: Route = routes[idx1]
            route2: Route = routes[idx2]
            cut1 = random.randrange(len(route1.stops)+1)
            cut2 = random.randrange(len(route2.stops)+1)
            L1, R1 = route1.stops[:cut1], route1.stops[cut1:]
            L2, R2 = route2.stops[:cut2], route2.stops[cut2:]
            new1: Route
            new2: Route
            if random.randrange(2):
                new1 = Route(L1+R2, self.depot)
                new2 = Route(L2+R1, self.depot)
            else:
                new1 = Route(L1+L2[::-1], self.depot)
                new2 = Route(R1+R2[::-1], self.depot)
            dobj = (new1.distance+new2.distance) - (route1.distance+route2.distance)
            cap: float = self.problem.truckCapacity
            dpen = (new1.isFeasible(cap)+route1.isFeasible(cap)) - (new2.isFeasible(cap)-route2.isFeasible(cap))
            if pred(dobj, dpen):
                routes[idx1] = new1
                routes[idx2] = new2


    @staticmethod
    def swapEdges(fullRoute: List[Node], edge1Idx: int, edge2Idx: int) -> List[Node]:
        seg1: List[Node] = fullRoute[:edge1Idx]
        seg2: List[Node] = fullRoute[edge1Idx:edge2Idx]
        seg3: List[Node] = fullRoute[edge2Idx:]

        seg2.reverse()

        return seg1 + seg2 + seg3

    def parseRoutes(self, fullRoute: List[Node]) -> List[Route]:
        routes: List[Route] = []
        # TODO: debugging asserts for now
        assert fullRoute[-1] == self.problem.depot

        prevRouteEndIdx: int = -1
        for routeEndIdx in find(self.problem.depot, fullRoute):
            routes.append(Route(fullRoute[prevRouteEndIdx + 1:routeEndIdx], self.problem.depot))
            prevRouteEndIdx = routeEndIdx

        assert len(routes) == self.problem.numTrucks
        return routes


