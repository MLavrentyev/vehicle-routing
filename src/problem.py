import random
from typing import List, Tuple, TypeVar, Optional
from abc import ABC
import math
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

    def neighbors(self): # -> List[Solution]
        pass
    @classmethod
    def any(cls, prob): # -> Solution
        pass

    @classmethod
    def rand(cls, prob): # -> Solution
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

        self.dist: Optional[float] = None

    def __str__(self) -> str:
        nodeIds: List[str] = [str(n) for n in self.stops]
        if nodeIds:
            return f"0 {' '.join(nodeIds)} 0"
        else:
            return "0 0"

    def __repr__(self) -> str:
        return f"(Route {str(self.stops)})"

    def addStop(self, node: Node) -> None:
        self.stops.append(node)

    @property
    def distance(self) -> float:
        if self.dist is not None: # memoize
            return self.dist
        else:
            dist: float = 0
            prevNode: Node = self.depot

            for nextNode in self.stops + [self.depot]:
                dist += prevNode.distance(nextNode)

            return dist

    def adjacents(self, i: int, depot: Node) -> Tuple[Node, Node]:
        stops = self.stops
        if i == 0: return (stops[1] if stops else depot, depot)
        elif i == len(stops)-1: return (stops[len(stops)-1], depot)
        return (stops[i-1], stops[i+1])

    @property
    def demand(self):
        return sum([stop.demand for stop in self.stops])

    def normalize(self):
        """Mutate to be equivalent canonicle representation"""
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
        self.depot = problem.depot
        self._objVal: Optional[float] = None

    def __str__(self) -> str:
        return '|'.join('-'.join(stop.name() for stop in route.stops) for route in self.routes)

    def check(self) -> bool: pass

    @property
    def objectiveValue(self) -> float:
        if self._objVal is not None: # memoize
            return self._objVal

        #TODO: infeasibility penalty (self.capacityOverflow)
        return sum(r.distance for r in self.routes)

    def neighbors(self):
        neighs = []
        # swap a random pair from each route (just an example)
        for route in self.routes:
            if len(route.stops) >= 3:
                stops: List[Node] = route.stops
                i: int = random.randrange(len(stops)-1)
                newRoute = Route(stops[:i]+[stops[i+1], stops[i]]+stops[i+2:], self.depot)
                j: int = self.routes.index(route)
                newRoutes: List[Route] = self.routes[:j]+[newRoute]+self.routes[j+1:]
                neighs.append(VRPSolution(self.problem, newRoutes))
        return neighs

    @classmethod
    def any(cls, problem: VRPProblem):
        indss = [range(i, len(problem.nodes), problem.numTrucks) for i in range(problem.numTrucks)]

        return cls(problem, [Route([problem.nodes[i] for i in inds], problem.depot) for inds in indss])

    @classmethod
    def rand(cls, prob: VRPProblem):
        #TODO: What does this do?
        sol: VRPSolution = cls(prob, [Route([], prob.depot) for _ in range(prob.numTrucks)])

        for node in prob.nodes:
            random.choice(sol.routes).stops.append(node)

        for route in sol.routes:
            random.shuffle(route.stops)

        return sol

    @property
    def distance(self) -> float:
        return sum([route.distance for route in self.routes])

    def adjacents(self, node: Node) -> Tuple[Node, Node]:
        """Get pair of nodes before + after a node"""
        for route in self.routes:
            i: int = route.stops.index(node)
            if i >= 0:
                return route.adjacents(i, self.depot)

        raise Exception(f"Node not found: {node}")

    # def solutionDiffDistance(self, other) -> float:
    #     """Get dist between 2 solutions"""
    #     # TODO: what does this do?
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

    @property
    def isOptimal(self) -> bool:
        # TODO: fill in
        return False

    def fullRoute(self) -> List[Node]:
        #TODO: what is this doing?
        return [stop for route in self.routes for stop in [self.depot] + route.stops] + [self.depot]

    def normalize(self):
        """Mutate to be equivalent canonical representation. Use for debugging, not solving."""
        for route in self.routes:
            route.normalize()
        self.routes.sort(key = lambda r: (len(r.stops), r.stops[0].id if r.stops else -1))

        return self