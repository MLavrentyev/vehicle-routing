from typing import List, Tuple, TypeVar
from abc import ABC
import math
import itertools
import functools

T = TypeVar('T')

class Solution(ABC):
    def dist(self, other) -> float: pass
    @classmethod
    def any(cls, prob: "Problem"): pass
    @classmethod
    def rand(cls, prob: "Problem"): pass

class Problem(ABC):
    def check(self, sol: Solution) -> bool: pass
    def objectiveValue(self, sol: Solution) -> float: pass

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


DEPOT = Node(-1, 0, 0, 0)


class Route:
    def __init__(self, stops: List[Node] = None):
        self.stops: List[Node] = stops if stops else []

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

    def distance(self, depot: Node) -> float:
        dist: float = 0
        prevNode: Node = depot

        for nextNode in self.stops + [depot]:
            dist += prevNode.distance(nextNode)

        return dist

    def adjacents(self, i: int) -> Tuple[Node, Node]:
        stops = self.stops
        if i == 0: return (stops[1], DEPOT)
        elif i == len(stops)-1: return (stops[len(stops)-1], DEPOT)
        return (stops[i-1], stops[i+1])

    @property
    def demand(self):
        return sum([stop.demand for stop in self.stops])


class VRPProblem(Problem):
    def __init__(self, numCustomers: int, numTrucks: int, truckCapacity: int, depotNode: Node, file: str = None):
        assert depotNode.demand == 0

        self.file = file

        self.numCustomers: int = numCustomers
        self.numTrucks: int = numTrucks
        self.truckCapacity: int = truckCapacity

        self.depotNode: Node = depotNode
        self.nodes: List[Node] = []

    def check(self, sol: Solution) -> bool: pass

    def objectiveValue(self, sol: Solution) -> float: pass

    def __repr__(self) -> str:
        return f"(Problem <#custs {self.numCustomers}>, <#trucks {self.numTrucks}>, <truckCap {self.truckCapacity}>)"

    def addNode(self, node: Node) -> None:
        self.nodes.append(node)

class VRPSolution(Solution):
    def __init__(self, problem: VRPProblem, routes: List[Route]):
        assert len(routes) == problem.numTrucks  # every truck has a route

        self.problem: VRPProblem = problem
        self.routes: List[Route] = routes

    @classmethod
    def any(cls, prob: VRPProblem):
        return cls(prob, [Route(prob.nodes)]+[Route([])]*(prob.numTrucks-1))

    @property
    def objectiveValue(self) -> float:
        # TODO: may need different multiplier for capOverflow infeasibility penalty
        infeasiblePenalty: float = self.capacityOverflow

        return -(self.distance + infeasiblePenalty)

    @property
    def distance(self) -> float:
        return sum([route.distance(self.problem.depotNode) for route in self.routes])

    def adjacents(self, node: Node) -> Tuple[Node, Node]:
        """Get pair of nodes before+after a node"""
        for route in self.routes:
            stops: List[Node] = route.stops
            i: int = stops.index(node)
            if i >= 0: return route.adjacents(i)
        raise Exception(f"Node not found: {node}")

    def dist(self, other) -> float:
        """Get dist between 2 solutions"""
        total: float = 0
        for (i, route) in enumerate(self.routes):
            for stop in route.stops:
                (pre1, suc1) = route.adjacents(i)
                (pre2, suc2) = other.adjacents(stop)
                # choose the min of both orders to maintain invariance under route reversal:
                total += min(pre1.distance(pre2)+suc1.distance(suc2), # same order
                             pre1.distance(suc2)+suc1.distance(pre2)) # swapped
        return total

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
