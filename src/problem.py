from typing import List
import math
import itertools


class Node:
    def __init__(self, nodeId: int, demand: int, xPos: float, yPos: float):
        self.id = nodeId
        self.demand: int = demand
        self.x: float = xPos
        self.y: float = yPos

    def __eq__(self, otherNode) -> bool:
        return self.demand == other.demand and self.x == other.x and self.y == other.y

    def __str__(self):
        return str(self.id)

    def __repr__(self):
        return f"(Node{self.id} <{self.x}, {self.y}>, {self.demand})"

    def distance(self, otherNode) -> float:
        return math.sqrt((self.x - otherNode.x) ** 2 + (self.y - otherNode.y) ** 2)


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


class Problem:
    def __init__(self, numCustomers: int, numTrucks: int, truckCapacity: int, depotNode: Node, file: str = None):
        assert depotNode.demand == 0

        self.file = file

        self.numCustomers: int = numCustomers
        self.numTrucks: int = numTrucks
        self.truckCapacity: int = truckCapacity

        self.depotNode: Node = depotNode
        self.nodes: List[Node] = []

    def __repr__(self) -> str:
        return f"(Problem <#custs {self.numCustomers}>, <#trucks {self.numTrucks}>, <truckCap {self.truckCapacity}>)"

    def addNode(self, node: Node) -> None:
        self.nodes.append(node)


class Solution:
    def __init__(self, problem: Problem, routes: List[Route], solveTimeSec: float):
        assert solveTimeSec >= 0
        assert len(routes) == problem.numTrucks  # every truck has a route

        self.solveTimeSec: float = solveTimeSec
        self.problem: Problem = problem
        self.routes: List[Route] = routes

    def objectiveValue(self) -> float:
        totalDist: float = 0
        for route in self.routes:
            totalDist += route.distance(self.problem.depotNode)

        return totalDist

    def isOptimal(self) -> bool:
        # TODO: fill in
        return False

    def solveTime(self) -> float:
        return self.solveTimeSec
