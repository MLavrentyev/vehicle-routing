import math
import itertools


class Node:
    def __init__(self, id: int, demand: int, xPos: float, yPos: float):
        self.demand: int = demand
        self.x: float = xPos
        self.y: float = yPos

    def __eq__(self, other: Node) -> bool:
        return self.demand == other.demand and self.x == other.x and self.y == other.y

    def __str__(self):
        return str(self.id)

    def __repr__(self):
        return f"(Node{self.id} <{self.x}, {self.y}>, {self.demand})"

    def distance(self, otherNode: Node) -> float:
        return math.sqrt((self.x - otherNode.x) ** 2 + (self.y - otherNode.y) ** 2)


class Route:
    def __init__(self):
        self.stops: List[Node] = []

    def __str__(self) -> str:
        return f"0 {' '.join([str(n) for n in self.stops])} 0"

    def __repr__(self) -> str:
        return f"(Route {str(self.stops)})"

    def addStop(self, node: Node) -> None:
        self.stops.append(node)


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
        assert solveTime >= 0
        assert len(routes) == problem.numTrucks  # every truck has a route

        self.solveTimeSec: float = solveTimeSec
        self.problem: Problem = problem
        self.routes: List[Route] = routes

    def objectiveValue(self) -> int:
        # TODO: fill in
        return 0

    def isOptimal(self) -> bool:
        # TODO: fill in
        return False