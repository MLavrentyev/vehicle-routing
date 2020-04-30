import math
import itertools


class Node:
    def __init__(self, id, demand, xPos, yPos):
        self.demand = demand
        self.x = xPos
        self.y = yPos

    def __eq__(self, other):
        return self.demand == other.demand and self.x == other.x and self.y == other.y

    def distance(self, otherNode):
        return math.sqrt((self.x - otherNode.x) ** 2 + (self.y - otherNode.y) ** 2)


origin = Node(0, 0, 0)


class Route:
    def __init__(self):
        self.stops = []

    def addStop(self, node):
        if self.stops:
            assert self.stops[0] == origin
            self.stops.insert(-1, node)
        else:
            self.stops.append(origin)
            self.stops.append(node)
            self.stops.append(origin)


class Problem:
    def __init__(self, numCustomers, numTrucks, truckCapacity, file=None):
        self.file = file

        self.numCustomers = numCustomers
        self.numTrucks = numTrucks
        self.truckCapacity = truckCapacity

        self.nodes = []

    def addNode(self, node):
        self.nodes.append(node)


class Solution:
    def __init__(self, problem, routes, solveTimeSec):
        assert solveTime >= 0
        assert len(routes) == problem.numTrucks  # every truck has a route

        self.solveTimeSec = solveTimeSec
        self.problem = problem
        self.routes = routes

    def objectiveValue(self):
        # TODO: fill in
        return 0

    def isOptimal(self):
        # TODO: fill in
        return False