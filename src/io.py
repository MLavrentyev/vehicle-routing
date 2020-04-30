import math


class Node:
    def __init__(self, demand, xPos, yPos):
        self.demand = demand
        self.x = xPos
        self.y = yPos

    def distance(self, otherNode):
        return math.sqrt((self.x - otherNode.x) ** 2 + (self.y - otherNode.y) ** 2)


class Problem:
    def __init__(self, numCustomers, numTrucks, truckCapacity):
        self.numCustomers = numCustomers
        self.numTrucks = numTrucks
        self.truckCapacity = truckCapacity

        self.nodes = []

    def addNode(self, node):
        self.nodes.append(node)


def readInput(file):
    with open(file, mode="r") as vrpFile:
        # read header in
        header = [int(arg) for arg in vrpFile.readline().strip().split()]
        problem = Problem(header[0], header[1], header[2])

        # read node locations in
        for line in vrpFile.readlines():
            values = line.strip().split()
            node = Node(int(values[0]), float(values[1]), float(values[2]))
            problem.addNode(node)

    return problem