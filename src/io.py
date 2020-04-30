import math
from problem import Problem, Node, Solution


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


def printSolution(solution, file=None):
    lines = []
    # add solution header (objective value, is optimal?)
    lines.append(f"{solution.objectiveValue()} {int(solution.isOptimal())}")
    # add route solutions
    for route in solution.routes:
        routeNodeIds = [node.id for node in route.stops]
        lines.append(f"0 {' '.join(routeNodeIds)} 0")

    # either write to file or print to command line
    if file:
        with open(file, mode="w") as solFile:
            solFile.writelines(lines)
    else:
        print(f"Instance: {solution.problem.file}"
              f" Time: {solution.solveTimeSec}"
              f" Result: {solution.objectiveValue()}"
              f" Solution {int(solution.isOptimal())} {' '.join(lines[1:])}")
