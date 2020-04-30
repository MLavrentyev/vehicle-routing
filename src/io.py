import math
from problem import Problem, Node, Solution


def readInput(file: str) -> Problem:
    with open(file, mode="r") as vrpFile:
        # read header in
        header: List[int] = [int(arg) for arg in vrpFile.readline().strip().split()]
        depotLine: List[str] = vrpFile.readline()
        depotNode: Node = Node(0, int(depotLine[0]), float(depotLine[1]), float(depotLine[2]))
        problem: Problem = Problem(header[0], header[1], header[2])

        # read node locations in
        nextId: int = 1
        for line in vrpFile.readlines():
            values: List[str] = line.strip().split()
            node: Node = Node(nextId, int(values[0]), float(values[1]), float(values[2]))
            problem.addNode(node)
            nextId += 1

    return problem


def formatSolution(solution: Solution) -> List[str]:
    lines = []
    # add solution header (objective value, is optimal?)
    lines.append(f"{solution.objectiveValue()} {int(solution.isOptimal())}")
    # add route solutions
    for route in solution.routes:
        lines.append(str(route))

    return lines


def writeSolutionToFile(solution: Solution, file: str) -> None:
    lines: List[str] = formatSolution(solution)

    with open(file, mode="w") as solFile:
        solFile.writelines(lines)


def printSolution(solution: Solution) -> None:
    lines: List[str] = formatSolution(solution)

    print(f"Instance: {solution.problem.file}"
          f" Time: {solution.solveTimeSec}"
          f" Result: {solution.objectiveValue()}"
          f" Solution {int(solution.isOptimal())} {' '.join(lines[1:])}")

