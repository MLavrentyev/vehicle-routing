from typing import List
import random
import sys
import time
import vrpIo
from problem import VRPProblem, VRPSolution
from solver import VRPSolver
import vis


if __name__ == '__main__':
    problem: VRPProblem = vrpIo.readInput(sys.argv[1])
    print(f"problem: {problem}")

    solver = VRPSolver()

    vis.init()
    sol: VRPSolution = solver.pickInitSolution().normalize()
    vis.display(sol)

    for _ in range(20):
        neighbs: List[VRPSolution] = sol.neighbors() + [sol]
        neighbs.sort(key = (lambda n: n.distance))

        best = neighbs[0]
        if best == sol:
            print("no better neighbor")
        else:
            sol = best
            vis.display(sol)


