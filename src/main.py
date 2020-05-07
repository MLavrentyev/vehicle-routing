from typing import List
import random
import sys
import time
import vrpIo
from problem import VRPProblem, VRPSolution
from solver import VRPSolver
from vis import plot, init

# Run: python3 src/main.py simpleInput/21_4_1.vrp

def display(solution: VRPSolution):
    print(f"sol\t\t: {solution} >> {solution.objectiveValue}")
    plot(solution)

if __name__ == '__main__':
    problem: VRPProblem = vrpIo.readInput(sys.argv[1])
    print(f"problem: {problem}")

    init()
    sol: VRPSolution = VRPSolution.rand(problem).normalize()
    display(sol)

    for _ in range(20):
        time.sleep(0.25)
        neighbs: List[VRPSolution] = sol.neighbors() + [sol]
        neighbs.sort(key = (lambda n: n.distance))

        best = neighbs[0]
        if best == sol:
            print("no better neighbor")
        else:
            sol = best
            display(sol)


