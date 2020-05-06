import random
import sys
import time

import vrpIo
from problem import VRPProblem, VRPSolution
from solver import VRPSolver

from vis import plot, init

### Run: python3 src/main.py simpleInput/21_4_1.vrp

if __name__ == '__main__':
    # print(f"sys.argv[1]: {sys.argv[1]}")
    prob: VRPProblem = vrpIo.readInput(sys.argv[1])
    print(f"problem: {prob}")

    def display():
        print(f"sol\t\t: {sol} >> {sol.objectiveValue}")
        plot(sol)

    init()
    sol: VRPSolution = VRPSolution.rand(prob).normalize()
    display()
    for _ in range(20):
        time.sleep(0.25)
        neighs = sol.neighbors()+[sol]
        neighs.sort(key=lambda n: n.objectiveValue)
        best = neighs[0]
        if best == sol:
            print("no better neighbor")
        else:
            sol = best
            display()


