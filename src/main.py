import sys
import time

import vrpIo
from problem import VRPProblem, VRPSolution
from solver import VRPSolver

from vis import plot, init

### Run: python3 src/main.py simpleInput/16_5_1.vrp

if __name__ == '__main__':
    # print(f"sys.argv[1]: {sys.argv[1]}")
    prob: VRPProblem = vrpIo.readInput(sys.argv[1])
    print(f"problem: {prob}")

    init()
    plot(VRPSolution.rand(prob))
    for _ in range(4):
        time.sleep(0.5)
        plot(VRPSolution.rand(prob))



