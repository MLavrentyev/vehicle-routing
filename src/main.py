import sys

import vrpIo
from problem import VRPProblem, VRPSolution
from solver import VRPSolver

from vis import plot

### Run: python src/main.py simpleInput/5_4_10.vrp

if __name__ == '__main__':
    # print(f"sys.argv[1]: {sys.argv[1]}")
    prob: VRPProblem = vrpIo.readInput(sys.argv[1])
    print(f"problem: {prob}")
    sol: VRPSolution = VRPSolution.any(prob)
    print(f"sol: {sol}")
    plot(sol)
