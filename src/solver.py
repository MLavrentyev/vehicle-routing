from typing import List, Callable
from abc import ABC
import multiprocessing



class Solver(ABC):
    ...


class VRPSolver(Solver):
    ...



def startMultiProcSolvers(solver: Solver, numProcs: int = multiprocessing.cpu_count()):
    ...


if __name__ == "__main__":
    ...