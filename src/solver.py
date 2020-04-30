from typing import List, Callable, Generator, Tuple
from abc import ABC, abstractmethod
from multiprocessing import Process, Queue
import sys

import vrpIo
from problem import Problem, Solution



class Solver(ABC):

    @classmethod
    @abstractmethod
    def factory(cls, startTime: float):
        solver = cls()
        solver.setStartTime(startTime)
        return solver

    def setStartTime(self, startTime: float) -> None:
        self.startTime = startTime

    @abstractmethod
    def solve(self) -> Solution:
        pass

    @abstractmethod
    def neighborhood(self, solution: Solution) -> Generator[Solution]:
        pass

    @abstractmethod
    def done(self) -> bool:
        pass


class VRPSolver(Solver):
    ...


def initSolverProcs(solverFactory: Callable[[], Solver], numSolvers: int, factoryArgs: Tuple = ()) -> Tuple[List[Solver], List[Process]]:
    solvers: List[Solver] = []
    solverProcs: List[Process] = []
    for _ in range(numSolvers):
        newSolver: Solver = solverFactory()
        solvers.append(newSolver)
        solverProcs.append(Process(target=newSolver.solve, args=factoryArgs))

    return solvers, solverProcs


def runMultiProcSolver(solverFactory: Callable[[], Solver], numProcs: int = multiprocessing.cpu_count()) -> Solution:
    queueConn: Queue = Queue()
    startTime: float = time()

    solvers: List[Solver]
    solverProcs: List[Process]
    solvers, solverProcs = initSolverProcs(solverFactory, numProcs, factoryArgs=(startTime,))

    # do a restart, with growing time increments
    timeout: int = 30
    timeoutMult: int = 2
    for solverProc in solverProcs:
        solverProc.start()
    while True:
        try:
            solution: Solution = queueConn.get(block=True, timeout=timeout)
            break
        except Empty:
            # kill solver and restart
            timeout *= timeoutMult
            print(f"Restarting solver with timeout {timeout:.2f}s")

            for solverProc in solverProcs:
                solverProc.terminate()
            solvers, solverProcs = initSolverProcs(solverFactory, numProcs, factoryArgs=(startTime,))
            for solverProc in solverProcs:
                solverProc.start()

    # clean up solvers and return
    for solverProc in solverProcs:
        solverProc.terminate()

    return solution



if __name__ == "__main__":
    problem: Problem = vrpIo.readInput(sys.argv[1])
    solution: Solution = runMultiProcSolver(...)

    if len(sys.argv) == 4 and sys.argv[2] == "-f":
        vrpIo.writeSolutionToFile(solution, sys.argv[3])
    elif len(sys.argv) == 2:
        vrpIo.printSolution(solution)
    else:
        raise IOError("Incorrect arguments passed.")

