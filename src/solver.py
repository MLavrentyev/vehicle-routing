from typing import List, Callable, Generator, Tuple
from abc import ABC, abstractmethod
from multiprocessing import Process, Queue, cpu_count
import sys
import time

import vrpIo
from problem import Problem, VRPProblem, Solution, VRPSolution



class Solver(ABC):

    @classmethod
    @abstractmethod
    def factory(cls, problem: Problem, startTime: float):
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


def initSolverProcs(solverFactory: Callable[[Problem, float], Solver], numSolvers: int, factoryArgs: Tuple = ()) -> Tuple[List[Solver], List[Process]]:
    solvers: List[Solver] = []
    solverProcs: List[Process] = []
    for _ in range(numSolvers):
        newSolver: Solver = solverFactory(*factoryArgs)
        solvers.append(newSolver)
        solverProcs.append(Process(target=newSolver.solve, args=()))

    return solvers, solverProcs


def runMultiProcSolver(solverFactory: Callable[[], Solver], problem: Problem, numProcs: int = cpu_count()) -> Solution:
    queueConn: Queue = Queue()
    startTime: float = time.time()

    solvers: List[Solver]
    solverProcs: List[Process]
    solvers, solverProcs = initSolverProcs(solverFactory, numProcs, factoryArgs=(problem, startTime))

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
            solvers, solverProcs = initSolverProcs(solverFactory, numProcs, factoryArgs=(problem, startTime))
            for solverProc in solverProcs:
                solverProc.start()

    # clean up solvers and return
    for solverProc in solverProcs:
        solverProc.terminate()

    return solution



if __name__ == "__main__":
    problem: VRPProblem = vrpIo.readInput(sys.argv[1])
    solution: Solution = runMultiProcSolver(VRPSolver.factory, problem)

    if len(sys.argv) == 4 and sys.argv[2] == "-f":
        vrpIo.writeSolutionToFile(solution, sys.argv[3])
    elif len(sys.argv) == 2:
        vrpIo.printSolution(solution)
    else:
        raise IOError("Incorrect arguments passed.")

