from typing import cast, List, Callable, Generator, Tuple
from abc import ABC, abstractmethod
from multiprocessing import Process, Queue, cpu_count
from queue import Empty
import sys
import time
import random

import vrpIo
from problem import Problem, VRPProblem, Solution, VRPSolution



class Solver(ABC):

    @classmethod
    def factory(cls, problem: Problem):
        solver = cls()
        solver.setProblem(problem)
        return solver

    def setProblem(self, problem: Problem) -> None:
        self.problem: Problem = problem

    @abstractmethod
    def solve(self) -> Solution:
        pass

    @abstractmethod
    def neighborhood(self, solution: Solution) -> Generator[Solution, None, None]:
        pass


class VRPSolver(Solver):

    def solve(self) -> VRPSolution:
        currState: VRPSolution = VRPSolver.pickInitSolution()

        done: bool = False
        while not done:
            done = True
            # II - look for better neighbor (if none, we are done)
            neighbSolution: VRPSolution
            for neighbSolution in self.neighborhood(currState):
                if neighbSolution.objectiveValue >= currState.objectiveValue:
                    currState = neighbSolution
                    done = False
                    break

        return currState

    def neighborhood(self, solution: Solution) -> Generator[VRPSolution, None, None]:
        solution = cast(VRPSolution, solution)
        yield solution # TODO: fill in

    def pickInitSolution(self) -> VRPSolution:
        problem = cast(VRPProblem, self.problem)

        nodes = copy(problem.nodes)
        random.shuffle(nodes)

        routes = []
        startIdx = 0
        for _ in range(problem.numCustomers):
            endIdx = random.randint(0, problem.numCustomers - startIdx)
            routes.append(nodes[startIdx:endIdx])
            startIdx = endIdx

        return VRPSolution(problem, routes)


def initSolverProcs(solverFactory: Callable[[Problem], Solver], numSolvers: int, factoryArgs: Tuple = ()) -> Tuple[List[Solver], List[Process]]:
    solvers: List[Solver] = []
    solverProcs: List[Process] = []
    for _ in range(numSolvers):
        newSolver: Solver = solverFactory(*factoryArgs)
        solvers.append(newSolver)
        solverProcs.append(Process(target=newSolver.solve, args=()))

    return solvers, solverProcs


def runMultiProcSolver(solverFactory: Callable[[Problem], Solver], problem: Problem, numProcs: int = cpu_count()) -> Tuple[Solution, float]:
    queueConn: Queue = Queue()
    startTime: float = time.time()

    solvers: List[Solver]
    solverProcs: List[Process]
    solvers, solverProcs = initSolverProcs(solverFactory, numProcs, factoryArgs=(problem,))

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
            solvers, solverProcs = initSolverProcs(solverFactory, numProcs, factoryArgs=(problem,))
            for solverProc in solverProcs:
                solverProc.start()

    # clean up solvers and return
    for solverProc in solverProcs:
        solverProc.terminate()

    return solution, (time.time() - startTime)



if __name__ == "__main__":
    problem: VRPProblem = vrpIo.readInput(sys.argv[1])

    solution: VRPSolution
    solveTime: float
    solution, solveTime = cast(Tuple[VRPSolution, float], runMultiProcSolver(VRPSolver.factory, problem))

    if len(sys.argv) == 4 and sys.argv[2] == "-f":
        vrpIo.writeSolutionToFile(solution, sys.argv[3])
    elif len(sys.argv) == 2:
        vrpIo.printSolution(solution, solveTime)
    else:
        raise IOError("Incorrect arguments passed.")

