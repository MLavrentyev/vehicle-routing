from typing import cast, List, Callable, Generator, Tuple
from abc import ABC, abstractmethod
from multiprocessing import Process, Queue, cpu_count
from queue import Empty
import sys
import time
import random
import vis
import vrpIo
from problem import Route, Problem, VRPProblem, Solution, VRPSolution



class Solver(ABC):

    @classmethod
    def factory(cls, problem: Problem):
        solver = cls()
        solver.setProblem(problem)
        return solver

    def setProblem(self, problem: Problem) -> None:
        self.problem: Problem = problem

    def solveWithQueue(self, queue: Queue, display: bool = False, maximizeObjV: bool = True) -> None:
        solution: Solution = self.solve(display=display, maximizeObjV=maximizeObjV)
        queue.put(solution)

    @abstractmethod
    def solve(self, display: bool, maximizeObjV: bool = True) -> Solution:
        pass

    @abstractmethod
    def neighborhood(self, solution: Solution) -> Generator[Solution, None, None]:
        pass

    @abstractmethod
    def pickInitSolution(self) -> Solution:
        pass

    @abstractmethod
    def pickAnySolution(self) -> Solution:
        pass


class VRPSolver(Solver):

    def solve(self, display: bool = False, maximizeObjV: bool = True) -> VRPSolution:
        currState: VRPSolution = self.pickInitSolution()
        improveCheck: Callable[[float, float], bool] = (lambda o, c: o > c) if maximizeObjV else (lambda o, c: o < c)

        if display:
            vis.init()
            vis.display(currState)

        done: bool = False
        while not done:
            done = True
            # II - look for better neighbor (if none, we are done)
            neighbSolution: VRPSolution
            for neighbSolution in self.neighborhood(currState):
                if improveCheck(neighbSolution.objectiveValue, currState.objectiveValue):
                    currState = neighbSolution
                    done = False
                    break

            if display and not done:
                vis.display(currState)

        return currState

    def neighborhood(self, solution: Solution) -> Generator[VRPSolution, None, None]:
        solution = cast(VRPSolution, solution)

        for neighbor in solution.neighbors():
            yield neighbor

    def pickInitSolution(self) -> VRPSolution:
        problem = cast(VRPProblem, self.problem)

        nodes = problem.nodes[:]
        random.shuffle(nodes)

        routes = []
        startIdx = 0
        for _ in range(problem.numTrucks):
            endIdx = random.randint(0, problem.numCustomers - startIdx)
            routes.append(Route(nodes[startIdx:endIdx], problem.depot))
            startIdx = endIdx

        return VRPSolution(problem, routes)

    def pickAnySolution(self) -> VRPSolution:
        indss = [range(i, len(problem.nodes), problem.numTrucks) for i in range(problem.numTrucks)]

        return VRPSolution(problem, [Route([problem.nodes[i] for i in inds], problem.depot) for inds in indss])


def initSolverProcs(solverFactory: Callable[[Problem], Solver], numSolvers: int, queueConn: Queue,
                    factoryArgs: Tuple = (), solveArgs: Tuple = ()) -> Tuple[List[Solver], List[Process]]:
    solvers: List[Solver] = []
    solverProcs: List[Process] = []
    for _ in range(numSolvers):
        newSolver: Solver = solverFactory(*factoryArgs)
        solvers.append(newSolver)
        solverProcs.append(Process(target=newSolver.solveWithQueue, args=(queueConn,) + solveArgs))

    return solvers, solverProcs


def runMultiProcSolver(solverFactory: Callable[[Problem], Solver], problem: Problem,
                       solveArgs: Tuple = (), numProcs: int = cpu_count()) -> Tuple[Solution, float]:
    queueConn: Queue = Queue()
    startTime: float = time.time()

    solvers: List[Solver]
    solverProcs: List[Process]
    solvers, solverProcs = initSolverProcs(solverFactory, numProcs, queueConn, factoryArgs=(problem,), solveArgs=solveArgs)

    # do o restart, with growing time increments
    timeout: int = 30
    timeoutMult: int = 2
    for solverProc in solverProcs:
        solverProc.start()
    while True:
        try:
            solution: Solution = queueConn.get(block=True, timeout=None) #TODO:fix
            break
        except Empty:
            # kill solver and restart
            timeout *= timeoutMult
            print(f"Restarting solver with timeout {timeout:.2f}s")

            for solverProc in solverProcs:
                solverProc.terminate()

            solvers, solverProcs = initSolverProcs(solverFactory, numProcs, queueConn, factoryArgs=(problem,), solveArgs=solveArgs)
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
    solution, solveTime = cast(Tuple[VRPSolution, float],
                               runMultiProcSolver(VRPSolver.factory, problem, solveArgs=(False, False), numProcs=1))

    if len(sys.argv) == 4 and sys.argv[2] == "-f":
        vrpIo.writeSolutionToFile(solution, sys.argv[3])
    elif len(sys.argv) == 2:
        vrpIo.printSolution(solution, solveTime)
    else:
        raise IOError("Incorrect arguments passed.")

