from typing import cast, List, Callable, Generator, Tuple
from abc import ABC, abstractmethod
from multiprocessing import Process, Queue, cpu_count
from queue import Empty
import sys
import time
import random
import vis
import vrpIo
import itertools
import math
from problem import Route, Node, Problem, VRPProblem, Solution, VRPSolution, getClosestNode


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
    def pickRandomSolution(self) -> Solution:
        pass

    @abstractmethod
    def pickAnySolution(self) -> Solution:
        pass


class VRPSolver(Solver):

    def solve(self, display: bool = False, maximizeObjV: bool = True) -> VRPSolution:
        annealingTemp: float = 0.97
        problem = cast(VRPProblem, self.problem)

        currState: VRPSolution = self.pickSectoredSolution()
        improveCheck: Callable[[float, float], bool] = (lambda o, c: o > c) if maximizeObjV else (lambda o, c: o < c)

        if display:
            vis.init()

        numBadSteps: int = 0
        numAccSteps: int = 0
        numSteps: int = 0
        while not (numBadSteps >= 5 and (numSteps > 0 and numAccSteps/numSteps > 0.02) and currState.isFeasible()):
            # plot display of current solution
            if display:
                vis.display(currState, doPlot=False)

            neighb: VRPSolution = currState.randomNeighbor()
            if (improveCheck(neighb.objectiveValue, currState.objectiveValue)) or\
               (random.random() <= math.exp((neighb.objectiveValue - currState.objectiveValue) / annealingTemp)):
                currState = neighb
                numBadSteps = 0 if improveCheck(neighb.objectiveValue, currState.objectiveValue) else numBadSteps + 1
                numAccSteps += 1

            numSteps += 1
            # Anneal temperature every certain number of steps
            if numSteps % (problem.numCustomers * (problem.numCustomers - 1)) == 0:
                annealingTemp *= 0.95

        return currState

    def pickRandomSolution(self) -> VRPSolution:
        problem = cast(VRPProblem, self.problem)

        nodes = problem.nodes[:]
        random.shuffle(nodes)

        routes: List[Route] = []
        startIdx = 0
        for _ in range(problem.numTrucks - 1):
            endIdx = random.randint(startIdx, problem.numCustomers)
            routes.append(Route(nodes[startIdx:endIdx], problem.depot))
            startIdx = endIdx

        routes.append(Route(nodes[startIdx:], problem.depot))

        return VRPSolution(problem, routes)

    def pickSectoredSolution(self) -> VRPSolution:
        # Pick an initial solution where each truck is limited to routing among nodes in a given pie slice
        problem = cast(VRPProblem, self.problem)

        # slices divided ccw starting from positive x-axis (where depot is origin)
        sliceSizeRad: float = (2 * math.pi) / problem.numTrucks

        sectors: List[List[Node]] = [[] for _ in range(problem.numTrucks)]
        node: Node
        for node in problem.nodes:
            # angle from 0 to 2pi
            nodeAngle: float = math.atan2(node.y - problem.depot.y, node.x - problem.depot.x) + math.pi
            sectorIdx: int = math.floor(nodeAngle / sliceSizeRad)
            sectors[sectorIdx].append(node)

        # Generate greedy routing for each sector
        routes: List[Route] = []
        sector: List[Node]
        for sector in sectors:
            route: Route = Route(sector, problem.depot)
            route.greedyReorder()
            routes.append(route)

        return VRPSolution(problem, routes)

    def pickAnySolution(self) -> VRPSolution:
        problem: VRPProblem = cast(VRPProblem, self.problem)

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
            solution: Solution = queueConn.get(block=True, timeout=None) #TODO:fix to set timeout
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
                               runMultiProcSolver(VRPSolver.factory, problem, solveArgs=(True, False), numProcs=1))

    if len(sys.argv) == 4 and sys.argv[2] == "-f":
        vrpIo.writeSolutionToFile(solution, sys.argv[3])
    elif len(sys.argv) == 2:
        vrpIo.printSolution(solution, solveTime)
    else:
        raise IOError("Incorrect arguments passed.")

