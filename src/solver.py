from typing import cast, List, Callable, Generator, Tuple, Type
from abc import ABC, abstractmethod
from multiprocessing import Process, Queue as ProcQueue, cpu_count
from queue import Empty, Queue
import sys
import time
import random
import platform
import vrpIo
import itertools
import math
from problem import Route, Node, Problem, VRPProblem, Solution, VRPSolution, getClosestNode, VRPSolution2Op


class Solver(ABC):

    @classmethod
    def factory(cls, problem: Problem):
        solver = cls()
        solver.setProblem(problem)
        return solver

    def setProblem(self, problem: Problem) -> None:
        self.problem: Problem = problem

    def solveWithQueue(self, queue: ProcQueue, display: bool = False, maximizeObjV: bool = True) -> None:
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
        doRandJumpProb: float = 0.1

        currState: VRPSolution = self.pickRandomSolution()
        # currState = VRPSolution2Op(currState.problem, currState.routes)

        improveCheck: Callable[[float, float], bool] = (lambda o, c: o > c) if maximizeObjV else (lambda o, c: o < c)

        if display:
            if platform.python_implementation() != "PyPy":
                import vis
                vis.init()

        done: bool = False
        numSteps: int = 1
        while not done:
            done = True

            # do random jump with some probability
            if random.random() <= doRandJumpProb:
                currState = currState.randomNeighbor()
                done = False
            else:
                # iterative improvement - look for better neighbor (if none, we are done)
                neighbSolution: VRPSolution
                for neighbSolution in currState.neighbors():
                    if improveCheck(neighbSolution.objectiveValue, currState.objectiveValue):
                        currState = neighbSolution
                        done = False
                        break
                # if the current state is infeasible and we didn't improve this round, randomly pick a neighbor
                if not currState.isFeasible() and done:
                    currState = currState.randomNeighbor()
                    done = False
            numSteps += 1

            if display:
                if numSteps % 10000 == 0:
                    if platform.python_implementation() != "PyPy":
                        vis.display(currState, doPlot=False)
                    else:
                        VRPSolver2OpSimAnneal.printState(numSteps, currState)

        return currState

    #TODO: make these static methods in the Problem class
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
            sectorIdx: int = math.floor(nodeAngle / sliceSizeRad) % problem.numTrucks
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

    @staticmethod
    def printState(step: int, currState: VRPSolution) -> None:
        print(f"Step: {step:,} ".ljust(25) +
              f"Score: {currState.objectiveValue:.2f} ".ljust(25) +
              f"(d: {currState.totalDistance:.2f}, inf: {currState.capacityOverflow:.2f})")

    @staticmethod
    def addToTabu(tabu: List[VRPSolution], state: VRPSolution, tabuMaxSize: int) -> List[VRPSolution]:
        if len(tabu) == tabuMaxSize:
            tabu.pop(0)
            tabu.append(state)
        else:
            tabu.append(state)

        return tabu


class VRPSolver2OpSimAnneal(VRPSolver):

    def solve(self, display: bool = False, maximizeObjV: bool = True) -> VRPSolution:
        problem = cast(VRPProblem, self.problem)

        annealingTemp: float = 1  # higher values promote more wandering at the beginning
        annealingSched: float = 0.65  # lower values go more directly to the minimum. higher values wander more
        annealingTime: int = (problem.numCustomers * (problem.numCustomers - 1)) // 2

        currState: VRPSolution = self.pickRandomSolution()
        improveCheck: Callable[[float, float], bool] = (lambda o, c: o > c) if maximizeObjV else (lambda o, c: o < c)

        if display:
            if platform.python_implementation() != "PyPy":
                import vis
                vis.init()

        scoreHistory: List[float] = []
        historySize: int = 1000  # speed vs. quality tradeoff (higher value is slower but better solution)
        numBadSteps: int = 0
        numAccSteps: int = 0
        numSteps: int = 1
        #  and numAccSteps/numSteps < 0.25
        while not (numBadSteps >= 5 and currState.isFeasible() and (max(scoreHistory) - min(scoreHistory) < 0.001 * max(scoreHistory))):

            neighb: VRPSolution = currState.randomNeighbor()
            worseProb: float = math.exp(min((currState.objectiveValue - neighb.objectiveValue) / annealingTemp, 1.0))
            # Pick neighbor and decide whether to move there
            if (improveCheck(neighb.objectiveValue, currState.objectiveValue)) or (random.random() <= worseProb):
                numBadSteps = 0 if improveCheck(neighb.objectiveValue, currState.objectiveValue) else numBadSteps + 1
                numAccSteps += 1
                currState = neighb

                # adjust score history with new step
                scoreHistory.append(currState.objectiveValue)
                if len(scoreHistory) > historySize:
                    scoreHistory.pop(0)

            numSteps += 1
            # Anneal temperature every certain number of steps
            if numSteps % annealingTime == 0:
                annealingTemp *= annealingSched
                if display:
                    print(f"Annealing temperature. Acceptance rate: {numAccSteps/numSteps:.2f}")

            # Check whether it's stuck in an infeasible minimum and jump out
            if (not currState.isFeasible()) and len(scoreHistory) == historySize and (max(scoreHistory) - min(scoreHistory) < 0.0001 * max(scoreHistory)):
                currState = self.pickRandomSolution()
                if display:
                    print("Randomizing to new solution.")

            # plot display of current solution
            if display:
                if numSteps % 1e4 == 0:
                    if platform.python_implementation() != "PyPy":
                        vis.display(currState, doPlot=False)
                    else:
                        VRPSolver2OpSimAnneal.printState(numSteps, currState)

        print(f"Solution found after {numAccSteps:,} accepted steps. Score: {currState.objectiveValue:.2f}")
        return currState

    def pickRandomSolution(self) -> VRPSolution2Op:
        baseSolution: VRPSolution = super().pickRandomSolution()
        return VRPSolution2Op(baseSolution.problem, baseSolution.routes)

    def pickSectoredSolution(self) -> VRPSolution2Op:
        baseSolution: VRPSolution = super().pickSectoredSolution()
        return VRPSolution2Op(baseSolution.problem, baseSolution.routes)

    def pickAnySolution(self) -> VRPSolution2Op:
        baseSolution: VRPSolution = super().pickAnySolution()
        return VRPSolution2Op(baseSolution.problem, baseSolution.routes)


def initSolverProcs(solverFactory: Callable[[Problem], Solver], numSolvers: int, queueConn: ProcQueue,
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
    queueConn: ProcQueue = ProcQueue()
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
    solverType: Type[Solver] = VRPSolver2OpSimAnneal

    solution: VRPSolution
    solveTime: float
    solution, solveTime = cast(Tuple[VRPSolution, float],
                               runMultiProcSolver(solverType.factory, problem, solveArgs=(False, False), numProcs=7))
    # solution, solveTime = cast(VRPSolution, solverType.factory(problem).solve(True, False)), 0 # only for profiling

    if len(sys.argv) == 4 and sys.argv[2] == "-f":
        vrpIo.writeSolutionToFile(solution, sys.argv[3])
    elif not len(sys.argv) == 2:
        raise IOError("Incorrect arguments passed.")

    vrpIo.printSolution(solution, solveTime)
