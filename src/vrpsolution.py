import random
from typing import List

from problem import Solution, Node, VRPProblem

class PNode:
    """Wraps a Node in pointy goodness to ward off slowness"""
    def __init__(self, node: Node):
        self.node = node
        self.L = None
        self.R = None
        self.Ldist = 0  # distance to the left
        self.Rdist = 0  # distance to the right
        self.Ldem  = 0  # demand to the left
        self.Rdem  = 0  # demand to the right
        # TODO: all ops could be made log-time by maintaining pointers 2^c to the left/right
    def setL(self, L):
        self.L = L
        return self
    def setR(self, R):
        self.R = R
        return self
    def setLdist(self, Ldist):
        self.Ldist = Ldist
        return self
    def setRdist(self, Rdist):
        self.Rdist = Rdist
        return self
    def setLdem(self, Ldem):
        self.Ldem = Ldem
        return self
    def setRdem(self, Rdem):
        self.Rdem = Rdem
        return self
    def totDist(self): return self.Ldist + self.Rdist
    def totDem(self): return self.Ldem + self.Rdem + self.node.demand
    def wayL(self): pass
    def wayR(self): pass
    def fix(self): pass
    def fixL(self, n, c=0): pass
    def fixR(self, n, c=0): pass

class VRPSolutionPointy(Solution):
    """
    Represents solutions with pointers between nodes.
    Is able to detect a when a change makes things worse in constant time.
    Is able to make that change in time linear of route length. (log-time is possible)
    """
    def __init__(self, problem: VRPProblem, routes: List[List[Node]]):
        self.pnodes = []
        self.lmost = set()
        self.totDist = 0    # total distance
        self.totOver = 0    # total amount over demand
        for route in routes:
            pn = PNode(route[0])
            self.lmost.add(pn)
            self.pnodes.append(pn)

            # update from left to right
            dist = 0
            dem = 0
            for node in route[1:]:
                dist += node.distance(pn.node)
                pn = PNode(node).setL(pn).setLdist(dist).setLdem(dem)
                dem += node.demand
                self.pnodes.append(pn)

            # update from right to left
            dist = 0
            dem = 0
            for node in route[:-1]:
                dist += node.distance(pn.node)
                pn = pn.L.setR(pn).setRist(dist).setRdem(dem)
                dem += node.demand

            self.totDist += pn.totDist()
            self.totOver += max(0, pn.totDem() - problem.truckCapacity)

    @staticmethod
    def rand(problem: VRPProblem) -> 'Solution':
        nodes = problem.nodes[:]
        random.shuffle(nodes)
        indss = [range(i, len(problem.nodes), problem.numTrucks) for i in range(problem.numTrucks)]
        return VRPSolutionPointy(problem, [[nodes[i] for i in inds] for inds in indss])

    def becomeNeighbor(self, probWorse: float) -> None:
        """mutate to a random neighbor, but if worse, only with some probability"""
        pass
