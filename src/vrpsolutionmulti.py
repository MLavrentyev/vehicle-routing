import random

from typing import List

from problem import VRPSolution2Op, Node, Route


class VRPSolutionMulti(VRPSolution2Op):
    """Neighbor function does some operations which are pairs of 2-opts"""
    # noinspection DuplicatedCode
    def randomNeighbor(self) -> 'VRPSolutionMulti':
        # TODO: accept parameters adjusting weights
        # TODO: keep track of dist/demand on either side of nodes to avoid bad edges

        r = random.randrange(3)
        if r == 0:   return self.rand2Opt()   # most change / early
        elif r == 1: return self.randTake()   # middle
        else:        return self.randSwap()   # least change / late

    # noinspection DuplicatedCode
    def randSwap(self) -> 'VRPSolutionMulti':
        """AXB+CYD <~> AYB+CXD"""
        # print("randSwap")
        routes = self.routes
        idx1: int = random.randrange(len(routes))      # not weighted
        idx2: int = random.randrange(len(routes))      # not weighted
        if idx1 == idx2:
            route: List[Node] = routes[idx1].stops
            if not route: return self
            node1 = random.randrange(len(route))
            node2 = random.randrange(len(route))
            if node1 > node2: node1, node2 = node2, node1
            L, M, R = route[:node1], route[node1+1:node2], route[node2+1:]
            new = Route(L+[route[node2]]+M+[route[node1]]+R, self.depot)
            return VRPSolutionMulti(self.problem, routes[:idx1]+[new]+routes[idx1+1:])
        else:
            if idx1 > idx2: idx1, idx2 = idx2, idx1
            route1: List[Node] = routes[idx1].stops
            route2: List[Node] = routes[idx2].stops
            if not route1 or not route2: return self
            node1 = random.randrange(len(route1))
            node2 = random.randrange(len(route2))
            L1, R1 = route1[:node1], route1[node1+1:]
            L2, R2 = route2[:node2], route2[node2+1:]
            new1 = Route(L1+[route2[node2]]+R1, self.depot)
            new2 = Route(L2+[route1[node1]]+R2, self.depot)
            return VRPSolutionMulti(self.problem, routes[:idx1]+[new1]+routes[idx1+1:idx2]+[new2]+routes[idx2+1:])

    # noinspection DuplicatedCode
    def randTake(self) -> 'VRPSolutionMulti':
        """AXB+CD <~> AB+CXD"""
        # print("randTake")
        # this one's a bit trickier than swap/2opt because of the extra asymmetry
        routes = self.routes
        idx1: int = random.randrange(len(routes))      # not weighted
        idx2: int = random.randrange(len(routes))      # not weighted
        if idx1 == idx2:
            route: List[Node] = routes[idx1].stops
            if not route: return self
            edge1 = random.randrange(len(route)+1)
            node2 = random.randrange(len(route))
            new: Route
            if edge1 < node2:
                L, M, R = route[:edge1], route[edge1:node2], route[node2+1:]
                new = Route(L+[route[node2]]+M+R, self.depot)
            else:
                L, M, R = route[:node2], route[node2+1:edge1], route[edge1:]
                new = Route(L+M+[route[node2]]+R, self.depot)
            return VRPSolutionMulti(self.problem, routes[:idx1]+[new]+routes[idx1+1:])
        else:
            route1: List[Node] = routes[idx1].stops
            route2: List[Node] = routes[idx2].stops
            if not route2:
                if not route1: return self
                idx1, idx2, route1, route2 = idx2, idx1, route2, route1
            edge1 = random.randrange(len(route1)+1)
            node2 = random.randrange(len(route2))
            L1, R1 = route1[:edge1], route1[edge1:]
            L2, R2 = route2[:node2], route2[node2+1:]
            new1 = Route(L1+[route2[node2]]+R1, self.depot)
            new2 = Route(L2+R2, self.depot)
            if idx1 < idx2:
                return VRPSolutionMulti(self.problem, routes[:idx1]+[new1]+routes[idx1+1:idx2]+[new2]+routes[idx2+1:])
            else:
                return VRPSolutionMulti(self.problem, routes[:idx2]+[new2]+routes[idx2+1:idx1]+[new1]+routes[idx1+1:])

    # noinspection DuplicatedCode
    def rand2Opt(self) -> 'VRPSolutionMulti':
        """AB+CD <~> AC+BD <~> AD+BC"""
        # print("rand2Opt")
        routes = self.routes
        idx1: int = random.randrange(len(routes))      # not weighted
        idx2: int = random.randrange(len(routes))      # not weighted
        if idx1 == idx2:
            route: List[Node] = routes[idx1].stops
            edge1 = random.randrange(len(route)+1)
            edge2 = random.randrange(len(route)+1)
            if edge1 > edge2: edge1, edge2 = edge2, edge1
            L, M, R = route[:edge1], route[edge1:edge2], route[edge2:]
            new = Route(L+M[::1]+R, self.depot)
            return VRPSolutionMulti(self.problem, routes[:idx1]+[new]+routes[idx1+1:])
        else:
            if idx1 > idx2: idx1, idx2 = idx2, idx1
            route1: List[Node] = routes[idx1].stops
            route2: List[Node] = routes[idx2].stops
            edge1 = random.randrange(len(route1)+1)
            edge2 = random.randrange(len(route2)+1)
            L1, R1 = route1[:edge1], route1[edge1:]
            L2, R2 = route2[:edge2], route2[edge2:]
            new1: Route
            new2: Route
            if random.randrange(2):
                new1 = Route(L1+R2, self.depot)
                new2 = Route(L2+R1, self.depot)
            else:
                new1 = Route(L1+L2[::-1], self.depot)
                new2 = Route(R1+R2[::-1], self.depot)
            return VRPSolutionMulti(self.problem, routes[:idx1]+[new1]+routes[idx1+1:idx2]+[new2]+routes[idx2+1:])
