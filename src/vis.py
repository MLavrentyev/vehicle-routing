import matplotlib.pyplot as plt # type: ignore
from problem import VRPSolution


def init():
    plt.ion()


def display(solution: VRPSolution):
    print(f"sol\t\t: {solution} >> {solution.objectiveValue:.2f}")
    vis.plot(solution)


def plot(sol: VRPSolution):
    for route in sol.routes:
        stops = [sol.depot] + route.stops + [sol.depot]
        xs = [stop.x for stop in stops]
        ys = [stop.y for stop in stops]
        plt.plot(xs, ys)

        ss = [stop.demand * 10 for stop in stops]
        plt.scatter(xs, ys, ss)
        plt.title(f"Feasible: {sol.isFeasible()} | Score: {sol.objectiveValue}")

    plt.draw()
    plt.pause(0.0001)
    plt.clf()

