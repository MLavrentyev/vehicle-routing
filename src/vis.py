import matplotlib.pyplot as plt # type: ignore
from problem import VRPSolution


def init():
    plt.ion()


def display(sol: VRPSolution, doPlot: bool = False):
    print(f"sol\t\t: {sol} >> {sol.objectiveValue:.2f}  (d: {sol.distance:.2f}, inf: {sol.infeasibilityPenalty:.2f})")
    if doPlot:
        plot(sol)


def plot(sol: VRPSolution):
    for route in sol.routes:
        stops = [sol.depot] + route.stops + [sol.depot]
        xs = [stop.x for stop in stops]
        ys = [stop.y for stop in stops]
        plt.plot(xs, ys)

        ss = [stop.demand * 10 for stop in stops]
        plt.scatter(xs, ys)
        plt.title(f"Feasible: {sol.isFeasible()} | Score: {sol.objectiveValue:.2f}")

    plt.draw()
    plt.pause(0.0001)
    plt.clf()

