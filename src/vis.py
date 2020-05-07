import matplotlib.pyplot as plt
from problem import VRPSolution


def init():
    plt.ion()

def plot(sol: VRPSolution):
    for route in sol.routes:
        stops = [sol.depot] + route.stops + [sol.depot]
        xs = [stop.x for stop in stops]
        ys = [stop.y for stop in stops]
        plt.plot(xs, ys)

        ss = [stop.demand * 10 for stop in stops]
        plt.scatter(xs, ys, ss)
        plt.title(f"Feasible: {'??'} | Score: {'??'}")

    plt.draw()
    plt.pause(0.0001)
    plt.clf()

