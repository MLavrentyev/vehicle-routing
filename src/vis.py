
import matplotlib.pyplot as plt

from problem import VRPSolution

####

def plot(sol: VRPSolution):
    for route in sol.routes:
        stops = [sol.depot]+route.stops+[sol.depot]
        xs = [stop.x for stop in stops]
        ys = [stop.y for stop in stops]
        plt.plot(xs, ys)
    plt.show()
