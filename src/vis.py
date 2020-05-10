
import matplotlib.pyplot as plt
from datetime import datetime

from problem import VRPSolution

####

G = {}

# time can be number of iterations or None for datetime.now()
def init(time = None):
    # plt.style.use('fivethirtyeight')
    plt.ion()
    G['start'] = time if time is not None else datetime.now().timestamp()
    G['ts'] = []
    G['os'] = []

def plot(sol: VRPSolution, time = None):
    if G['os'] and sol.objectiveValue >= G['os'][-1]: return
    time = time if time is not None else datetime.now().timestamp()
    dt = time - G['start']
    G['ts'].append(dt)
    G['os'].append(sol.objectiveValue)
    if len(G['ts']) >= 2:
        xs = [n.x for n in sol.problem.nodes]
        ys = [n.y for n in sol.problem.nodes]
        minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
        mint, maxt, mino, maxo = min(G['ts']), max(G['ts']), min(G['os']), max(G['os'])
        rx = (maxx-minx)/(maxt-mint)
        ry = (maxy-miny)/(maxo-mino)
        ts = [minx+(t-mint)*rx for t in G['ts']]
        os = [miny+(o-mino)*ry for o in G['os']]
        plt.plot(ts, os, color="black", linewidth=5)

    for route in sol.routes:
        stops = [sol.depot]+route.stops+[sol.depot]
        xs = [stop.x for stop in stops]
        ys = [stop.y for stop in stops]
        plt.plot(xs, ys)
        ss = [stop.demand*10 for stop in stops]
        plt.scatter(xs, ys, ss)
        plt.title(f"Feasible: {'??'} | Score: {'??'}")
    plt.draw()
    plt.pause(0.0001)   # whytf is this necessary?
    plt.clf()
