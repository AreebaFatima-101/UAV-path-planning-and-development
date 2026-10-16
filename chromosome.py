"""Hybrid real-valued / binary chromosome for a whole fleet.

For every drone d and every waypoint k (K per drone) there are four genes:
    xy[d,k]      real (x, y)  waypoint position on the grid
    speed[d,k]   real         speed on the leg that ARRIVES at waypoint k
    power[d,k]   real 0..1    sensor power on that leg (sets the sensing radius)
    active[d,k]  binary       sensor on / off on that leg
The leg from the last waypoint back to the base is flown at cruise speed, sensor off.
Waypoint ORDER is the visiting order, so the chromosome also encodes the route.
"""
import numpy as np


class Chromosome:
    __slots__ = ("xy", "speed", "power", "active", "metrics", "fitness")

    def __init__(self, xy, speed, power, active):
        self.xy, self.speed, self.power, self.active = xy, speed, power, active
        self.metrics = None
        self.fitness = -np.inf

    @classmethod
    def random(cls, env, rng, sector_bias=False):
        c = env.cfg
        D, K = c.n_drones, c.n_waypoints
        n = c.size - 1
        if sector_bias:   # drone d gets an angular sector around the base -> diverse start
            ang = (np.arange(D)[:, None] + rng.random((D, K))) / D * (np.pi / 2)
            rad = rng.random((D, K)) * n * 1.3
            xy = np.stack([c.base[0] + rad * np.cos(ang), c.base[1] + rad * np.sin(ang)], -1)
        else:
            xy = rng.random((D, K, 2)) * n
        xy = env.repair_points(xy)
        speed = rng.uniform(c.v_min, c.v_max, (D, K))
        power = rng.random((D, K))
        active = rng.random((D, K)) < 0.7
        return cls(xy, speed, power, active)

    def copy(self):
        ch = Chromosome(self.xy.copy(), self.speed.copy(), self.power.copy(), self.active.copy())
        ch.metrics, ch.fitness = self.metrics, self.fitness
        return ch

    def invalidate(self):
        self.metrics, self.fitness = None, -np.inf
