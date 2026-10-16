"""Selection, crossover, mutation: all aware of spatial coordinates."""
import numpy as np
from .chromosome import Chromosome


# ------------------------------------------------------------------ selection
def tournament_select(pop, rng, k):
    idx = rng.integers(0, len(pop), k)
    return pop[max(idx, key=lambda i: pop[i].fitness)]


def rank_select(pop, rng, pressure=1.7):
    """Linear ranking: pressure in (1, 2]; higher = stronger selection."""
    n = len(pop)
    order = np.argsort([p.fitness for p in pop])           # worst ... best
    ranks = np.empty(n); ranks[order] = np.arange(n)
    prob = (2 - pressure) / n + 2 * ranks * (pressure - 1) / (n * (n - 1))
    return pop[rng.choice(n, p=prob / prob.sum())]


# ------------------------------------------------------------------ crossover
def crossover(p1, p2, rng, env):
    """Per drone pick one of three operators:
       1. segment swap: exchange a block of consecutive waypoints (keeps good route pieces)
       2. geometric (blend) crossover: child point lies between / slightly beyond parents
       3. route swap: take the whole route of one parent for that drone
    """
    c1, c2 = p1.copy(), p2.copy()
    c1.invalidate(); c2.invalidate()
    D, K = p1.speed.shape
    for d in range(D):
        op = rng.integers(0, 3)
        if op == 0:
            i, j = sorted(rng.integers(0, K + 1, 2))
            if i == j:
                j = min(K, i + 1)
            for name in ("xy", "speed", "power", "active"):
                a, b = getattr(c1, name), getattr(c2, name)
                tmp = a[d, i:j].copy(); a[d, i:j] = b[d, i:j]; b[d, i:j] = tmp
        elif op == 1:
            alpha = rng.uniform(-0.2, 1.2, (K, 1))          # BLX-style: may overshoot a little
            x1, x2 = p1.xy[d], p2.xy[d]
            c1.xy[d] = alpha * x1 + (1 - alpha) * x2
            c2.xy[d] = (1 - alpha) * x1 + alpha * x2
            beta = rng.random(K)
            for name in ("speed", "power"):
                a, b = getattr(p1, name)[d], getattr(p2, name)[d]
                getattr(c1, name)[d] = beta * a + (1 - beta) * b
                getattr(c2, name)[d] = (1 - beta) * a + beta * b
            mask = rng.random(K) < 0.5                         # uniform crossover on bits
            c1.active[d] = np.where(mask, p1.active[d], p2.active[d])
            c2.active[d] = np.where(mask, p2.active[d], p1.active[d])
        else:
            for name in ("xy", "speed", "power", "active"):
                a, b = getattr(c1, name), getattr(c2, name)
                tmp = a[d].copy(); a[d] = b[d]; b[d] = tmp
    for c in (c1, c2):
        c.xy = env.repair_points(c.xy)
    return c1, c2


# ------------------------------------------------------------------- mutation
def mutate(ch, rng, env, rate, sigma):
    """Spatial mutation set (each applied gene-wise with probability `rate`):
       gaussian shift of a waypoint (step = sigma cells), guided relocation to a
       high-priority cell, swap of two waypoints in the route, speed/power jitter,
       sensor bit flip."""
    c = env.cfg
    D, K = ch.speed.shape
    changed = False
    shift = rng.random((D, K)) < rate
    if shift.any():
        ch.xy[shift] += rng.normal(0, sigma, (shift.sum(), 2)); changed = True
    jump = rng.random((D, K)) < rate * 0.25                    # guided / random relocation
    if jump.any():
        for d, k in zip(*np.nonzero(jump)):
            if rng.random() < 0.6:                              # sample a cell by priority
                p = (env.priority_flat + 1e-3); p = p / p.sum()
                cell = rng.choice(p.size, p=p)
                ch.xy[d, k] = (cell % c.size, cell // c.size)
            else:
                ch.xy[d, k] = rng.random(2) * (c.size - 1)
        changed = True
    for d in range(D):                                          # route-order swap
        if rng.random() < rate * 1.5:
            i, j = rng.choice(K, 2, replace=False)
            for name in ("xy", "speed", "power", "active"):
                arr = getattr(ch, name)[d]
                arr[[i, j]] = arr[[j, i]]
            changed = True
    sp = rng.random((D, K)) < rate
    if sp.any():
        ch.speed[sp] = np.clip(ch.speed[sp] + rng.normal(0, 0.25, sp.sum()), c.v_min, c.v_max); changed = True
    pw = rng.random((D, K)) < rate
    if pw.any():
        ch.power[pw] = np.clip(ch.power[pw] + rng.normal(0, 0.2, pw.sum()), 0, 1); changed = True
    fl = rng.random((D, K)) < rate * 0.5
    if fl.any():
        ch.active[fl] = ~ch.active[fl]; changed = True
    if changed:
        ch.xy = env.repair_points(ch.xy)
        ch.invalidate()
    return ch
