"""The Genetic Algorithm engine."""
import numpy as np
from .chromosome import Chromosome
from .config import GAConfig
from .operators import tournament_select, rank_select, crossover, mutate


def population_diversity(pop, rng, n_pairs=60):
    """Mean distance between waypoint sets of random individual pairs (in cells)."""
    ds = []
    for _ in range(n_pairs):
        a, b = rng.choice(len(pop), 2, replace=False)
        ds.append(np.linalg.norm(pop[a].xy - pop[b].xy, axis=-1).mean())
    return float(np.mean(ds))


class GeneticAlgorithm:
    def __init__(self, env, evaluator, cfg: GAConfig, snapshot_gens=()):
        self.env, self.ev, self.cfg = env, evaluator, cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.snapshot_gens = set(snapshot_gens)
        self.history = dict(best=[], mean=[], worst=[], diversity=[], sigma=[])
        self.snapshots = {}            # generation -> best chromosome copy (with maps)
        self.best = None
        self.stop_reason = ""

    def _init_population(self):
        n = self.cfg.pop_size
        return [Chromosome.random(self.env, self.rng, sector_bias=(i % 2 == 0)) for i in range(n)]

    def _select(self, pop):
        if self.cfg.selection == "rank":
            return rank_select(pop, self.rng)
        return tournament_select(pop, self.rng, self.cfg.tournament_k)

    def run(self):
        cfg, rng = self.cfg, self.rng
        pop = self._init_population()
        sigma, stagnant, best_val = cfg.sigma0, 0, -np.inf
        for g in range(cfg.generations + 1):
            progress = g / cfg.generations
            for ch in pop:
                self.ev.evaluate(ch, progress)       # re-score with the current penalty weight
            pop.sort(key=lambda c: c.fitness, reverse=True)
            fits = np.array([c.fitness for c in pop])
            div = population_diversity(pop, rng)
            self.best = pop[0].copy()
            h = self.history
            h["best"].append(fits[0]); h["mean"].append(fits.mean()); h["worst"].append(fits.min())
            h["diversity"].append(div); h["sigma"].append(sigma)
            if g in self.snapshot_gens or g == cfg.generations:
                snap = pop[0].copy(); self.ev.simulate(snap, keep_maps=True)
                self.snapshots[g] = snap
            # termination criteria
            if fits[0] > best_val + 1e-6:
                best_val, stagnant = fits[0], 0
            else:
                stagnant += 1
            if g == cfg.generations:
                self.stop_reason = "max generations"; break
            if stagnant >= cfg.patience:
                self.stop_reason = f"stagnation ({cfg.patience} generations)"
                snap = pop[0].copy(); self.ev.simulate(snap, keep_maps=True); self.snapshots[g] = snap
                break
            # adaptive step size: shrink over time, but boost when diversity collapses
            sigma = max(cfg.sigma_min, cfg.sigma0 * (1 - 0.85 * progress))
            rate = cfg.mutation_rate
            if div < cfg.diversity_floor:
                sigma *= 1.8; rate = min(0.5, rate * 2)
            # next generation: elitism + offspring
            nxt = [c.copy() for c in pop[:cfg.elite]]
            while len(nxt) < cfg.pop_size:
                p1, p2 = self._select(pop), self._select(pop)
                if rng.random() < cfg.crossover_rate:
                    c1, c2 = crossover(p1, p2, rng, self.env)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                nxt.append(mutate(c1, rng, self.env, rate, sigma))
                if len(nxt) < cfg.pop_size:
                    nxt.append(mutate(c2, rng, self.env, rate, sigma))
            # random immigrants when the search has stalled (keeps diversity alive)
            if stagnant > 10:
                k = max(1, int(cfg.immigrant_frac * cfg.pop_size))
                for i in range(k):
                    nxt[-1 - i] = Chromosome.random(self.env, rng, sector_bias=bool(i % 2))
            pop = nxt
        self.final_generation = g
        return self.best
