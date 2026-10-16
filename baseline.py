"""Random-search baseline with the same number of fitness evaluations as the GA."""
import numpy as np
from .chromosome import Chromosome


def random_search(env, evaluator, n_evals, seed=0, log_every=60):
    rng = np.random.default_rng(seed)
    best, best_f, curve = None, -np.inf, []
    for i in range(n_evals):
        ch = Chromosome.random(env, rng, sector_bias=(i % 2 == 0))
        f = evaluator.evaluate(ch, 1.0)
        if f > best_f:
            best, best_f = ch, f
        if (i + 1) % log_every == 0:
            curve.append(best_f)
    best = best.copy(); evaluator.simulate(best, keep_maps=True)
    return best, curve
