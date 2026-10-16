"""Hyper-parameter sensitivity + GA-vs-random-search comparison over several seeds.
Usage:  python experiments.py [--seeds 5] [--workers 2]
Results go to results/ (figures + experiments.json)."""
import argparse, json, os, pickle
from multiprocessing import Pool
import numpy as np
from uav_ga.config import EnvConfig, FitnessConfig, GAConfig
from uav_ga.environment import Environment
from uav_ga.fitness import Evaluator
from uav_ga.ga import GeneticAlgorithm
from uav_ga.baseline import random_search
from uav_ga import visualize as viz

GENS = 100
GRIDS = {
    "Population size": ("pop_size", [20, 40, 60, 100]),
    "Mutation rate": ("mutation_rate", [0.02, 0.06, 0.12, 0.25]),
    "Selection pressure (tournament k)": ("tournament_k", [1, 2, 4, 8]),
}


def run_one(args):
    param, value, seed = args
    env = Environment(EnvConfig()); ev = Evaluator(env, FitnessConfig())
    kw = dict(pop_size=60, generations=GENS, seed=seed); kw[param] = value
    ga = GeneticAlgorithm(env, ev, GAConfig(**kw)); best = ga.run()
    return param, value, seed, float(best.fitness), ev.n_evals, ga.history["diversity"][-1]


def run_compare(seed):
    env = Environment(EnvConfig())
    ev = Evaluator(env, FitnessConfig())
    ga = GeneticAlgorithm(env, ev, GAConfig(pop_size=60, generations=120, seed=seed)); best = ga.run()
    budget = ev.n_evals
    ev.simulate(best, keep_maps=True)   # attach coverage map for the overlap analysis
    ev2 = Evaluator(env, FitnessConfig())
    rs, curve = random_search(env, ev2, budget, seed=seed, log_every=budget // 50)
    ga_curve = [max(ga.history["best"][:max(1, int(i / 50 * len(ga.history["best"])))]) for i in range(1, 51)]
    return dict(seed=seed, ga_fit=float(best.fitness), rs_fit=float(ev2.score(rs.metrics)), ga_curve=ga_curve,
                rs_curve=curve[:50], ga=best, rs=rs, budget=budget)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--out", default="results")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    jobs = [(p, v, s) for _, (p, vals) in GRIDS.items() for v in vals for s in range(a.seeds)]
    with Pool(a.workers) as pool:
        out = pool.map(run_one, jobs)
        cmp = pool.map(run_compare, range(a.seeds))
    pickle.dump((out, cmp), open(f'{a.out}/raw_experiments.pkl', 'wb'))
    res = {}
    for title, (p, vals) in GRIDS.items():
        res[title] = {str(v): [r[3] for r in out if r[0] == p and r[1] == v] for v in vals}
    viz.plot_sensitivity(res, f"{a.out}/fig7_sensitivity.png")
    viz.plot_ga_vs_random([c["ga_curve"] for c in cmp], [c["rs_curve"] for c in cmp], f"{a.out}/fig6_ga_vs_random.png")
    env = Environment(EnvConfig())
    viz.plot_uniqueness(env, [c["ga"] for c in cmp], [c["rs"] for c in cmp], f"{a.out}/fig8_uniqueness.png")

    def overlap(sols):          # mean Jaccard overlap of covered cells between different seeds
        masks = [s.metrics["cov_map"] > 0.2 for s in sols]; v = []
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                v.append((masks[i] & masks[j]).sum() / (masks[i] | masks[j]).sum())
        return float(np.mean(v))

    def route_dist(sols):       # mean distance between waypoints of different seeds' routes
        v = []
        for i in range(len(sols)):
            for j in range(i + 1, len(sols)):
                v.append(np.linalg.norm(sols[i].xy - sols[j].xy, axis=-1).mean())
        return float(np.mean(v))

    ga_s, rs_s = [c["ga"] for c in cmp], [c["rs"] for c in cmp]
    keys = ["coverage", "detection", "energy", "time", "hazard"]
    summary = dict(sensitivity={t: {k: dict(mean=float(np.mean(v)), std=float(np.std(v)), runs=v) for k, v in d.items()} for t, d in res.items()},
                   compare=dict(ga_fit=[c["ga_fit"] for c in cmp], rs_fit=[c["rs_fit"] for c in cmp], budget=cmp[0]["budget"],
                                ga_metrics={k: float(np.mean([s.metrics[k] for s in ga_s])) for k in keys},
                                rs_metrics={k: float(np.mean([s.metrics[k] for s in rs_s])) for k in keys},
                                ga_overlap=overlap(ga_s), rs_overlap=overlap(rs_s),
                                ga_route_dist=route_dist(ga_s), rs_route_dist=route_dist(rs_s),
                                ga_nfz_violations=[s.metrics["nfz_frac"] for s in ga_s],
                                rs_nfz_violations=[s.metrics["nfz_frac"] for s in rs_s]))
    json.dump(summary, open(f"{a.out}/experiments.json", "w"), indent=2)
    print(json.dumps(summary["compare"], indent=2))
    for t, d in summary["sensitivity"].items():
        print(t, {k: (round(v["mean"], 3), round(v["std"], 3)) for k, v in d.items()})
