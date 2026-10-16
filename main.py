"""Run one full GA simulation and save all plots into results/.
Usage:  python main.py [--generations 120] [--pop 60] [--seed 0]"""
import argparse, json, os, time
import numpy as np
from uav_ga.config import EnvConfig, FitnessConfig, GAConfig
from uav_ga.environment import Environment
from uav_ga.fitness import Evaluator
from uav_ga.ga import GeneticAlgorithm
from uav_ga.baseline import random_search
from uav_ga import visualize as viz

ap = argparse.ArgumentParser()
ap.add_argument("--generations", type=int, default=120)
ap.add_argument("--pop", type=int, default=60)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="results")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

env = Environment(EnvConfig()); fcfg = FitnessConfig()
gcfg = GAConfig(pop_size=a.pop, generations=a.generations, seed=a.seed)
viz.plot_environment(env, f"{a.out}/fig1_environment.png")

ev = Evaluator(env, fcfg)
snaps = [0, a.generations // 12, a.generations // 4, a.generations // 2]
t0 = time.time()
ga = GeneticAlgorithm(env, ev, gcfg, snapshot_gens=snaps)
best = ga.run()
print(f"GA finished in {time.time()-t0:.1f}s, stop reason: {ga.stop_reason}, evaluations: {ev.n_evals}")

ev2 = Evaluator(env, fcfg)
rs_best, rs_curve = random_search(env, ev2, ev.n_evals, seed=a.seed)
rs_fit = ev2.score(rs_best.metrics)

viz.plot_convergence(ga.history, f"{a.out}/fig2_convergence.png")
viz.plot_generations(env, ga.snapshots, f"{a.out}/fig3_paths_generations.png", "paths")
viz.plot_generations(env, ga.snapshots, f"{a.out}/fig4_heatmaps_generations.png", "heat")
viz.plot_final(env, ga.snapshots[max(ga.snapshots)], f"{a.out}/fig5_final_solution.png")

m = best.metrics
summary = dict(ga_best_fitness=float(best.fitness), random_best_fitness=float(rs_fit),
               ga_metrics={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in m.items() if k not in ("cov_map", "det_vec")},
               random_metrics={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in rs_best.metrics.items() if k not in ("cov_map", "det_vec")},
               evaluations=ev.n_evals, stop_reason=ga.stop_reason)
json.dump(summary, open(f"{a.out}/main_run_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
