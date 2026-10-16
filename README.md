# UAV Strategy Engine: Genetic Algorithm for Multi-Drone Surveillance

Bahria University, Lahore Campus | BSCS 5 | Artificial Intelligence | Assignment 2 (CLO4)

A modular, object-oriented Genetic Algorithm that plans waypoints, speeds and sensor
power for a fleet of 3 UAVs over a 50 x 50 hazard grid. It balances four competing goals:
priority-zone coverage, target detection, energy/time cost and hazard exposure,
under hard constraints (range limit, time limit, no-fly zones).

## Project layout

```
uav-ga-strategy-engine/
├── main.py              # one full run: GA + random baseline + figures 1-5
├── experiments.py       # sensitivity study + multi-seed GA vs random (figures 6-8)
├── requirements.txt
├── uav_ga/
│   ├── config.py        # EnvConfig, FitnessConfig, GAConfig (every tunable number)
│   ├── environment.py   # grid: priority map, hazard pockets, no-fly zones, targets, repair
│   ├── chromosome.py    # hybrid real/binary chromosome
│   ├── fitness.py       # simulation + multi-objective fitness + dynamic penalties
│   ├── operators.py     # selection, crossover, mutation
│   ├── ga.py            # GA loop: init, elitism, adaptive mutation, immigrants, termination
│   ├── baseline.py      # random search with equal evaluation budget
│   └── visualize.py     # all plots
├── results/             # generated figures + json summaries
└── report/              # technical report
```

## How to run

```bash
python -m venv venv && source venv/bin/activate      # optional
pip install -r requirements.txt
python main.py                      # ~15 s, writes results/fig1..fig5 + main_run_summary.json
python main.py --generations 200 --pop 100 --seed 3   # custom run
python experiments.py --seeds 5     # ~10 min, writes results/fig6..fig8 + experiments.json
```

Python 3.9+ is needed. Runs are reproducible: the map is fixed by `EnvConfig.seed`, the GA by `GAConfig.seed`.

## Design summary

**Chromosome (hybrid real + binary).** For each of 3 drones and 8 waypoints: `(x, y)` real,
`speed` real, `power` real 0..1, `sensor_on` bit. Waypoint order = visiting order.
Total 3 x 8 x 5 = 120 genes. Every drone starts and ends at the base.

**Fitness (maximise).**
`F = 0.40*Coverage + 0.25*Detection - 0.12*Energy - 0.08*Time - 0.15*Hazard - Penalties`
Each term is normalised to roughly 0..1. Weights live in `FitnessConfig`.

**Operators.**
- Selection: tournament (size k = selection pressure) or linear rank.
- Crossover (chosen per drone): segment swap, geometric/blend (BLX), whole-route swap.
- Mutation: Gaussian waypoint shift, guided relocation to high-priority cells, waypoint-order swap,
  speed/power jitter, sensor bit flip. Step size shrinks over time but is boosted when diversity collapses.
- Diversity: elitism (2), random immigrants when stuck, diversity-triggered mutation boost.

**Constraints.** Waypoints inside no-fly zones are repaired to the nearest free cell.
Flights crossing a no-fly zone, exceeding range or mission time are penalised, and
the penalty weight grows during the run (dynamic penalty).

**Termination.** Max generations, or no improvement for `patience` generations.

## Outputs

| File | Content |
|------|---------|
| fig1_environment | Grid with priority zones, hazards, no-fly zones, targets |
| fig2_convergence | Best / average / worst fitness + population diversity |
| fig3_paths_generations | Best flight paths at generations 0, 10, 30, 60, 120 |
| fig4_heatmaps_generations | Coverage heatmaps at the same generations |
| fig5_final_solution | Final routes + coverage heatmap |
| fig6_ga_vs_random | GA vs random search (equal evaluations, several seeds) |
| fig7_sensitivity | Population size, mutation rate, selection pressure |
| fig8_uniqueness | Best routes from every seed, GA vs random |
