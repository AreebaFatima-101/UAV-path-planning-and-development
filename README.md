# UAV-path-planning-and-development
# UAV Strategy Engine: Genetic Algorithm for Drone Surveillance

**Course:** Artificial Intelligence, BSCS 5, Bahria University Lahore Campus
**Assignment:** 2 (CLO4)
**Group members:** _Name (Roll No), Name (Roll No), Name (Roll No)_

## About the project

This project plans the flights of 3 drones (UAVs) over a 50 x 50 grid. The grid has:

- 3 disaster zones that matter most (high priority),
- 3 hazard areas (like strong wind or radiation),
- 3 no-fly zones,
- 30 hidden targets to find.

Every drone starts and ends at the same base. The goals pull against each other: seeing more of the map costs battery and time and often means flying through hazards. So there is no single best answer. We use a Genetic Algorithm (GA) to find a good compromise, and we compare it with plain random search.

## What the GA decides

Each plan (chromosome) stores, for every drone and each of its 8 waypoints:

- x and y position,
- speed,
- sensor power,
- sensor on/off.

That is 3 x 8 x 5 = 120 genes. The order of the waypoints is the order the drone flies them.

**Fitness score (higher is better):**

```
F = 0.40 Coverage + 0.25 Detection - 0.12 Energy - 0.08 Time - 0.15 Hazard - Penalties
```

**Hard rules:** waypoints inside a no-fly zone are moved to the nearest free cell (repair). Crossing a no-fly zone, flying too far (170 cells) or taking too long (140 time units) is penalised, and the penalty gets stronger during the run.

**Operators:**

- Selection: tournament (or rank)
- Crossover: segment swap, blend (geometric), whole-route swap
- Mutation: position shift, move to a high-priority cell, swap two waypoints, speed/power change, sensor flip
- Diversity: elitism, random newcomers when stuck, bigger mutation when the population gets too similar
- Stop: maximum generations, or 60 generations without improvement

## Project structure

```
uav-ga-strategy-engine/
├── main.py              # one full run: GA + random search + figures 1 to 5
├── experiments.py       # parameter tests + GA vs random search (figures 6 to 8)
├── tuned_check.py       # GA with the best settings found (mutation 0.03, k = 5)
├── requirements.txt
├── uav_ga/
│   ├── config.py        # all settings in one place
│   ├── environment.py   # the grid: priorities, hazards, no-fly zones, targets
│   ├── chromosome.py    # how a plan is stored
│   ├── fitness.py       # simulates a plan and scores it
│   ├── operators.py     # selection, crossover, mutation
│   ├── ga.py            # the main GA loop
│   ├── baseline.py      # random search for comparison
│   └── visualize.py     # all the plots
├── results/             # figures and result files
└── report/              # technical report
```

## How to run

You need Python 3.9 or newer.

```bash
# 1. go into the project folder
cd uav-ga-strategy-engine

# 2. (optional) make a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac / Linux

# 3. install the libraries
pip install -r requirements.txt

# 4. run the main simulation (about 15 to 60 seconds)
python main.py
```

Other commands:

```bash
python main.py --generations 200 --pop 100 --seed 3   # your own settings
python experiments.py --seeds 5                       # parameter tests, about 10 to 25 minutes
python tuned_check.py                                 # GA with the tuned settings
```

Results are repeatable: the map is fixed and each run uses a seed.

## Output figures

All figures are saved in the `results/` folder.

| File | What it shows |
|------|---------------|
| fig1_environment.png | The grid with priority zones, hazards, no-fly zones and targets |
| fig2_convergence.png | Best, average and worst fitness over generations, plus diversity |
| fig3_paths_generations.png | Best flight paths at generations 0, 10, 30, 60 and 120 |
| fig4_heatmaps_generations.png | Coverage heatmaps at the same generations |
| fig5_final_solution.png | The final routes and coverage heatmap |
| fig6_ga_vs_random.png | GA vs random search with the same number of evaluations |
| fig7_sensitivity.png | Effect of population size, mutation rate and selection pressure |
| fig8_uniqueness.png | Best routes from every seed, GA vs random |

## Main results

Average of 5 seeds, same number of evaluations (about 7,000) for both:

| Measure | GA | Random search |
|---------|----|---------------|
| Final fitness | 0.166 | -0.698 |
| Priority coverage | 0.60 | 0.32 |
| Target detection | 0.70 | 0.32 |
| Hazard exposure | 45.1 | 86.4 |
| Runs crossing a no-fly zone | 0 of 5 | 4 of 5 |

With mutation rate 0.03 and tournament size 5, the GA reached a fitness of 0.306 (very steady across seeds). Our first default (mutation 0.12) was too high.

## Limitations

- The fitness weights were chosen by reasoning, not tested.
- Everything was tested on one fixed map and 5 seeds per setting.
- The drone model is simple (straight legs, simple energy and sensor rules), so this shows the method works, not that the plans are ready to fly.

## Report

The full technical report is in the `report/` folder.
