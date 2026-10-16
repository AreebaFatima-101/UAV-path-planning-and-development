"""All tunable settings in one place (dataclasses, so experiments can override them)."""
from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    size: int = 50                 # grid is size x size cells
    n_drones: int = 3
    n_waypoints: int = 8           # waypoints per drone (K)
    base: tuple = (1.0, 1.0)       # launch / landing point
    n_targets: int = 30            # hidden targets inside disaster zones
    max_range: float = 170.0       # hard limit: flight distance per drone (cells)
    max_time: float = 140.0        # hard limit: mission time (time units)
    v_min: float = 0.6             # speed range (cells per time unit)
    v_max: float = 2.0
    r_min: float = 2.0             # sensor radius range (cells)
    r_max: float = 6.0
    e_move: float = 1.0            # energy per cell flown at v = 0
    e_speed: float = 0.5           # extra energy per cell, grows with v^2
    e_sensor: float = 1.2          # sensor energy per time unit at full power
    samples_per_segment: int = 12  # points checked along every straight leg
    seed: int = 7                  # seed of the (fixed) disaster map


@dataclass
class FitnessConfig:
    w_cov: float = 0.40            # priority-weighted coverage      (maximise)
    w_det: float = 0.25            # target detection rate           (maximise)
    w_energy: float = 0.12         # battery use                     (minimise)
    w_time: float = 0.08           # mission time                    (minimise)
    w_hazard: float = 0.15         # exposure to hazard pockets      (minimise)
    lam_range: float = 2.0         # penalty weight: range limit
    lam_time: float = 1.0          # penalty weight: time limit
    lam_nfz: float = 3.0           # penalty weight: no-fly zone crossing
    penalty_growth: float = 3.0    # penalty grows (1 -> 1+growth) over the run


@dataclass
class GAConfig:
    pop_size: int = 60
    generations: int = 120
    tournament_k: int = 3          # selection pressure
    selection: str = "tournament"  # "tournament" or "rank"
    crossover_rate: float = 0.85
    mutation_rate: float = 0.12    # per-gene mutation probability
    elite: int = 2
    sigma0: float = 6.0            # initial coordinate mutation step (cells)
    sigma_min: float = 1.0
    diversity_floor: float = 6.0   # mean waypoint distance below this => boost mutation
    immigrant_frac: float = 0.08   # random newcomers injected when stagnating
    patience: int = 60             # stop early after this many generations w/o gain
    seed: int = 0
