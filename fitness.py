"""Simulation of a chromosome + multi-objective fitness with dynamic penalties.

Evaluation is split in two steps so that penalties can grow during the run
without re-simulating every individual:
    simulate(chromosome)      -> raw metrics (coverage, energy, time, ...)   [expensive]
    score(metrics, progress)  -> scalar fitness to MAXIMISE                  [cheap]
"""
import numpy as np
from .config import EnvConfig, FitnessConfig


class Evaluator:
    def __init__(self, env, fcfg: FitnessConfig):
        self.env, self.f, self.c = env, fcfg, env.cfg
        self.n_evals = 0
        self.S = self.c.samples_per_segment
        self.t = np.linspace(0, 1, self.S, endpoint=False, dtype=np.float32)[None, None, :, None] + 0.5 / self.S

    # ---------------------------------------------------------------- simulation
    def simulate(self, ch, keep_maps=False):
        env, c = self.env, self.c
        self.n_evals += 1
        D, K = ch.speed.shape
        base = np.tile(np.array(c.base, dtype=np.float32), (D, 1, 1))
        pts = np.concatenate([base, ch.xy.astype(np.float32), base], axis=1)   # (D, K+2, 2)
        a, b = pts[:, :-1], pts[:, 1:]                                         # K+1 legs
        seg_len = np.linalg.norm(b - a, axis=-1)                               # (D, K+1)
        speed = np.concatenate([ch.speed, np.full((D, 1), 1.0)], 1)            # return leg: cruise
        power = np.concatenate([ch.power, np.zeros((D, 1))], 1)
        active = np.concatenate([ch.active, np.zeros((D, 1), bool)], 1)

        t_seg = seg_len / speed                                                # time per leg
        e_fly = seg_len * (c.e_move + c.e_speed * speed ** 2)
        e_sen = t_seg * c.e_sensor * power ** 2 * active
        energy = float((e_fly + e_sen).sum())
        dist = seg_len.sum(1)                                                  # per drone
        time_d = t_seg.sum(1)

        # sample points along every leg
        sp = a[:, :, None, :] + (b - a)[:, :, None, :] * self.t                # (D,K+1,S,2)
        in_nfz = env.lookup(env.nfz, sp)                                       # bool (D,K+1,S)
        haz = env.lookup(env.hazard, sp)
        dt = (t_seg / self.S)[:, :, None]
        hazard_exposure = float((haz * dt * speed[:, :, None]).sum())          # hazard x distance
        nfz_frac = float(in_nfz.mean())

        # sensing: radius from power, quality drops with speed
        radius = c.r_min + power * (c.r_max - c.r_min)
        quality = np.clip(1.25 - 0.5 * speed, 0.2, 1.0)
        on = np.broadcast_to(active[:, :, None], in_nfz.shape)
        sx, sy = sp[..., 0][on], sp[..., 1][on]
        r2 = np.broadcast_to((radius ** 2)[:, :, None], in_nfz.shape)[on].astype(np.float32)
        q = np.broadcast_to(quality[:, :, None], in_nfz.shape)[on].astype(np.float32)

        cov_cells = np.zeros(env.cx.shape, np.float32)
        det = np.zeros(len(env.targets), np.float32)
        if sx.size:
            d2 = (sx[:, None] - env.cx[None, :]) ** 2 + (sy[:, None] - env.cy[None, :]) ** 2
            cov_cells = ((d2 <= r2[:, None]) * q[:, None]).max(0)
            dt2 = (sx[:, None] - env.targets[None, :, 0]) ** 2 + (sy[:, None] - env.targets[None, :, 1]) ** 2
            det = ((dt2 <= r2[:, None]) * q[:, None]).max(0)
        coverage = float((cov_cells * env.priority_flat).sum() / env.total_priority)

        m = dict(coverage=coverage, detection=float(det.mean()), energy=energy,
                 time=float(time_d.max()), hazard=hazard_exposure,
                 range_viol=float(np.maximum(dist - c.max_range, 0).sum() / c.max_range),
                 time_viol=max(0.0, float(time_d.max()) - c.max_time) / c.max_time,
                 nfz_frac=nfz_frac, dist=dist.copy())
        if keep_maps:
            m["cov_map"] = cov_cells.reshape(c.size, c.size)
            m["det_vec"] = det
        ch.metrics = m
        return m

    # --------------------------------------------------------------------- score
    def score(self, m, progress=1.0):
        f, c, env = self.f, self.c, self.env
        scale = 1.0 + f.penalty_growth * progress          # dynamic penalty weight
        reward = f.w_cov * m["coverage"] + f.w_det * m["detection"]
        cost = (f.w_energy * m["energy"] / env.e_ref + f.w_time * m["time"] / c.max_time
                + f.w_hazard * m["hazard"] / env.h_ref)
        penalty = scale * (f.lam_range * m["range_viol"] + f.lam_time * m["time_viol"]
                           + f.lam_nfz * m["nfz_frac"] * 10)
        return reward - cost - penalty

    def evaluate(self, ch, progress=1.0):
        if ch.metrics is None:
            self.simulate(ch)
        ch.fitness = self.score(ch.metrics, progress)
        return ch.fitness
