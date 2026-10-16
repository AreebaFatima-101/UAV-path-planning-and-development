"""The hazard grid: priority zones, hazard pockets, no-fly zones and hidden targets."""
import numpy as np
from scipy.ndimage import distance_transform_edt
from .config import EnvConfig


class Environment:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        n = cfg.size
        rng = np.random.default_rng(cfg.seed)
        yy, xx = np.mgrid[0:n, 0:n]
        self.cx = xx.ravel().astype(np.float32)          # cell centres, flattened
        self.cy = yy.ravel().astype(np.float32)

        def blob(cx, cy, s):
            return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s ** 2))

        # priority map: 0 = ordinary cell, up to 1 = critical disaster zone
        self.priority = np.clip(0.05 + 1.0 * blob(36, 38, 6) + 0.9 * blob(12, 40, 5)
                                + 0.8 * blob(40, 10, 5), 0, 1)
        # hazard map: wind / radiation pockets (0 safe .. 1 severe)
        self.hazard = np.clip(0.9 * blob(24, 26, 6) + 0.8 * blob(26, 8, 4)
                              + 0.7 * blob(8, 22, 4), 0, 1)
        # no-fly zones (True = blocked): axis-aligned rectangles, [x0, x1, y0, y1]
        self.nfz = np.zeros((n, n), dtype=bool)
        self.nfz_rects = [(18, 24, 36, 44), (28, 33, 16, 22), (5, 10, 8, 14)]
        for x0, x1, y0, y1 in self.nfz_rects:
            self.nfz[y0:y1, x0:x1] = True
        # nearest free cell for every cell (used to repair waypoints in a no-fly zone)
        _, idx = distance_transform_edt(self.nfz, return_indices=True)
        self._nearest_free = idx                          # shape (2, n, n) -> (row, col)
        # hidden targets, more likely where priority is high
        p = (self.priority ** 2).copy()
        p[self.nfz] = 0
        p = (p / p.sum()).ravel()
        cells = rng.choice(n * n, size=cfg.n_targets, replace=False, p=p)
        self.targets = np.stack([cells % n, cells // n], axis=1).astype(np.float32)
        self.priority_flat = self.priority.ravel().astype(np.float32)
        self.total_priority = float(self.priority_flat.sum())
        # normalisation references (so every objective lives roughly in 0..1)
        c = cfg
        self.e_ref = c.n_drones * c.max_range * (c.e_move + c.e_speed * 1.0)
        self.h_ref = c.n_drones * c.max_range * 0.25     # exposure if 1/4 of range is in full hazard

    def in_bounds_clip(self, xy):
        return np.clip(xy, 0, self.cfg.size - 1)

    def repair_points(self, xy):
        """Move points that fall inside a no-fly zone to the nearest free cell."""
        xy = self.in_bounds_clip(xy)
        col = np.rint(xy[..., 0]).astype(int)
        row = np.rint(xy[..., 1]).astype(int)
        bad = self.nfz[row, col]
        if bad.any():
            xy = xy.copy()
            r2 = self._nearest_free[0][row[bad], col[bad]]
            c2 = self._nearest_free[1][row[bad], col[bad]]
            xy[bad] = np.stack([c2, r2], axis=-1)
        return xy

    def lookup(self, grid, pts):
        col = np.rint(np.clip(pts[..., 0], 0, self.cfg.size - 1)).astype(int)
        row = np.rint(np.clip(pts[..., 1], 0, self.cfg.size - 1)).astype(int)
        return grid[row, col]
