"""All plotting helpers (matplotlib, saved as PNG)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]


def _draw_env(ax, env, show_priority=True):
    n = env.cfg.size
    if show_priority:
        ax.imshow(env.priority, origin="lower", cmap="Greens", alpha=0.55, vmin=0, vmax=1)
    ax.imshow(np.ma.masked_less(env.hazard, 0.25), origin="lower", cmap="Reds", alpha=0.55, vmin=0, vmax=1)
    for x0, x1, y0, y1 in env.nfz_rects:
        ax.add_patch(Rectangle((x0 - .5, y0 - .5), x1 - x0, y1 - y0, fc="black", alpha=0.75, ec="k"))
    ax.scatter(env.targets[:, 0], env.targets[:, 1], marker="*", s=28, c="gold", ec="k", lw=.4, zorder=5)
    ax.plot(*env.cfg.base, "ks", ms=9, zorder=6)
    ax.set_xlim(-.5, n - .5); ax.set_ylim(-.5, n - .5); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def plot_environment(env, path):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    _draw_env(ax, env)
    ax.set_title("Hazard grid 50x50\ngreen = disaster priority, red = hazard pockets,\nblack = no-fly zones, star = target, square = base", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _draw_paths(ax, env, ch):
    c = env.cfg
    for d in range(ch.xy.shape[0]):
        pts = np.vstack([c.base, ch.xy[d], c.base])
        ax.plot(pts[:, 0], pts[:, 1], "-", color=COLORS[d % 5], lw=1.8, zorder=4, label=f"UAV {d+1}")
        ax.scatter(ch.xy[d][:, 0], ch.xy[d][:, 1], s=14, color=COLORS[d % 5], ec="k", lw=.3, zorder=5)


def plot_convergence(history, path, title="GA convergence"):
    g = np.arange(len(history["best"]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.plot(g, history["best"], lw=2, label="Best fitness")
    a1.plot(g, history["mean"], lw=1.5, label="Average fitness")
    a1.fill_between(g, history["worst"], history["best"], alpha=0.12)
    a1.set_ylim(min(history["mean"][2:] + [0]) - .3 if len(g) > 3 else None, max(history["best"]) + .1)
    a1.set_xlabel("Generation"); a1.set_ylabel("Fitness"); a1.set_title(title); a1.legend(); a1.grid(alpha=.3)
    a2.plot(g, history["diversity"], color="purple"); a2.set_xlabel("Generation")
    a2.set_ylabel("Population diversity (cells)"); a2.set_title("Diversity (mean waypoint distance)"); a2.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_generations(env, snapshots, path, kind="paths"):
    gens = sorted(snapshots)
    fig, axes = plt.subplots(1, len(gens), figsize=(3.3 * len(gens), 3.8))
    for ax, g in zip(np.atleast_1d(axes), gens):
        ch = snapshots[g]
        if kind == "paths":
            _draw_env(ax, env); _draw_paths(ax, env, ch)
        else:
            im = ax.imshow(ch.metrics["cov_map"], origin="lower", cmap="inferno", vmin=0, vmax=1)
            for x0, x1, y0, y1 in env.nfz_rects:
                ax.add_patch(Rectangle((x0 - .5, y0 - .5), x1 - x0, y1 - y0, fc="none", ec="cyan", lw=1))
            ax.set_xticks([]); ax.set_yticks([])
        m = ch.metrics
        ax.set_title(f"Gen {g}\ncov {m['coverage']:.2f}  det {m['detection']:.2f}", fontsize=9)
    if kind != "paths":
        fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01, label="detection quality")
    fig.suptitle("Evolved UAV flight paths across generations" if kind == "paths"
                 else "Coverage heatmap of the best solution across generations (cyan = no-fly)", y=1.02)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_final(env, ch, path, name="Final evolved solution"):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.6))
    _draw_env(a1, env); _draw_paths(a1, env, ch); a1.legend(loc="upper right", fontsize=7)
    a1.set_title(name)
    im = a2.imshow(ch.metrics["cov_map"], origin="lower", cmap="inferno", vmin=0, vmax=1)
    a2.contour(env.priority, levels=[0.4], colors="cyan", linewidths=1)
    for x0, x1, y0, y1 in env.nfz_rects:
        a2.add_patch(Rectangle((x0 - .5, y0 - .5), x1 - x0, y1 - y0, fc="none", ec="white", ls="--"))
    a2.set_xticks([]); a2.set_yticks([]); a2.set_title("Coverage heatmap (cyan = high-priority zones)")
    fig.colorbar(im, ax=a2, fraction=0.046, label="detection quality")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_ga_vs_random(ga_curves, rs_curves, path):
    fig, ax = plt.subplots(figsize=(6.2, 4))
    for curves, lab, col in ((ga_curves, "Genetic Algorithm", "C0"), (rs_curves, "Random search", "C3")):
        arr = np.array(curves); x = np.linspace(0, 1, arr.shape[1]) * 100
        ax.plot(x, arr.mean(0), color=col, lw=2, label=f"{lab} (mean of {len(arr)} runs)")
        ax.fill_between(x, arr.min(0), arr.max(0), color=col, alpha=.15)
    ax.set_xlabel("Fitness evaluations used (% of budget)"); ax.set_ylabel("Best fitness so far")
    ax.set_title("GA vs random search, equal evaluation budget"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_sensitivity(res, path):
    fig, axes = plt.subplots(1, len(res), figsize=(4.2 * len(res), 3.6))
    for ax, (name, d) in zip(axes, res.items()):
        labels = list(d.keys()); means = [np.mean(v) for v in d.values()]; sds = [np.std(v) for v in d.values()]
        ax.bar(range(len(labels)), means, yerr=sds, capsize=4, color="C0", alpha=.8)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
        ax.set_title(name); ax.set_ylabel("Final best fitness"); ax.grid(axis="y", alpha=.3)
        lo = min(means) - 2.5 * max(sds + [0.01]); ax.set_ylim(lo, max(means) + 2.5 * max(sds + [0.01]))
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_uniqueness(env, ga_best, rs_best, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, sols, t in ((axes[0], ga_best, "GA: best route of every seed"), (axes[1], rs_best, "Random search: best route of every seed")):
        _draw_env(ax, env, show_priority=True)
        for i, ch in enumerate(sols):
            for d in range(ch.xy.shape[0]):
                pts = np.vstack([env.cfg.base, ch.xy[d], env.cfg.base])
                ax.plot(pts[:, 0], pts[:, 1], color=COLORS[d % 5], lw=1, alpha=.45)
        ax.set_title(t, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
