"""Render the conceptual figure for Company Growth and Diversity."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/2026-10-04-company-growth-diversity"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "svg.fonttype": "path"})
rng = np.random.default_rng(42)
a = np.array([1.6, 1.0])
c = np.array([1.28, .85])
b = a + 16 * (a - c)
cloud = a + rng.normal(size=(50, 2)) * [.20, .13]
fig, axes = plt.subplots(2, 1, figsize=(8, 8.6), layout="constrained")
blue, orange = "#326a97", "#c76627"
for ax in axes:
    ax.scatter(cloud[:, 0], cloud[:, 1], s=14, color=blue, alpha=.40, zorder=2)
    ax.add_patch(Ellipse(a, 1.35, .95, facecolor=blue, alpha=.08, edgecolor="none"))
    ax.scatter(*a, s=85, color=blue, edgecolor="white", zorder=6)
    ax.annotate("A", a, xytext=(-2, -22), textcoords="offset points", color=blue,
                fontsize=14, weight="bold")
    ax.text(.35, .25, "Existing organization", color=blue, fontsize=12)
    ax.set(xlim=(-.25, 8), ylim=(-.70, 4.6), aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.annotate("", (7.8, 0), (0, 0), arrowprops={"arrowstyle": "->", "color": "#aab1b8", "lw": 1.2})
    ax.annotate("", (0, 4.35), (0, 0), arrowprops={"arrowstyle": "->", "color": "#aab1b8", "lw": 1.2})
    ax.text(7.9, -.02, "$x$", color="#727b83", fontsize=14)
    ax.text(-.08, 4.47, "$y$", color="#727b83", fontsize=14)
axes[0].set_title("1. Similar members", loc="left", pad=14, fontsize=14)
axes[0].plot([a[0], c[0]], [a[1], c[1]], color=blue, lw=4)
axes[0].scatter(*c, s=38, color=blue, zorder=5)
axes[0].annotate("Narrow interpolation", c, xytext=(.5, 2.1),
                 color=blue, fontsize=12, arrowprops={"arrowstyle": "-", "color": blue})
end = a + 12 * (a - c)
axes[0].annotate("", end, a, arrowprops={"arrowstyle": "->", "color": "#7c8792",
                                          "linestyle": "--", "lw": 1.5}, zorder=3)
axes[0].text(3.4, 2.65, "Extrapolation", color="#66717c", fontsize=12)
axes[1].set_title("2. A distant newcomer", loc="left", pad=14, fontsize=14)
axes[1].plot([a[0], b[0]], [a[1], b[1]], color=orange, lw=2.5, zorder=4)
alphas = np.linspace(.15, .85, 6)
points = (1 - alphas[:, None]) * a + alphas[:, None] * b
axes[1].scatter(points[:, 0], points[:, 1], color=orange, s=25, zorder=5)
axes[1].scatter(*b, s=90, color=orange, edgecolor="white", zorder=6)
axes[1].annotate("B", b, xytext=(5, 10), textcoords="offset points",
                 color=orange, fontsize=14, weight="bold")
axes[1].text(5.25, 3.95, "Distant newcomer", color=orange, fontsize=12)
axes[1].text(3.05, 1.40, "New interpolation range", color=orange, fontsize=12)
axes[1].text(4, -.55, r"$\mathbf{x}(\alpha)=(1-\alpha)\mathbf{x}_A+\alpha\mathbf{x}_B$",
             ha="center", fontsize=13)
fig.savefig(OUT / "exploration-space.svg", facecolor="white")
svg = OUT / "exploration-space.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
fig.savefig("/tmp/okdalto-company-growth-figure.png", dpi=160, facecolor="white")
print(OUT / "exploration-space.svg")
