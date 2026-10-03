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
t = np.linspace(-2, 2, 400)
curve = lambda x: 0.40 * x**2 - 0.45
a = np.array([-1.25, curve(-1.25)])
c = np.array([-1.47, curve(-1.47)])
b = np.array([1.55, curve(1.55)])
cloud = a + rng.normal(size=(50, 2)) * [0.13, 0.09]
fig, axes = plt.subplots(2, 1, figsize=(8, 8.6), layout="constrained")
blue, orange = "#326a97", "#c76627"
for ax in axes:
    ax.plot(t, curve(t), color="#bcc3ca", lw=2, zorder=1)
    ax.scatter(cloud[:, 0], cloud[:, 1], s=14, color=blue, alpha=.40, zorder=2)
    ax.add_patch(Ellipse(a, .85, .60, facecolor=blue, alpha=.08, edgecolor="none"))
    ax.scatter(*a, s=85, color=blue, edgecolor="white", zorder=6)
    ax.annotate("A", a, xytext=(-2, -22), textcoords="offset points", color=blue,
                fontsize=14, weight="bold")
    ax.text(-1.85, -.35, "Existing organization", color=blue, fontsize=10)
    ax.text(.85, 1.12, "Manifold", color="#777e86", fontsize=10)
    ax.set(xlim=(-2.05, 2.05), ylim=(-.90, 1.40), aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
axes[0].set_title("1. Similar members: a narrow interpolation range", loc="left", pad=14)
axes[0].plot([a[0], c[0]], [a[1], c[1]], color=blue, lw=4)
axes[0].scatter(*c, s=38, color=blue, zorder=5)
axes[0].annotate("Nearby member", c, xytext=(0, 28), textcoords="offset points",
                 color=blue, fontsize=10, ha="center", arrowprops={"arrowstyle": "-", "color": blue})
end = a + 2.8 * (a - c)
for ax in axes:
    ax.annotate("", end, a, arrowprops={"arrowstyle": "->", "color": "#7c8792",
                                         "linestyle": "--", "lw": 1.5}, zorder=3)
axes[0].text(-.40, -.69, "Extrapolation from A", color="#66717c", fontsize=10)
axes[1].set_title("2. A distant newcomer: a new interpolation path", loc="left", pad=14)
axes[1].plot([a[0], b[0]], [a[1], b[1]], color=orange, lw=2.5, zorder=4)
alphas = np.linspace(.15, .85, 6)
points = (1 - alphas[:, None]) * a + alphas[:, None] * b
axes[1].scatter(points[:, 0], points[:, 1], color=orange, s=25, zorder=5)
axes[1].scatter(*b, s=90, color=orange, edgecolor="white", zorder=6)
axes[1].annotate("B", b, xytext=(5, 10), textcoords="offset points",
                 color=orange, fontsize=14, weight="bold")
axes[1].text(.55, .74, "Distant newcomer", color=orange, fontsize=10)
axes[1].text(-.45, -.10, "New exploration path", color=orange, fontsize=10)
axes[1].text(0, -.75, r"$\mathbf{x}(\alpha)=(1-\alpha)\mathbf{x}_A+\alpha\mathbf{x}_B$",
             ha="center", fontsize=13)
fig.savefig(OUT / "exploration-space.svg", facecolor="white")
svg = OUT / "exploration-space.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
fig.savefig("/tmp/okdalto-company-growth-figure.png", dpi=160, facecolor="white")
print(OUT / "exploration-space.svg")
