"""Render the x-y exploration-space figure used by both language editions."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Ellipse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/2026-10-04-company-growth-diversity"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path"})
rng = np.random.default_rng(42)
a = np.array([5.75, 1.1])
b = np.array([1.1, 5.75])
blue, orange = "#537d9a", "#c88460"
fig, ax = plt.subplots(figsize=(7, 7))
fig.subplots_adjust(left=.05, right=.95, bottom=.05, top=.95)
ax.set(xlim=(-.35, 7.1), ylim=(-.35, 7.1), aspect="equal")
ax.axis("off")
for endpoint in [(6.9, 0), (0, 6.9)]:
    ax.annotate("", endpoint, (0, 0), arrowprops={"arrowstyle": "->", "color": "#b9bfc4", "lw": 1.25})
ax.text(7.02, 0, "$x$", ha="center", va="center", fontsize=19, color="#7b858d")
ax.text(0, 7.02, "$y$", ha="center", va="center", fontsize=19, color="#7b858d")
for center, color, count in [(a, blue, 32), (b, orange, 20)]:
    ax.add_patch(Ellipse(center, 1.22, 1.22, facecolor=color, edgecolor="none", alpha=.09))
    samples = center + rng.normal(size=(count, 2)) * .15
    ax.scatter(samples[:, 0], samples[:, 1], s=16, color=color, alpha=.5, edgecolors="none", zorder=3)
ax.plot([a[0], b[0]], [a[1], b[1]], color="#adb5bc", lw=1.5, zorder=2)
alphas = np.linspace(.16, .84, 5)
points = (1-alphas[:, None])*a + alphas[:, None]*b
colors = (1-alphas[:, None])*np.array(to_rgb(blue)) + alphas[:, None]*np.array(to_rgb(orange))
ax.scatter(points[:, 0], points[:, 1], s=28, c=colors, edgecolors="white", linewidths=.6, zorder=4)
ax.scatter(*a, s=95, color=blue, edgecolor="white", linewidth=1.6, zorder=5)
ax.scatter(*b, s=95, color=orange, edgecolor="white", linewidth=1.6, zorder=5)
ax.text(a[0]+.80, a[1], "$A$", color=blue, fontsize=25, va="center", ha="center")
ax.text(b[0], b[1]+.80, "$B$", color=orange, fontsize=25, va="center", ha="center")
fig.savefig(OUT / "exploration-space.svg", facecolor="white")
svg = OUT / "exploration-space.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
fig.savefig("/tmp/okdalto-company-growth-figure.png", dpi=160, facecolor="white")
print(OUT / "exploration-space.svg")
