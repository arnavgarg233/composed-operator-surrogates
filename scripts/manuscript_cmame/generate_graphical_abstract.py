"""Generate the CMAME graphical abstract.

Every plotted value is a ten-seed median degradation ratio read from the family
result files (narrow and mean-broadened: `family2_resolution_horizon/results/
N256_L1/RESULT.json`, `family1_rerun/results/*/RESULT.json`,
`family1_second_pairs/results/P3_negative_control/RESULT.json`; state-space:
`family1c_state_support/RESULT.json`, `family1e_state_broadening_new_pairs/
results/RESULT.json`).
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parents[1] / "figures" / "graphical_abstract.pdf"

# CMAME minimum canvas: 1328 x 531 px at 300 dpi.
fig, ax = plt.subplots(figsize=(1328 / 300, 531 / 300), dpi=300)
fig.patch.set_facecolor("white")
ax.set_facecolor("#f7f8fa")

pairs = ["Fisher–KPP\nr = 1.0", "Fisher–KPP\nr = 0.5", "Gray–Scott", "Cahn–Hilliard"]
narrow = np.array([115.4, 58.68, 105.6, 46.28])
mean_broad = np.array([15.98, 16.41, 28.55, 35.80])
state = np.array([2.569, 1.948, 3.70, 0.787])
x = np.arange(len(pairs))
width = 0.26

ax.bar(x - width, narrow, width, label="Narrow dictionary", color="#9aa7b8")
ax.bar(x, mean_broad, width, label="Mean-broadened", color="#5b9bd5")
ax.bar(x + width, state, width, label="State-space broadened", color="#d95f02")
ax.set_yscale("log")
ax.set_ylabel("Degradation ratio $R$", fontsize=8)
ax.set_xticks(x, pairs, fontsize=8)
title = ax.set_title("Broadening training dictionaries in state space\n"
                     "repairs composed-operator degradation on four pairs",
                     fontsize=8.5, pad=5)
ax.grid(axis="y", which="both", color="#d9dde3", linewidth=0.6, alpha=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=1,
          handlelength=1.4, borderaxespad=0.2)
ax.set_ylim(0.3, 4000)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
fig.tight_layout(pad=0.5)

# Render-time guard: the title must not run past the canvas.
fig.canvas.draw()
_tb = title.get_window_extent(fig.canvas.get_renderer())
if _tb.x0 < 1.0 or _tb.x1 > fig.bbox.x1 - 1.0:
    raise RuntimeError(
        f"graphical-abstract title overruns the canvas: {_tb.x0:.1f}..{_tb.x1:.1f} "
        f"against 0..{fig.bbox.x1:.1f}")
fig.savefig(OUT, format="pdf", dpi=300,
            metadata={"Creator": "generate_graphical_abstract.py",
                      "CreationDate": datetime(2000, 1, 1)})
plt.close(fig)
