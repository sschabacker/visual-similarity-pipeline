"""Regenerate results/headline.png from results/final_metrics.json.

    python make_headline_figure.py
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
m = json.loads((HERE / "results" / "final_metrics.json").read_text(encoding="utf-8"))

r, (lo, hi), ceil = m["consensus_r"], m["consensus_r_ci95"], m["noise_ceiling"]
pct_r = 100 * r / ceil              # ratio of correlations
pct_r2 = 100 * r**2 / ceil**2       # ratio of variance explained

fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150)
ax.bar(0, r, width=0.6, color="#4C72B0", edgecolor="#333", linewidth=0.6)
ax.bar(1, ceil, width=0.6, color="#C8CBD3", edgecolor="#333", linewidth=0.6)
ax.errorbar(0, r, yerr=[[r - lo], [hi - r]], color="#1f2d4d", capsize=5, linewidth=1.6)

# value labels sit above the error bar / bar so nothing is struck through
ax.text(0, hi + 0.025, f"{r:.2f}", ha="center", va="bottom", fontweight="bold")
ax.text(1, ceil + 0.025, f"{ceil:.2f}", ha="center", va="bottom", fontweight="bold")

ax.annotate(
    f"r is ~{pct_r:.0f}% of the ceiling\n(~{pct_r2:.0f}% in variance terms, r²)",
    xy=(0.33, r * 0.9), xytext=(0.5, 0.17), ha="center", fontsize=9, color="#1f2d4d",
    arrowprops=dict(arrowstyle="->", color="#1f2d4d"),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#9aa3b5"),
)

ax.set_xticks([0, 1])
ax.set_xticklabels(["Model\n(consensus r)", "Noise ceiling\n(human agreement)"])
ax.set_ylabel("Pearson r (vs consensus similarity)")
ax.set_ylim(0, 0.9)
ax.set_title("Model reaches about half of the human-agreement ceiling (in r)", fontsize=11.5)
ax.spines[["top", "right"]].set_visible(False)

fig.text(
    0.5, 0.01,
    f"{m['split']} · n = {m['n_unique_test_pairs']} unique test pairs · 95% CI [{lo:.3f}, {hi:.3f}]",
    ha="center", fontsize=8, color="#555",
)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(HERE / "results" / "headline.png")
print(f"wrote results/headline.png  (r={r}, ceiling={ceil}, {pct_r:.0f}% of r, {pct_r2:.0f}% of r^2)")
