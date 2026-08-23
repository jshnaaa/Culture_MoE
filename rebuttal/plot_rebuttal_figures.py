"""
Generate two figures for the rebuttal:
  1. expert-region.png  — Expert × Cultural-Region activation-ratio heatmaps
                          (Llama-3.1-8B backbone only; full CulDPD vs w/o CRL control)
  2. Expert-count.png  — Expert-count ablation line plots
                          (Llama-3.1-8B and Qwen2.5-7B backbones, separate subplots)

All experimental values are hardcoded below (mirrors review.docx tables).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---------- global style ----------
rcParams["font.family"] = "DejaVu Sans"
rcParams["axes.unicode_minus"] = False
rcParams["font.size"] = 11

CONTINENTS = ["Asia", "Europe", "N.America", "S.America", "Africa", "Oceania"]
EXPERTS = ["Expert 1", "Expert 2", "Expert 3", "Expert 4"]

# ---------- Expert × Region activation ratios (Llama-3.1-8B backbone) ----------
# rows: experts, cols: continents
full_culdpd_llama = np.array([
    [0.95, 0.24, 0.27, 0.25, 0.40, 0.37],
    [0.25, 0.96, 0.85, 0.63, 0.17, 0.84],
    [0.10, 0.15, 0.31, 0.93, 0.89, 0.30],
    [0.70, 0.65, 0.57, 0.19, 0.54, 0.49],
], dtype=float)

wo_crl_llama = np.array([
    [0.72, 0.45, 0.31, 0.43, 0.51, 0.42],
    [0.40, 0.66, 0.64, 0.52, 0.37, 0.63],
    [0.42, 0.29, 0.63, 0.67, 0.65, 0.47],
    [0.46, 0.60, 0.42, 0.38, 0.47, 0.48],
], dtype=float)


def plot_expert_region():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    matrices = [(full_culdpd_llama, "Full CulDPD"),
                (wo_crl_llama, "w/o CRL (control)")]
    vmax = 1.0
    vmin = 0.0
    im = None
    for ax, (mat, title) in zip(axes, matrices):
        im = ax.imshow(mat, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(CONTINENTS)))
        ax.set_xticklabels(CONTINENTS, rotation=30, ha="right")
        ax.set_yticks(range(len(EXPERTS)))
        ax.set_yticklabels(EXPERTS)
        ax.set_title(title, fontsize=12, pad=8)
        # annotate values
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                color = "white" if v >= 0.6 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=color, fontsize=10)
        ax.set_xlabel("Cultural Region", fontsize=10)
        ax.set_ylabel("Expert", fontsize=10)
    # shared colorbar
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Activation Ratio", fontsize=10)
    fig.suptitle("Expert × Cultural-Region Activation Ratio (Llama-3.1-8B backbone)",
                 fontsize=13, y=1.02)
    plt.savefig("expert-region.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved: expert-region.png")


# ---------- Expert-count ablation ----------
DATASETS = ["CulturalBench", "CultureLLM", "NormAd", "CultureAtlas"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
MARKERS = ["o", "s", "^", "D"]

# rows: N=1..6, cols: 4 datasets; None = missing
ablation_llama = np.array([
    [78.69, 90.31, 96.20, 76.44],  # N=1
    [80.33, 90.91, 96.58, 77.01],  # N=2
    [80.33, 91.01, 97.34, 78.74],  # N=3
    [82.46, 91.43, 99.24, 78.86],  # N=4 (default)
    [83.61, 90.11, 98.10, 78.74],  # N=5
    [82.79, 90.31, 98.48, 78.16],  # N=6
], dtype=float)

# Qwen N=6: only CulturalBench filled (83.61); others missing
ablation_qwen = np.array([
    [79.51, 91.61, 96.58, 78.16],  # N=1
    [80.33, 91.71, 96.96, 79.31],  # N=2
    [83.61, 92.01, 98.10, 81.03],  # N=3
    [83.94, 92.83, 98.86, 81.71],  # N=4 (default)
    [84.43, 92.71, 98.86, 81.61],  # N=5
    [83.61, np.nan, np.nan, np.nan],  # N=6 (partial)
], dtype=float)

NS = [1, 2, 3, 4, 5, 6]


def plot_ablation(ax, data, title):
    for j, ds in enumerate(DATASETS):
        ys = data[:, j]
        valid = ~np.isnan(ys)
        xs = np.array(NS)[valid]
        ys_v = ys[valid]
        ax.plot(xs, ys_v, color=COLORS[j], marker=MARKERS[j],
                linewidth=2, markersize=7, label=ds)
        # annotate values
        for x, y in zip(xs, ys_v):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, color=COLORS[j])
    # mark default N=4
    ax.axvline(4, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(4, ax.get_ylim()[0], "default", color="gray", fontsize=9,
            ha="center", va="bottom")
    ax.set_xticks(NS)
    ax.set_xlabel("# Experts N", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)


def plot_expert_count():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    plot_ablation(axes[0], ablation_llama, "Llama-3.1-8B backbone")
    plot_ablation(axes[1], ablation_qwen, "Qwen2.5-7B backbone")
    # adjust y-limits per subplot for clarity
    axes[0].set_ylim(74, 101)
    axes[1].set_ylim(76, 101)
    fig.suptitle("Expert-count Ablation", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("Expert-count.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved: Expert-count.png")


if __name__ == "__main__":
    plot_expert_region()
    plot_expert_count()
