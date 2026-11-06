"""Create a Wang Figure 3a-style map for the five-target pilot."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "wang-pilot-v1"
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataexp" / "wang_pilot"
OUTPUT_PNG = DATA_DIR / "pilot_figure3_style.png"
OUTPUT_SVG = DATA_DIR / "pilot_figure3_style.svg"
MODEL_FILES = sorted(DATA_DIR.glob("pilot_model_*_bias.csv"))

PURPLE = "#4b4fc4"
LIGHT_PURPLE = "#aaa9dc"
CORAL = "#ef786f"
TORSO = "#d8d8d8"
TORSO_EDGE = "#707070"


def draw_torso(ax):
    """Draw a compact top-view torso cue centered on the C7 body origin."""
    ax.add_patch(FancyBboxPatch(
        (-15.0, -4.0), 30.0, 7.0,
        boxstyle="round,pad=0.3,rounding_size=3.0",
        facecolor=TORSO, edgecolor=TORSO_EDGE, linewidth=1.2, zorder=0,
    ))
    ax.add_patch(Circle((0.0, 1.0), 3.2, facecolor="#eeeeee",
                        edgecolor=TORSO_EDGE, linewidth=1.0, zorder=0))
    ax.scatter([0.0], [0.0], marker="x", s=25, color=TORSO_EDGE,
               linewidth=1.2, zorder=1)
    ax.text(0.0, -3.7, "Torso / C7 reference", ha="center", va="top",
            fontsize=8.5, color="#555555")


def main():
    if not MODEL_FILES:
        raise FileNotFoundError("No pilot_model_*_bias.csv files found")
    data = pd.concat([pd.read_csv(path) for path in MODEL_FILES], ignore_index=True)
    late = data[data["window"] == "late_hold_250ms"].copy()
    if late["model"].nunique() != 4:
        raise ValueError("Figure expects the four valid coefficient-seed-0 models")

    order = ["center_left", "near_center", "center", "far_center", "center_right"]
    means = late.groupby("target", sort=False).agg(
        actual_right_cm=("actual_right_cm", "mean"),
        actual_forward_cm=("actual_forward_cm", "mean"),
        decoded_right_cm=("decoded_right_cm", "mean"),
        decoded_forward_cm=("decoded_forward_cm", "mean"),
    )

    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    draw_torso(ax)

    # Individual matched models, analogous to repeated localization maps.
    for _, row in late.iterrows():
        ax.annotate(
            "",
            xy=(row["decoded_right_cm"], row["decoded_forward_cm"]),
            xytext=(row["actual_right_cm"], row["actual_forward_cm"]),
            arrowprops=dict(arrowstyle="->", color=LIGHT_PURPLE,
                            lw=1.0, alpha=0.75, mutation_scale=8),
            zorder=2,
        )

    # Bold across-model mean vectors.
    for target in order:
        row = means.loc[target]
        ax.annotate(
            "",
            xy=(row["decoded_right_cm"], row["decoded_forward_cm"]),
            xytext=(row["actual_right_cm"], row["actual_forward_cm"]),
            arrowprops=dict(arrowstyle="-|>", color=PURPLE,
                            lw=2.2, mutation_scale=14),
            zorder=4,
        )
        ax.scatter(row["actual_right_cm"], row["actual_forward_cm"],
                   s=25, color=PURPLE, edgecolor="white", linewidth=0.5,
                   zorder=5)

    # Common center-out origin. It coincides with the center target, so use an
    # open marker large enough to remain visible behind the target dot.
    # All trials share this same canonical achieved center posture. Use its
    # physical Handle coordinate rather than the nominal requested target.
    start_right = means.loc["center", "actual_right_cm"]
    start_forward = means.loc["center", "actual_forward_cm"]
    ax.scatter(start_right, start_forward, marker="o", s=150,
               facecolor="none", edgecolor=CORAL, linewidth=2.0, zorder=3)
    ax.scatter(start_right, start_forward, marker="+", s=75,
               color=CORAL, linewidth=1.8, zorder=3)

    ax.set_xlim(-25, 25)
    ax.set_ylim(-5, 50)
    ax.set_xticks(range(-25, 26, 5))
    ax.set_yticks(range(0, 51, 5))
    ax.set_xlabel("Rightward position relative to C7 (cm)", fontsize=11)
    ax.set_ylabel("Forward position relative to C7 (cm)", fontsize=11)
    ax.set_title(
        "Right-hand proprioceptive localization pilot\n"
        "Achieved -> decoded position; late hold; four matched models",
        fontsize=13, pad=12,
    )
    ax.grid(True, color="#e7e7e7", linewidth=0.7, zorder=-2)
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(1.0)
    ax.tick_params(direction="out", length=4, color="#555555", labelsize=9)

    legend_handles = [
        Line2D([0], [0], color=PURPLE, lw=2.2, marker=">", markersize=7,
               markevery=[1], label="Across-model mean bias"),
        Line2D([0], [0], color=LIGHT_PURPLE, lw=1.0,
               label="Individual matched model"),
        Line2D([0], [0], marker="o", color=PURPLE, lw=0, markersize=5,
               label="Achieved hand location"),
        Line2D([0], [0], marker="o", markerfacecolor="none",
               markeredgecolor=CORAL, markeredgewidth=1.8, color="none",
               markersize=9, label="Achieved common reach start"),
        Line2D([0], [0], marker="o", markerfacecolor=TORSO,
               markeredgecolor=TORSO_EDGE, color="none", markersize=8,
               label="Torso (C7 at x = 0, y = 0)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True,
              facecolor="white", edgecolor="#cccccc", framealpha=0.96,
              fontsize=8.5, borderpad=0.8)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_SVG, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_SVG}")


if __name__ == "__main__":
    main()
