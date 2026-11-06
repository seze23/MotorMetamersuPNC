"""Create a Figure 3a-style map from the complete 20-model grid."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "wang-grid-v1"
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataexp" / "wang_grid"
SUMMARY = DATA_DIR / "ensemble_target_window_summary.csv"
PNG = DATA_DIR / "wang_grid_figure3_style.png"
SVG = DATA_DIR / "wang_grid_figure3_style.svg"
COMPACT_PNG = DATA_DIR / "wang_grid_compact_direction_map.png"
COMPACT_SVG = DATA_DIR / "wang_grid_compact_direction_map.svg"
PURPLE = "#4b4fc4"
CORAL = "#ef786f"
GRAY = "#777777"


def torso(ax, alpha=1.0):
    ax.add_patch(FancyBboxPatch(
        (-15, -4), 30, 7, boxstyle="round,pad=0.3,rounding_size=3",
        facecolor="#d8d8d8", edgecolor=GRAY, linewidth=1.0, alpha=alpha,
    ))
    ax.add_patch(Circle((0, 1), 3.2, facecolor="#eeeeee",
                        edgecolor=GRAY, linewidth=0.9, alpha=alpha))
    ax.scatter([0], [0], marker="x", color=GRAY, s=18, zorder=4)


def compact_direction_map(data):
    """Plot bias vectors on a compressed display grid without changing direction."""
    column_spacing = 1.0
    row_spacing = 2.2
    vector_gain = 0.55  # display units per cm of physical localization bias
    x = data["grid_column"].to_numpy() * column_spacing
    y = data["grid_row"].to_numpy() * row_spacing
    dx = data["mean_outward_cm"].to_numpy() * vector_gain
    dy = data["mean_forward_cm"].to_numpy() * vector_gain

    fig, ax = plt.subplots(figsize=(13.2, 7.4))
    for x0, y0, du, dv in zip(x, y, dx, dy):
        ax.annotate(
            "", xy=(x0 + du, y0 + dv), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.15,
                            mutation_scale=8), zorder=3,
        )
    ax.scatter(x, y, s=10, color=PURPLE, zorder=4)

    x_tick_columns = np.array([0, 4, 9, 14, 19])
    x_lookup = data.groupby("grid_column")["unscaled_right_cm"].first()
    y_lookup = data.groupby("grid_row")["unscaled_forward_cm"].first()
    ax.set_xticks(x_tick_columns)
    ax.set_xticklabels([f"{x_lookup.loc[col]:.3f}" for col in x_tick_columns])
    ax.set_yticks(np.arange(5) * row_spacing)
    ax.set_yticklabels([f"{y_lookup.loc[row]:.3f}" for row in range(5)])
    ax.set_xlabel("Original Wang-grid rightward coordinate (cm; selected labels)")
    ax.set_ylabel("Original Wang-grid forward coordinate (cm)")
    ax.set_title(
        "Full-grid proprioceptive bias directions\n"
        "Compressed display grid; target positions are not spatially to scale",
        fontsize=13, pad=10,
    )
    ax.grid(True, color="#e8e8e8", linewidth=0.65)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.65, 20.6)
    ax.set_ylim(-0.65, 10.0)
    ax.tick_params(direction="out", length=4, labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color(GRAY)

    handles = [
        Line2D([0], [0], color=PURPLE, lw=1.4, marker=">", markersize=6,
               markevery=[1], label="20-model mean bias (direction preserved)"),
        Line2D([0], [0], marker="o", color=PURPLE, lw=0, markersize=4,
               label="Target display position"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, frameon=True,
              framealpha=0.96, edgecolor="#cccccc")
    ax.text(
        0.995, 0.018,
        "Arrows use one isotropic gain: 0.55 display units per cm of bias\n"
        "Tail = achieved target; head = decoded proprioceptive location",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(COMPACT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(COMPACT_SVG, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {COMPACT_PNG}")


def main():
    data = pd.read_csv(SUMMARY)
    data = data[data["window"] == "late_hold_250ms"].copy()
    if len(data) != 100 or not (data["model_count"] == 20).all():
        raise ValueError("Figure requires 100 targets and 20 models")

    first_muscle = np.load(
        ROOT / "outputs" / "wang_grid" / "muscles" / "r00_c00.npz"
    )
    start = first_muscle["wrist_xyz_body"][0, :2]

    fig, (ax, context) = plt.subplots(
        1, 2, figsize=(12.0, 7.2),
        gridspec_kw={"width_ratios": [4.8, 1.5], "wspace": 0.16},
    )
    for _, row in data.iterrows():
        ax.annotate(
            "", xy=(row.decoded_right_cm, row.decoded_forward_cm),
            xytext=(row.actual_right_cm, row.actual_forward_cm),
            arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.15,
                            mutation_scale=8), zorder=3,
        )
    ax.scatter(data.actual_right_cm, data.actual_forward_cm, s=9,
               color=PURPLE, zorder=4)
    ax.scatter(*start, marker="o", s=115, facecolor="none",
               edgecolor=CORAL, linewidth=2.0, zorder=5)
    ax.scatter(*start, marker="+", s=60, color=CORAL, linewidth=1.7, zorder=5)

    ax.set_xlim(-7.0, 7.0)
    ax.set_ylim(29.5, 37.5)
    ax.set_xticks(np.arange(-7, 7.1, 1))
    ax.set_yticks(np.arange(30, 37.1, 1))
    ax.set_xlabel("Rightward position relative to C7 (cm)")
    ax.set_ylabel("Forward position relative to C7 (cm)")
    ax.set_title(
        "Right-hand proprioceptive localization - full 100-target grid\n"
        "Achieved -> decoded position; late hold; mean of 20 matched models",
        fontsize=13, pad=10,
    )
    ax.grid(True, color="#e8e8e8", linewidth=0.65)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(direction="out", length=4, labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color("#777777")

    torso(context)
    context.scatter(*start, marker="o", s=45, facecolor="none",
                    edgecolor=CORAL, linewidth=1.5, zorder=5)
    context.add_patch(Rectangle(
        (data.actual_right_cm.min(), data.actual_forward_cm.min()),
        data.actual_right_cm.max() - data.actual_right_cm.min(),
        data.actual_forward_cm.max() - data.actual_forward_cm.min(),
        facecolor=PURPLE, edgecolor=PURPLE, alpha=0.18,
    ))
    context.set_xlim(-19, 19); context.set_ylim(-5, 39)
    context.set_xticks([-15, 0, 15]); context.set_yticks([0, 15, 30])
    context.tick_params(labelsize=7, length=3)
    context.set_title("Body context", fontsize=10)
    context.set_xlabel("Right (cm)", fontsize=8)
    context.set_ylabel("Forward (cm)", fontsize=8)
    context.set_aspect("equal", adjustable="box")
    context.grid(True, color="#eeeeee", linewidth=0.6)

    handles = [
        Line2D([0], [0], color=PURPLE, lw=1.4, marker=">", markersize=6,
               markevery=[1], label="20-model mean bias"),
        Line2D([0], [0], marker="o", color=PURPLE, lw=0, markersize=4,
               label="Achieved hand location"),
        Line2D([0], [0], marker="o", markerfacecolor="none",
               markeredgecolor=CORAL, markeredgewidth=1.6, color="none",
               markersize=8, label="Achieved common reach start"),
        Line2D([0], [0], marker="s", color=PURPLE, alpha=0.2, lw=0,
               markersize=8, label="Scaled target workspace"),
        Line2D([0], [0], marker="o", markerfacecolor="#d8d8d8",
               markeredgecolor=GRAY, color="none", markersize=7,
               label="Torso (C7 at 0, 0)"),
    ]
    context.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                   fontsize=7.5, frameon=True, framealpha=0.96,
                   edgecolor="#cccccc")
    ax.text(0.99, 0.015, "Uniform grid scale = 0.25",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="#555555")
    fig.tight_layout()
    fig.savefig(PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(SVG, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {PNG}")
    compact_direction_map(data)


if __name__ == "__main__":
    main()
