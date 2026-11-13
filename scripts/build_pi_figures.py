from pathlib import Path
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection


ROOT = Path(__file__).resolve().parents[1]
_archived_data = ROOT / "local_artifacts" / "dataexp"
SRC = Path(os.environ.get(
    "MOTOR_META_DATAEXP",
    _archived_data if _archived_data.exists() else ROOT / "dataexp",
))
OUT = ROOT / "figures" / "height_bias"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 13,
    "axes.titlesize": 17,
    "axes.labelsize": 14,
})


def save_workspace_height():
    pose_path = SRC / "poster_bias_investigation" / "final_workspace_comparison_figures" / "opensim_3d_poster_vs_wang_postures.png"
    pose = Image.open(pose_path)
    w, h = pose.size
    # Crop away the large title/footer while retaining the two labeled posture panels.
    pose_crop = pose.crop((0, int(0.07*h), w, int(0.94*h)))

    fig = plt.figure(figsize=(14, 7.5), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.85, 1], wspace=0.12)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(pose_crop)
    ax_img.axis("off")
    ax_img.set_title("Representative OpenSim postures", pad=10, weight="bold")

    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(-0.65, 1.75)
    ax.set_ylim(-30, 12)
    ax.axhline(0, color="#20242b", lw=2)
    ax.text(1.68, 0.8, "C7 marker height (0 cm)", ha="right", va="bottom", weight="bold")

    # Wang plane
    ax.hlines(-23.2, -0.35, 0.55, color="#2a6fbb", lw=8)
    ax.scatter([0.1], [-23.2], s=120, color="#2a6fbb", zorder=3)
    ax.text(0.64, -23.2, "Wang plane\n−23.2 cm", va="center", color="#1f578f", weight="bold")

    # Poster workspace range and center
    poster_low, poster_high, poster_center = -2.86, 7.91, 3.86
    ax.add_patch(Rectangle((-0.35, poster_low), 0.9, poster_high-poster_low,
                           facecolor="#b14aa7", alpha=0.22, edgecolor="#8e2a87", lw=2))
    ax.hlines(poster_center, -0.35, 0.55, color="#8e2a87", lw=7)
    ax.scatter([0.1], [poster_center], s=120, color="#8e2a87", zorder=3)
    ax.text(0.64, poster_center, "Poster endpoints\nmean +3.9 cm", va="center",
            color="#7b2575", weight="bold")
    ax.text(0.1, poster_high+1.0, "+7.9", ha="center", color="#7b2575")
    ax.text(0.1, poster_low-1.0, "−2.9", ha="center", va="top", color="#7b2575")

    ax.annotate("", xy=(-0.22, poster_center), xytext=(-0.22, -23.2),
                arrowprops=dict(arrowstyle="<->", color="#444", lw=1.8))
    ax.text(-0.32, -9.5, "27.1 cm separation", ha="right", va="center",
            rotation=90, color="#333")
    ax.set_ylabel("Physical endpoint height relative to body (cm)")
    ax.set_xticks([])
    ax.set_yticks([-25, -20, -15, -10, -5, 0, 5, 10])
    ax.grid(axis="y", color="#d9dde3", lw=0.8)
    for s in ["top", "right", "bottom"]:
        ax.spines[s].set_visible(False)
    ax.set_title("The workspaces differ mainly in height", pad=10, weight="bold")

    fig.suptitle("Physical location of the tested workspaces", fontsize=23, weight="bold", y=0.99)
    fig.text(0.5, 0.015,
             "Heights refer to the physical endpoint. The decoded outcome is lateral error relative to the analytic wrist training label.",
             ha="center", fontsize=11, color="#4c5159")
    fig.savefig(OUT / "01_workspace_height_relative_to_body.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def load_height_data():
    base = SRC / "poster_bias_investigation" / "wang_vertical_shift_full"
    summary = pd.read_csv(base / "target_summary.csv")
    models = pd.read_csv(base / "model_target_results.csv")
    physical = models.groupby("target", as_index=False).agg(
        right=("physical_right_cm", "mean"), forward=("physical_forward_cm", "mean"))
    return summary.merge(physical, on="target", how="left")


def load_height_model_data():
    base = SRC / "poster_bias_investigation" / "wang_vertical_shift_full"
    return pd.read_csv(base / "model_target_results.csv")


def save_height_arrow_comparison():
    data = load_height_data()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharex=True, sharey=True, constrained_layout=True)
    panels = [
        ("original_outward_cm", "Lower plane: −23.2 cm", "#2a6fbb"),
        ("shifted_outward_cm", "Higher plane: +4.0 cm", "#b43b2b"),
    ]
    for ax, (column, title, color) in zip(axes, panels):
        x = data.right.to_numpy(); y = data.forward.to_numpy(); bias = data[column].to_numpy()
        ax.scatter(x, y, s=19, color=color, zorder=3)
        ax.quiver(x, y, bias, np.zeros_like(bias), angles="xy", scale_units="xy", scale=1,
                  width=.004, headwidth=4.4, headlength=5.4, color=color, zorder=4)
        ax.axvline(0, color="#777", lw=.7); ax.grid(alpha=.2)
        ax.set_title(f"{title}\nmean {bias.mean():+.2f} cm; {(bias > 0).sum()}/{len(bias)} targets outward",
                     weight="bold")
        ax.set_xlabel("Physical right relative to C7 (cm)")
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Physical forward relative to C7 (cm)")
    fig.suptitle("Raising the same Wang grid reverses the lateral decoding bias", fontsize=22, weight="bold")
    fig.text(.5, .01, "Arrows run from the physical endpoint to the decoded analytic-wrist position; horizontal length is lateral bias.",
             ha="center", fontsize=11, color="#4c5159")
    fig.savefig(OUT / "02_wang_height_arrow_comparison.png", dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_height_full_vector_comparison():
    data = load_height_data()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharex=True, sharey=True, constrained_layout=True)
    panels = [
        ("original_outward_cm", "original_forward_cm", "Lower plane: −23.2 cm", "#2a6fbb"),
        ("shifted_outward_cm", "shifted_forward_cm", "Higher plane: +4.0 cm", "#b43b2b"),
    ]
    all_x=[data.right.to_numpy()]; all_y=[data.forward.to_numpy()]
    for right_col, forward_col, _, _ in panels:
        all_x.append(data.right.to_numpy()+data[right_col].to_numpy())
        all_y.append(data.forward.to_numpy()+data[forward_col].to_numpy())
    x_min=min(v.min() for v in all_x)-1.0; x_max=max(v.max() for v in all_x)+1.0
    y_min=min(v.min() for v in all_y)-1.0; y_max=max(v.max() for v in all_y)+1.0
    for ax, (right_col, forward_col, title, color) in zip(axes, panels):
        x=data.right.to_numpy(); y=data.forward.to_numpy()
        dx=data[right_col].to_numpy(); dy=data[forward_col].to_numpy()
        ax.scatter(x, y, s=16, color=color, zorder=3)
        ax.quiver(x, y, dx, dy, angles="xy", scale_units="xy", scale=1,
                  width=.004, headwidth=4.2, headlength=5.2, color=color, zorder=4)
        ax.axvline(0, color="#777", lw=.7); ax.grid(alpha=.2)
        ax.set_title(f"{title}\nmean vector: ({dx.mean():+.2f} right, {dy.mean():+.2f} forward) cm", weight="bold")
        ax.set_xlabel("Physical right relative to C7 (cm)")
        ax.set_xlim(x_min,x_max); ax.set_ylim(y_min,y_max)
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Physical forward relative to C7 (cm)")
    fig.suptitle("Full horizontal decoding-error vectors", fontsize=22, weight="bold")
    fig.text(.5, .01, "Arrow tails are physical endpoints; arrowheads are decoded analytic-wrist positions. Both lateral and forward components are shown at physical scale.",
             ha="center", fontsize=11, color="#4c5159")
    fig.savefig(OUT / "02b_wang_height_full_vector_comparison.png", dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_height_arrow_comparison_with_models():
    means=load_height_data(); models=load_height_model_data()
    fig,axes=plt.subplots(1,2,figsize=(15,6.5),sharex=True,sharey=True)
    panels=[("original_outward_cm","Lower plane: −23.2 cm","#2a6fbb"),
            ("shifted_outward_cm","Higher plane: +4.0 cm","#b43b2b")]
    for ax,(column,title,color) in zip(axes,panels):
        segments=[]
        for row in models.itertuples():
            segments.append([(row.physical_right_cm,row.physical_forward_cm),
                             (row.physical_right_cm+getattr(row,column),row.physical_forward_cm)])
        ax.add_collection(LineCollection(segments,colors="#a9adb3",linewidths=.55,alpha=.16,zorder=1))
        x=means.right.to_numpy();y=means.forward.to_numpy();bias=means[column].to_numpy()
        ax.scatter(x,y,s=18,color=color,zorder=3)
        ax.quiver(x,y,bias,np.zeros_like(bias),angles="xy",scale_units="xy",scale=1,width=.004,
                  headwidth=4.4,headlength=5.4,color=color,zorder=4)
        ax.axvline(0,color="#777",lw=.7);ax.grid(alpha=.2)
        ax.set_title(f"{title}\nmean {bias.mean():+.2f} cm; {(bias>0).sum()}/{len(bias)} targets outward",weight="bold")
        ax.set_xlabel("Physical right relative to C7 (cm)");ax.set_aspect("equal",adjustable="box")
    axes[0].set_ylabel("Physical forward relative to C7 (cm)")
    fig.suptitle("Lateral bias with individual-model dispersion",fontsize=22,weight="bold")
    fig.tight_layout(rect=(0,.09,1,.94),w_pad=2.5)
    fig.text(.5,.01,"Light gray lines are individual CNN biases; colored arrows are 20-model target means.",ha="center",fontsize=11,color="#4c5159")
    fig.savefig(OUT/"02_wang_height_arrow_comparison_with_models.png",dpi=250,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def save_height_full_vector_comparison_with_models():
    means=load_height_data();models=load_height_model_data()
    fig,axes=plt.subplots(1,2,figsize=(15,6.5),sharex=True,sharey=True)
    panels=[("original_outward_cm","original_forward_cm","Lower plane: −23.2 cm","#2a6fbb"),
            ("shifted_outward_cm","shifted_forward_cm","Higher plane: +4.0 cm","#b43b2b")]
    all_x=[models.physical_right_cm.to_numpy()];all_y=[models.physical_forward_cm.to_numpy()]
    for rc,fc,_,_ in panels:
        all_x.append(models.physical_right_cm.to_numpy()+models[rc].to_numpy())
        all_y.append(models.physical_forward_cm.to_numpy()+models[fc].to_numpy())
    x_min=min(v.min() for v in all_x)-1;x_max=max(v.max() for v in all_x)+1
    y_min=min(v.min() for v in all_y)-1;y_max=max(v.max() for v in all_y)+1
    for ax,(rc,fc,title,color) in zip(axes,panels):
        segments=[]
        for row in models.itertuples():
            segments.append([(row.physical_right_cm,row.physical_forward_cm),
                             (row.physical_right_cm+getattr(row,rc),row.physical_forward_cm+getattr(row,fc))])
        ax.add_collection(LineCollection(segments,colors="#a9adb3",linewidths=.5,alpha=.12,zorder=1))
        x=means.right.to_numpy();y=means.forward.to_numpy();dx=means[rc].to_numpy();dy=means[fc].to_numpy()
        ax.scatter(x,y,s=16,color=color,zorder=3)
        ax.quiver(x,y,dx,dy,angles="xy",scale_units="xy",scale=1,width=.004,headwidth=4.2,headlength=5.2,color=color,zorder=4)
        ax.axvline(0,color="#777",lw=.7);ax.grid(alpha=.2);ax.set_xlim(x_min,x_max);ax.set_ylim(y_min,y_max)
        ax.set_title(f"{title}\nmean vector: ({dx.mean():+.2f} right, {dy.mean():+.2f} forward) cm",weight="bold")
        ax.set_xlabel("Physical right relative to C7 (cm)");ax.set_aspect("equal",adjustable="box")
    axes[0].set_ylabel("Physical forward relative to C7 (cm)")
    fig.suptitle("Full horizontal vectors with individual-model dispersion",fontsize=22,weight="bold")
    fig.tight_layout(rect=(0,.09,1,.94),w_pad=2.5)
    fig.text(.5,.01,"Light gray lines are individual CNN vectors; colored arrows are 20-model target means.",ha="center",fontsize=11,color="#4c5159")
    fig.savefig(OUT/"02b_wang_height_full_vector_comparison_with_models.png",dpi=250,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def save_height_heatmap_comparison():
    data = load_height_data()
    def grid(column):
        values = np.full((5, 20), np.nan)
        for row in data.itertuples():
            values[int(row.target[1:3]), int(row.target[5:7])] = getattr(row, column)
        return values
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.7), constrained_layout=True)
    panels = [
        (grid("original_outward_cm"), "Lower plane: −23.2 cm"),
        (grid("shifted_outward_cm"), "Higher plane: +4.0 cm"),
    ]
    image = None
    for ax, (values, title) in zip(axes, panels):
        image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3, origin="upper")
        ax.set_title(title, weight="bold"); ax.set_ylabel("Grid row")
        ax.set_yticks(range(5)); ax.set_xticks(range(20)); ax.set_xticklabels(range(20), fontsize=8)
        for row, col in np.argwhere(np.isnan(values)):
            ax.text(col, row, "×", ha="center", va="center", weight="bold")
    axes[-1].set_xlabel("Grid column")
    fig.colorbar(image, ax=axes, label="Lateral outward component (cm)", pad=.015)
    fig.suptitle("Same grid, isolated endpoint-height manipulation", fontsize=22, weight="bold")
    fig.savefig(OUT / "03_wang_height_heatmap_comparison.png", dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_lower_grid_start_schematic():
    right = (np.arange(20) - 9.5) * 2.438
    forward = np.array([22.671, 28.063, 33.455, 38.847, 44.239])
    xx, yy = np.meshgrid(right, forward)
    start = np.array([0.0, 33.455])

    fig, ax = plt.subplots(figsize=(6.7, 7.2))
    # Body context in the style of the reference schematic.
    torso = FancyBboxPatch((-24, -4), 48, 7, boxstyle="round,pad=0.35,rounding_size=4",
                           facecolor="#d9d9d9", edgecolor="#888888", linewidth=1.2, zorder=1)
    ax.add_patch(torso)
    ax.add_patch(Circle((0, 0), 3.2, facecolor="#f4f4f4", edgecolor="#999999", linewidth=1.0, zorder=2))
    ax.scatter([0], [0], marker="x", s=35, color="#777777", linewidth=1.5, zorder=3, label="Torso (C7 at 0, 0)")
    ax.add_patch(Rectangle((right.min()-1.2, forward.min()-1.7), right.max()-right.min()+2.4,
                           forward.max()-forward.min()+3.4, facecolor="#b8c4f4", edgecolor="#7184d7",
                           alpha=.32, linewidth=1.2, zorder=1))
    ax.scatter(xx, yy, s=23, color="#4d62cc", alpha=.85, zorder=3, label="Wang-grid targets")
    ax.scatter([start[0]], [start[1]], marker="o", s=90, facecolor="none", edgecolor="#ff6b61",
               linewidth=2.0, zorder=5, label="Achieved common reach start")
    ax.set_xlim(-30, 30); ax.set_ylim(-6, 50)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Right relative to C7 (cm)"); ax.set_ylabel("Forward relative to C7 (cm)")
    ax.set_title("Body context", weight="bold")
    ax.grid(alpha=.16); ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), frameon=True, ncol=1)

    fig.suptitle("Lower Wang grid: top view", fontsize=20, weight="bold")
    fig.tight_layout(rect=(0, .13, 1, .94))
    fig.text(.5, .02, "Grid and common start are on the −23.2 cm height plane.",
             ha="center", fontsize=11, color="#4c5159")
    fig.savefig(OUT / "04_lower_wang_grid_start_schematic.png", dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_poster_wang_physical_overlay():
    poster = pd.read_csv(SRC / "poster_bias_investigation" / "endpoint_factor_analysis" / "poster_endpoint_physical_table.csv")
    analytic = pd.read_csv(SRC / "old_centerout_corrected_tmin_comparison" / "multiring_archived_vs_corrected_results.csv")
    analytic = analytic[analytic.condition == "Corrected TMIN only"].groupby("direction",as_index=False)[["true_x","true_y"]].mean()
    joined = poster.merge(analytic,left_on="target",right_on="direction")
    source_xy=joined[["true_x","true_y"]].to_numpy(); achieved_xy=joined[["physical_right_cm","physical_forward_cm"]].to_numpy()
    source_center=np.array([41.353928,-6.801852])
    physical_center=np.array([9.787661514055083,42.51133963040473])
    source_zero=source_xy-source_center; physical_zero=achieved_xy-physical_center
    u,_,vt=np.linalg.svd(source_zero.T@physical_zero); rotation=u@vt
    if np.linalg.det(rotation)<0:
        u[:,-1]*=-1;rotation=u@vt
    intended_physical=source_zero@rotation+physical_center
    right = (np.arange(20) - 9.5) * 2.438
    forward = np.array([22.671, 28.063, 33.455, 38.847, 44.239])
    xx, yy = np.meshgrid(right, forward)
    fig, ax = plt.subplots(figsize=(7.2, 7.5))
    torso = FancyBboxPatch((-24, -4), 48, 7, boxstyle="round,pad=0.35,rounding_size=4",
                           facecolor="#d9d9d9", edgecolor="#888888", linewidth=1.2, zorder=1)
    ax.add_patch(torso); ax.add_patch(Circle((0, 0), 3.2, facecolor="#f4f4f4", edgecolor="#999", zorder=2))
    ax.scatter([0], [0], marker="x", s=35, color="#777", linewidth=1.5, zorder=3, label="Torso (C7 at 0, 0)")
    ax.add_patch(Rectangle((right.min()-1.2, forward.min()-1.7), right.max()-right.min()+2.4,
                           forward.max()-forward.min()+3.4, facecolor="#90a4e8", edgecolor="#5269c7",
                           alpha=.20, linewidth=1.2, zorder=1))
    ax.scatter(xx, yy, s=18, color="#5269c7", alpha=.75, label="Wang grid targets", zorder=3)
    ax.scatter(intended_physical[:,0],intended_physical[:,1],s=28,color="#9b159b",
               alpha=.82,label="Intended multiring targets",zorder=4)
    ax.scatter([physical_center[0]],[physical_center[1]],marker="o",s=90,facecolor="none",edgecolor="#ff6b61",
               linewidth=2.0,label="Archived common start",zorder=5)
    ax.set_xlim(-30, 30); ax.set_ylim(-6, 62); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Right relative to C7 (cm)"); ax.set_ylabel("Forward relative to C7 (cm)")
    ax.set_title("Physical workspace alignment", weight="bold"); ax.grid(alpha=.16)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.11), frameon=True)
    fig.suptitle("Original multiring targets relative to the Wang grid", fontsize=19, weight="bold")
    fig.tight_layout(rect=(0, .12, 1, .94))
    fig.text(.5, .015, "Nominal radii: 4, 8, 10, and 12 cm. Geometry is rigidly registered at the measured archived common-start Handle position.",
             ha="center", fontsize=10.5, color="#4c5159")
    fig.savefig(OUT / "05_multiring_vs_wang_physical_workspace.png", dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_poster_full_vector_bias():
    results=pd.read_csv(SRC/"old_centerout_corrected_tmin_comparison"/"multiring_archived_vs_corrected_results.csv")
    results=results[results.condition=="Corrected TMIN only"].copy()
    poster=pd.read_csv(SRC/"poster_bias_investigation"/"endpoint_factor_analysis"/"poster_endpoint_physical_table.csv")
    analytic=results.groupby("direction",as_index=False)[["true_x","true_y"]].mean()
    joined=poster.merge(analytic,left_on="target",right_on="direction")
    source_center=np.array([41.353928,-6.801852]);physical_center=np.array([9.787661514055083,42.51133963040473])
    source_xy=joined[["true_x","true_y"]].to_numpy();achieved_xy=joined[["physical_right_cm","physical_forward_cm"]].to_numpy()
    u,_,vt=np.linalg.svd((source_xy-source_center).T@(achieved_xy-physical_center));rotation=u@vt
    if np.linalg.det(rotation)<0:
        u[:,-1]*=-1;rotation=u@vt
    registered=(source_xy-source_center)@rotation+physical_center
    positions=pd.DataFrame({"direction":joined.direction,"right":registered[:,0],"forward":registered[:,1]})
    means=results.groupby("direction",as_index=False).agg(right_bias=("body_right_cm","mean"),forward_bias=("body_forward_cm","mean"))
    means=positions.merge(means,on="direction")

    fig,ax=plt.subplots(figsize=(9.4,7.1))
    x=means.right.to_numpy();y=means.forward.to_numpy();dx=means.right_bias.to_numpy();dy=means.forward_bias.to_numpy()
    ax.scatter(x,y,s=25,color="#9b159b",zorder=4,label="Intended target location")
    ax.quiver(x,y,dx,dy,angles="xy",scale_units="xy",scale=1,width=.0065,headwidth=5.2,headlength=6.2,
              headaxislength=5.4,color="#9b159b",zorder=5,label="Decoded error vector")
    all_x=np.r_[x,x+dx];all_y=np.r_[y,y+dy]
    ax.set_xlim(all_x.min()-2,all_x.max()+2);ax.set_ylim(all_y.min()-2,all_y.max()+2)
    ax.set_aspect("equal",adjustable="box");ax.grid(alpha=.16)
    ax.set_xlabel("Right relative to C7 (cm)");ax.set_ylabel("Forward relative to C7 (cm)")
    ax.set_title(f"Original poster workspace\nmean vector ({results.body_right_cm.mean():+.2f} right, {results.body_forward_cm.mean():+.2f} forward) cm",weight="bold")
    ax.axvline(0,color="#888",lw=.7,alpha=.6);ax.axhline(0,color="#888",lw=.7,alpha=.6)
    fig.tight_layout()
    fig.savefig(OUT/"06_original_multiring_full_vector_bias.png",dpi=250,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def save_multiring_height_full_vector_comparison():
    base = SRC / "poster_bias_investigation" / "multiring_height_pair_static"
    models = pd.read_csv(base / "model_target_results.csv")
    means = models.groupby(["target", "condition"], as_index=False).agg(
        right=("physical_right_cm", "mean"),
        forward=("physical_forward_cm", "mean"),
        right_bias=("outward_cm", "mean"),
        forward_bias=("forward_cm", "mean"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4), sharex=True, sharey=True,
                             constrained_layout=True)
    panels = [
        ("low", "Lower plane: -23.2 cm", "#2a6fbb"),
        ("high", "Higher plane: +4.0 cm", "#b43b2b"),
    ]
    all_x, all_y = [], []
    for condition, _, _ in panels:
        data = means[means.condition == condition]
        all_x.extend([data.right.to_numpy(), (data.right + data.right_bias).to_numpy()])
        all_y.extend([data.forward.to_numpy(), (data.forward + data.forward_bias).to_numpy()])
    limits = (min(v.min() for v in all_x)-1.5, max(v.max() for v in all_x)+1.5,
              min(v.min() for v in all_y)-1.5, max(v.max() for v in all_y)+1.5)
    for ax, (condition, title, color) in zip(axes, panels):
        data = means[means.condition == condition]
        x = data.right.to_numpy(); y = data.forward.to_numpy()
        dx = data.right_bias.to_numpy(); dy = data.forward_bias.to_numpy()
        ax.scatter(x, y, s=22, color=color, zorder=3)
        ax.quiver(x, y, dx, dy, angles="xy", scale_units="xy", scale=1,
                  width=.005, headwidth=4.8, headlength=5.8, color=color, zorder=4)
        ax.set_title(f"{title}\nmean vector: ({dx.mean():+.2f} right, {dy.mean():+.2f} forward) cm",
                     weight="bold", fontsize=14)
        ax.set_xlabel("Physical right relative to C7 (cm)", fontsize=12)
        ax.grid(alpha=.2); ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(limits[0], limits[1]); ax.set_ylim(limits[2], limits[3])
    axes[0].set_ylabel("Physical forward relative to C7 (cm)", fontsize=12)
    fig.suptitle("Static multiring endpoints: isolated height manipulation",
                 fontsize=19, weight="bold")
    fig.savefig(OUT / "04_multiring_height_full_vector_comparison.png", dpi=250,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    save_workspace_height()
    save_height_full_vector_comparison()
    save_height_full_vector_comparison_with_models()
    save_height_heatmap_comparison()
    save_multiring_height_full_vector_comparison()
    save_poster_wang_physical_overlay()
    print(f"Wrote figures to {OUT}")
