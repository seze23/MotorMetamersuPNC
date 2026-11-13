"""Plot original versus height-shifted Wang endpoint bias over the grid."""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"dataexp/poster_bias_investigation/wang_vertical_shift_full"


def grid(data,column):
    result=np.full((5,20),np.nan)
    for _,row in data.iterrows():result[int(row.target[1:3]),int(row.target[5:7])]=row[column]
    return result


def main():
    data=pd.read_csv(OUT/"target_summary.csv");original=grid(data,"original_outward_cm");shifted=grid(data,"shifted_outward_cm");delta=grid(data,"outward_shift_cm")
    fig,axes=plt.subplots(3,1,figsize=(11,6.8),constrained_layout=True)
    panels=[(original,"Original Wang plane (up = −23.2 cm)","coolwarm",-3,3),(shifted,"Height-shifted plane (up = +4 cm)","coolwarm",-3,3),(delta,"Shifted − original","PiYG",-4,4)]
    for ax,(values,title,cmap,vmin,vmax) in zip(axes,panels):
        image=ax.imshow(values,aspect="auto",cmap=cmap,vmin=vmin,vmax=vmax,origin="upper")
        ax.set_title(title);ax.set_ylabel("Grid row");ax.set_yticks(range(5));ax.set_xticks(range(20));ax.set_xticklabels(range(20),fontsize=7)
        fig.colorbar(image,ax=ax,label="Outward component (cm)" if title!="Shifted − original" else "Outward shift (cm)",pad=.015)
        for row,column in np.argwhere(np.isnan(values)):ax.text(column,row,"×",ha="center",va="center",color="black",fontweight="bold")
    axes[-1].set_xlabel("Grid column");fig.suptitle("Wang grid: isolated endpoint-height manipulation (20 CNNs)",fontweight="bold")
    for suffix in ("png","svg"):fig.savefig(OUT/f"wang_vertical_shift_comparison.{suffix}",dpi=220,bbox_inches="tight")
    shared=data[data.shared_in_distribution_abs_z_le_3]
    extra={"shared_original_outward_target_means":int((shared.original_outward_cm>0).sum()),"shared_shifted_outward_target_means":int((shared.shifted_outward_cm>0).sum()),
        "shared_targets":len(shared),"reachable_targets":len(data)}
    (OUT/"spatial_comparison_summary.json").write_text(json.dumps(extra,indent=2))


if __name__=="__main__":main()
