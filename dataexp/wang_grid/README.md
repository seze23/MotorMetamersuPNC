# Full Wang-grid proprioception analysis

This directory contains the complete 100-target by 20-model analysis. Every
model is paired with spindle signals generated from its matching coefficient
seed: five coefficient seeds times four network training seeds.

## Geometry and coordinates

Wang et al. explicitly report a 5-row by 20-column target matrix, with
24.38 mm between adjacent columns and 53.92 mm between adjacent rows, in a
48.76 cm by 26.96 cm workspace whose near edge was 20 cm in front of the
participant. Figure 3 omits numeric axis ticks, so absolute coordinates cannot
be read from the figure itself. The unscaled coordinates stored here are a
centered reconstruction from the reported geometry, not coordinates published
target by target.

The original Wang rectangle did not pass the fixed 0.5 cm endpoint geometry
gate for this right-arm setup. The complete rectangle was therefore uniformly
scaled by 0.25 about `(right=0, forward=33.455)` cm, as required by the
pre-registered scaling rule. All 100 scaled targets were retained. The maximum
requested-versus-achieved endpoint error is 0.453 cm and the mean is 0.205 cm.

Primary errors are decoded-joint OpenSim Handle minus actual OpenSim Handle in
C7-relative anatomical coordinates. Positive lateral error is rightward and
outward for this right arm; positive depth error is forward and away from the
body. The main table keeps the unscaled Wang coordinate alongside the scaled
requested and physically achieved coordinates.

## Late-hold result

Across 2,000 model-target observations, 94.75% of lateral errors are outward.
The mean lateral error is +1.201 cm. Every one of the 100 targets has an
outward 20-model mean; 99 targets have at least 70% of models outward, and 58
targets have all 20 models outward.

All 100 target means are also forward/away, with a mean of +0.560 cm. Across
individual model-target observations, 80.3% are forward. The sign is stable
from reach arrival through 100, 250, and 500 ms and the late endpoint hold.

The lateral error is larger on the left half than the right half (+1.574 versus
+0.829 cm) and larger in the far rows than the near rows (+1.474 versus +0.930
cm). For the mirrored right-arm experiment, this is qualitatively consistent
with larger behavioral errors farther from the modeled shoulder.

## Behavioral and old-model interpretation

Wang et al. plotted physical matching error from visual target to matched hand
location. A left-hand rightward matching displacement corresponds to an
outward proprioceptive estimate; its right-hand homolog is the outward decoded
error used here. The new result therefore agrees strongly in lateral sign with
the homologous behavioral pattern. Its forward decoded component is also
consistent with perceiving the hand farther from the body when the behavioral
matching displacement is toward the body. Magnitudes should not be compared
directly because the model grid is uniformly scaled and models are robustness
replicates rather than human participants.

The old right-arm polar experiment used raw analytic coordinates and plotted
arrows from predicted to true, reversing decoder error. Model 0_9 had mean raw
rightward-axis error -0.679 cm and only 30.6% of targets were rightward. Across
the four valid coefficient-seed-0 old checkpoints, every target's mean raw
rightward-axis error was negative (mean -0.861 cm). Thus the old underlying
decoder result was predominantly inward, even though its reversed poster
arrows visually pointed outward. Old analytic and new physical errors are not
numerically pooled; the comparison is qualitative and sign-based.

## Files

- `model_<coefficient>_<training>_bias.csv`: one row per target and endpoint
  window for each of the 20 matched models.
- `ensemble_target_window_summary.csv`: target-level mean, spread, and sign
  fractions across models.
- `model_late_hold_summary.csv`: one late-hold summary row per model.
- `behavior_old_model_comparison.csv`: concise regional, behavioral-sign, and
  old-model comparison.
- `wang_grid_figure3_style.png` and `.svg`: Figure 3a-style full-grid map.
- `wang_grid_compact_direction_map.png` and `.svg`: direction-focused view on
  an intentionally compressed display grid. Target locations are not spatially
  to scale; one isotropic arrow gain preserves the physical bias direction and
  relative magnitude.
- `wang_grid_paper_convention.png` and `.svg`: body-coordinate map using Wang
  Figure 3's perceived/target-to-physical-hand arrow direction. Both endpoints
  are directly computed: decoded proprioceptive location to IK-achieved hand.
- `wang_grid_paper_convention_compact.png` and `.svg`: the same Wang convention
  on the intentionally compressed display grid.

The expensive stages are cached: IK and muscle lengths are generated once;
spindle signals are generated once per coefficient seed and reused by its four
matched checkpoints; each checkpoint is loaded once for batched inference; and
identical decoded hold poses are forward-kinematically evaluated once.
