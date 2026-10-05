# MuJoCo muscle-geometry and spindle-adapter audit

Status: implementation and validation record, October 5, 2026. The retained routing changes are available through the `myosuite_corrected` backend; rejected and superseded experiments remain documented for traceability.

## Goal

Replace the OpenSim runtime with a MuJoCo model without retraining the existing CNN. The CNN should continue to receive firing rates produced by the existing Mathis spindle layer. MuJoCo actuator length is treated as musculotendon length, not fiber length.

The current adapter uses the MoBL muscle parameters

```text
h = L0 * sin(alpha0)
Lf = sqrt(max(LMT - LTS, 0)^2 + h^2)
```

where `LMT` is MuJoCo actuator length, `LTS` is tendon slack length, `L0` is optimal fiber length, and `alpha0` is pennation at optimal fiber length. The reconstructed `Lf` is differentiated and normalized exactly as in the existing pipeline before applying the seed-0 Mathis coefficients.

This adapter corrects the musculotendon-versus-fiber interpretation. It cannot correct an incorrect posture-to-musculotendon-length function caused by attachment, via-point, wrapping, or joint-definition differences.

## Current evaluation protocol

- Six smooth multijoint trajectories at 240 Hz, 480 samples each.
- Three trajectories used to select a MyoArm/MS-Human-700 source per muscle.
- Three different trajectories held out for reporting.
- Five Ia and five II channels per muscle using the repository's seed-0 coefficients.
- No per-muscle affine length fitting.
- Preliminary synthetic-motion audit; real pipeline trajectories remain required before final acceptance.

The numerical audit used the current composed `myo_sim` `myoarm_r` model cloned for analysis. The branch loader currently targets the packaged MyoSuite `myoarm.xml` (documented as `myosuite==2.11.6`). Before editing branch model assets, reproduce these metrics with the exact pinned artifact and record its commit/package hash; routing conclusions should not be transferred blindly between model revisions.

Across the 22 directly compared non-LAT muscles, MyoArm plus the adapter achieved median spindle-rate RMSE 3.10 Hz and median correlation 0.971. The hard MyoArm/MS hybrid achieved 1.67 Hz and 0.982, respectively.

## MyoArm classification

The categories are based primarily on held-out spindle-rate behavior, with routing inspection used to interpret failures. `Good` means no current evidence that routing must be edited; it does not mean anatomically exact. `Review` means trajectory-sensitive evidence. `Adjust` means the adapter alone is insufficient.

### Good / retain initially

`CORB`, `DELT3`, `INFSP`, `PECM2`, `PECM3`, `SUBSC`, `SUPSP`, `ANC`, `BIClong`, `BICshort`, `BRA`, `BRD`, `ECRL`, `PT`, `TRIlat`, `TRImed`.

These should be frozen while the failing paths are corrected, then regression-tested to ensure shared wrap changes do not disturb them.

### Review before changing

- `DELT1`: results were trajectory-sensitive. It was poor in an early single-trajectory test but did not remain among the dominant errors in the held-out multitrajectory audit. Expand the posture sweep before deciding whether to edit it.

### Adjust

| Muscle | Held-out evidence | Main geometry discrepancy found |
|---|---:|---|
| `DELT2` | 9.54 Hz RMSE, correlation 0.490 | MoBL uses three wraps (`Deltoid2`, `delt2hum`, and `DELT_TMAJ_LAT_PEC_CORBhh`) plus a moving point. MyoArm currently routes through the first two wraps only. |
| `TMIN` | 31.08 Hz, 0.048 | MoBL declares three active wraps (`TMIN`, `INFSP_and_TMIN_hum_head`, `TMINhum`). MyoArm's tendon has no active wrap; the humeral-head wrap is commented out. |
| `TMAJ` | 11.83 Hz, 0.637 | MoBL includes two conditional humeral points and three wraps. MyoArm uses fixed sites and only two of the wrap families. |
| `PECM1` | 6.01 Hz, 0.817 | MoBL contains humeral and ground moving points and the `Thorax`, `PEC12hum`, and `PEC1hh` wraps. MyoArm reduces this to fixed sites and one `PEC1hh` wrap. |
| `TRIlong` | 12.40 Hz, 0.573 | MoBL uses `TRI`, `TRIlonghh`, and `TRIlongglen`. MyoArm uses the first two, omits `TRIlongglen`, and inserts a fixed `P1b` site. |
| `LAT1` | 12.02 Hz, 0.346 | MoBL uses conditional humeral points, moving scapular points, and three wrap families. MyoArm replaces the moving/conditional behavior with fixed sites. |
| `LAT2` | 12.69 Hz, 0.549 | Same structural issue as LAT1; the current second wrap also has no explicit side site. |
| `LAT3` | 16.49 Hz, 0.429 | Same structural issue as LAT1/2; fixed-site approximation does not preserve posture dependence. |

## Initial routing plan (historical)

The objective is to match both `LMT(q)` and its joint derivatives (moment arms), not merely length at one neutral posture. Every change should be optimized on a posture grid and evaluated on separate continuous trajectories.

## Fixed muscle groups

### `TRIlong` — provisional pass

An experimental routing copy replaced the converted synthetic `TRIlonghh`/fixed `P1b` detour with the MoBL-conversion `TRIlongglen` cylinder and side site. The elbow `TRI` cylinder and remaining sites were retained.

| Metric | Before | After |
|---|---:|---:|
| Held-out spindle-rate RMSE | 12.40 Hz | 1.24 Hz |
| Held-out correlation | 0.573 | 0.990 |
| Maximum frame-to-frame change in normalized velocity on the audit trajectories | not recorded in the original baseline run | 0.036 |

This passes the provisional 5 Hz / 0.95 criterion and showed no large transition spike in the tested trajectories. It remains provisional until reproduced using the exact branch-pinned MyoSuite artifact and real center-out movements.

## Attempted but not fixed

| Muscle/change | Before | Experimental result | Decision |
|---|---:|---:|---|
| `TMIN`: restore `TMINhum` cylinder with generic side site | 31.08 Hz, 0.048 | 30.43 Hz, 0.179 | Improved slightly; still bad. |
| `TMIN`: use MoBL-conversion-specific side site | 31.08 Hz, 0.048 | 27.75 Hz, 0.543 | Meaningful improvement but still fails; retain only as a candidate for the next iteration. |
| `DELT2`: replace current two-wrap approximation with converted common humeral-head sphere while leaving its point approximation fixed | 9.54 Hz, 0.490 | 21.16 Hz, 0.252; large velocity jump | Reverted. The moving point and wrap must be reconstructed together. |
| `TMAJ`: substitute converted `LAT_TMAJhh` primitive without restoring conditional points | 11.83 Hz, 0.637 | 11.52 Hz, 0.637 | No material benefit; reverted. |
| `PECM1`: substitute `PEC12hum` for current `PEC1hh` without restoring moving points | 6.01 Hz, 0.817 | 8.40 Hz, 0.784; larger velocity jump | Worse; reverted. |

The experiments in this section were originally made in a temporary model clone. The later retained subset is now reproduced at runtime by the tracked `myosuite_corrected` backend.

## Differentiability of wraps

MuJoCo spatial-tendon wrapping is differentiable within a fixed contact/path regime, so moment arms and local derivatives remain usable. It is generally only piecewise smooth globally. Non-smooth behavior can occur when a tendon engages or disengages a wrap, changes sides, or switches topology. Conditional/moving path points can introduce similar transition issues. Consequently, every routing change must be checked for jumps in length, velocity, acceleration, and moment arms over dense trajectories; successful compilation and lower static length error are not sufficient.

### TMIN

1. Restore an active humeral-head wrap corresponding to MoBL `INFSP_and_TMIN_hum_head` rather than leaving it commented out.
2. Determine whether separate MuJoCo primitives are needed for MoBL `TMIN` and `TMINhum`; do not assume one ellipsoid reproduces all three OpenSim wraps.
3. Tune wrap pose/size and side site using length and shoulder moment-arm error.
4. Keep anatomical endpoints fixed initially; move them only if wrapping correction cannot reproduce the reference.

### DELT2

1. Preserve the existing `Deltoid2` and `delt2hum` primitives.
2. Add or activate the omitted common humeral-head wrap corresponding to `DELT_TMAJ_LAT_PEC_CORBhh`.
3. Reproduce the OpenSim moving path point as a joint-dependent site or polynomially coupled slide joints.
4. Validate across elevation plane, elevation, and rotation; DELT2 should not be calibrated on a one-dimensional shoulder sweep.

### TMAJ

1. Replace the fixed approximation near the humerus with conditional sites that activate over the same coordinate intervals as MoBL's two conditional points.
2. Add the missing `LAT_TMAJhh` wrap and verify the existing `TMAJ_LAThum` and `LAT_TMAJ2hh` side-site choices.
3. Optimize the conditional transition for continuity; reject any solution with path switching spikes in velocity or spindle rate.

### PECM1

1. Restore the missing thorax and `PEC12hum` wrapping effects before changing endpoints.
2. Encode the MoBL moving humeral and ground points as polynomial/joint-dependent MuJoCo sites.
3. Check whether the ground point should instead be expressed in the corresponding MyoArm torso/clavicle frame to avoid a coordinate-frame mismatch.
4. Fit geometry against shoulder moment arms as well as total path length.

### TRIlong

1. Add the omitted `TRIlongglen` wrap near the scapular origin.
2. Reassess the synthetic fixed `TRIlong-P1b` point; retain it only if it is needed to reproduce OpenSim's `TRIlonghh`/glenoid wrap sequence.
3. Tune shoulder-side geometry first, then verify the existing elbow `TRI` cylinder independently.

### LAT1–3

1. Treat the three MoBL LAT paths separately; do not assume a shared affine correction.
2. Convert each pair of conditional humeral points and moving scapular points into joint-dependent MuJoCo sites.
3. Correct the wrap sequence to represent `TMAJ_LAThum`, `LAT_TMAJhh`, and `LAT_TMAJ2hh`. Audit the current references to `delt2hum_cylinder`; their side-site names indicate they may be conversion substitutions rather than faithful LAT wraps.
4. Preserve the distinct ground/torso endpoints for LAT1, LAT2, and LAT3.
5. Evaluate continuity around conditional-point activation boundaries and test all three shoulder coordinates jointly.

## Calibration and acceptance procedure

For each edited muscle:

1. Sample training postures over the valid shoulder/elbow range.
2. Minimize normalized error in musculotendon length and moment arms, with regularization toward the current anatomical geometry.
3. Use bounds that keep sites on their intended body and wraps near their anatomical surfaces.
4. Do not optimize Mathis coefficients or CNN weights.
5. Freeze geometry and evaluate on held-out postures and continuous reaches.
6. Pass the resulting MuJoCo length through the fixed fiber adapter.
7. Report per-muscle Ia/II RMSE, correlation, maximum transient error, and path-continuity failures.

A provisional acceptance target is median channel correlation at least 0.95 and spindle-rate RMSE below 5 Hz per muscle, with no large derivative spikes. This threshold is an engineering criterion for this project, not a published physiological standard.

## MS-Human-700 LAT mapping correction

The base `MS-Human-700.xml` does not expose LAT actuators, although some LAT sites are present. The `myosuite` branch intentionally loads `MS-Human-700-Manipulation.xml`, where LAT is represented by different bundles. Its current approximate mapping is:

```text
MoBL LAT1 -> LD_L1_r
MoBL LAT2 -> LD_L2_r
MoBL LAT3 -> LD_L3_r
```

These are not exact muscle correspondences. Any MS-Human-700 LAT comparison must use the manipulation model and must remain labeled approximate. Previous direct-base-model LAT conclusions must not be applied to this branch.

## Open questions

- Re-run the complete 25-muscle held-out audit using the branch's manipulation-model LAT mapping.
- Evaluate actual center-out trajectories rather than synthetic movements alone.
- Decide whether corrected MyoArm routes remove the need for a hybrid or whether a small set of MS-Human-700 routes remains preferable.
- Determine whether tendon compliance is needed after geometry correction; the present adapter assumes rigid tendon.

## Deeper diagnosis of the remaining group

An isolated-coordinate sweep exposed two distinct effects that the original broad-trajectory score conflated: genuine MyoArm geometry errors and discontinuous OpenSim reference paths at extreme postures. The figures below describe musculotendon length and numerical moment-arm behavior before the fiber adapter.

### `DELT2`: genuine directional routing mismatch

- Across elevation-plane angle, MyoArm and MoBL length correlation was `-0.990`; moment-arm correlation was `-0.998`.
- Across shoulder rotation, length correlation was `-0.464` and moment-arm correlation `-0.950`.
- Across shoulder elevation, length correlation was `0.972`, but moment-arm correlation was `-0.371` and MyoArm excursion was only 61% of MoBL.

This is not an offset or tendon-slack problem. The current fixed approximation of MoBL's shoulder-rotation-dependent moving point gives the wrong directional sensitivity in two shoulder coordinates. A direct sphere insertion without restoring that moving point worsened RMSE and introduced a wrap transition. The moving point must be implemented first or jointly with the wrap.

### `TMIN`: wrap topology plus a problematic OpenSim reference transition

- Restoring the converted cylinder and side site improved spindle correlation from `0.048` to `0.543`, confirming that missing wrapping is part of the error.
- Elevation-plane moment-arm correlation after that change was `0.972`, but shoulder-elevation correlation was `-0.333`.
- During a full shoulder-rotation sweep, MoBL jumped approximately 210 mm near 65 degrees while MyoArm remained smooth. MoBL also jumped about 15 mm near 161 degrees elevation.

TMIN has no moving or conditional path points in MoBL, so its residual is not caused by a frozen point. It is caused by different wrap engagement/topology and by extreme-posture discontinuities in the OpenSim path solution. Do not tune MyoArm to reproduce the 210 mm jump. Re-evaluate it inside the actual center-out envelope, then tune the cylinder/ellipsoid only on the continuous branch used by the experiment.

### `TMAJ`: missing two-coordinate conditional activation

- Length behavior was excellent for elevation plane (`0.997`) and shoulder elevation (`0.999`).
- Shoulder-rotation length correlation fell to `0.340`.
- MoBL conditionally inserts the same humeral point when shoulder rotation is 15–120 degrees and when elevation-plane angle is approximately -95 to -10 degrees. MyoArm's fixed path cannot reproduce this two-coordinate on/off behavior.

The primary correction is conditional point activation, not another isolated wrap substitution. The transition should be smoothed or hysteretic if a hard OpenSim-style insertion causes derivative spikes.

### `PECM1`: correct general direction, wrong elevation gain

- Correlation remained positive for elevation plane (`0.975`) and shoulder rotation (`0.975`).
- MyoArm excursion was only 47% of MoBL across elevation plane, but 236% across shoulder elevation and 133% across rotation.
- MoBL moves both a humeral point and a ground-frame point as explicit functions of `shoulder_elv`; MyoArm freezes their approximations.

This explains why swapping a wrap alone made the result worse. Implement the two linear shoulder-elevation point functions first, preserving their parent frames, then retune `PEC12hum`. The evidence does not currently justify moving the anatomical endpoints.

### `LAT1`: mostly aligned length, incorrect elevation/rotation derivatives

- Length correlations were `0.993` for elevation plane and `0.997` for shoulder elevation.
- Shoulder-elevation moment-arm correlation was only `0.152`; rotation length correlation was `0.579`.
- MoBL has two conditional humeral points controlled separately by rotation and elevation plane, plus two scapular points that move linearly with shoulder elevation. MyoArm freezes all four effects.
- MoBL showed a roughly 9.8 mm step near 167 degrees elevation, outside the likely center-out envelope.

Implement the moving scapular points first and then conditional humeral activation. The good gross length correlation means origin/insertion changes are not the first choice.

### `LAT2`: likely usable in ordinary elevation, pathological extreme rotation reference

- Elevation-plane and shoulder-elevation correlations were `0.997` and `0.999`; moment-arm correlations were `0.986` and `0.992`.
- MoBL produced an approximately 451 mm path jump near 107 degrees shoulder rotation; MyoArm remained smooth.
- The full-range rotation correlation (`0.226`) and the earlier spindle score are therefore dominated by a reference topology switch, not ordinary excursion.

LAT2 must be reclassified using the actual task range. If center-out motion never approaches the transition, this muscle may not require anatomical editing beyond the adapter. It should not be modified to reproduce the 451 mm jump.

### `LAT3`: missing rotation-conditioned path behavior

- Elevation-plane correlation was `0.970`; shoulder-elevation correlation was `0.999` with nearly identical excursion amplitude.
- Shoulder-rotation correlation was `-0.179`, despite reasonable behavior in the other coordinates.
- Its MoBL conditional humeral points activate at different rotation/elevation-plane thresholds than LAT1/2, while its two scapular points have their own shoulder-elevation functions. The shared fixed MyoArm approximation loses this bundle-specific rotation dependence.

Focus on the rotation-conditioned humeral point before altering the already-good shoulder-elevation geometry.

### Consequence for the audit protocol

The broad synthetic trajectories were useful for finding failures but are not a fair final score when they cross OpenSim wrap/path discontinuities outside the task's operating range. Before making further geometry changes:

1. extract the joint envelope of the real training and center-out trajectories;
2. identify which OpenSim path branch those data occupy;
3. score length, moment arms, and spindle rates only inside that envelope, plus a small safety margin;
4. separately report out-of-envelope topology transitions rather than allowing them to dominate RMSE.

## Paper OpenSim-reference verification

The published code for *Deep-learning models of the ascending proprioceptive pathway are subject to illusions* points to `MOBL_ARMS_41_seb_writing_pos.osim`, whereas this repository has used `DefaultMOBL_ARMS_fixed_41.osim`. They are not byte-identical:

```text
paper model SHA-256: A2B5544682E0DDB2A2CF74E7AF2CF24F6A8FC2B0F5EBE7AF0234AE712C60E05F
local model SHA-256: 5224BED1F1C315727745969B43D7507141193ABB470C0703DCE85C980C87F449
```

They nevertheless have identical `L0`, tendon-slack-length, and optimal-pennation arrays for all 25 pipeline muscles and are functionally equivalent for the paper task. A six-trajectory comparison used the paper's reported arm envelope: elevation plane 19–79 degrees, shoulder elevation 39–99 degrees, shoulder rotation -6–54 degrees, and elbow flexion 45–130 degrees.

| Comparison | Median spindle RMSE | Median correlation | 90th-percentile RMSE |
|---|---:|---:|---:|
| Paper OpenSim vs local OpenSim | approximately 0 Hz | approximately 1.000 | 0.139 Hz |
| MyoArm + adapter vs local OpenSim | 1.228 Hz | 0.989 | 3.272 Hz |
| MyoArm + adapter vs paper OpenSim | 1.041 Hz | 0.989 | 3.243 Hz |

Twenty-five-muscle paper-vs-local differences were exactly/nearly zero except for `BIClong` (0.182 Hz), `BICshort` (0.227 Hz), `BRD` (0.029 Hz), `ECRL` (0.086 Hz), and `PT` (0.174 Hz); all corresponding correlations were at least 0.9996. Thus, changing to the paper model produces no meaningful reduction in MyoArm error and does not explain the residual geometry mismatches.

## Paper-baseline routing fixes

The paper OpenSim model is the reference for these routing experiments. The retained edits were first tested in a temporary `myo_sim` clone and are now reproduced by the branch's `myosuite_corrected` backend. A supplemental six-trajectory test stayed inside the published joint envelope, used 480 samples at 240 Hz, and applied the unchanged fiber adapter and seed-0 Mathis layer. Its trajectories are independent of the earlier table, so compare its before/after values within this section.

| Muscle | Temporary structural change | Before | After | Continuity result |
|---|---|---:|---:|---:|
| `DELT2` | Replaced the static two-wrap approximation with the paper model's shoulder-rotation-dependent point and common humeral-head sphere route. | 4.70 Hz, 0.743 on the original paper-envelope audit | 2.16 Hz, 0.962 on the supplemental audit | Maximum within-trajectory one-frame rate change 1.91 Hz |
| `PECM1` | Added the paper model's elevation-dependent humeral and torso-frame points, restored the `PEC12hum` cylinder, and used the actual terminal clavicle attachment rather than the legacy intermediate site. | 4.39 Hz, 0.634 on the matched supplemental baseline | 1.69 Hz, 0.932 | Maximum within-trajectory one-frame rate change 0.56 Hz |
| `BIClong` | Restored the second and third sequential contacts around the existing humeral-head sphere; endpoints and the elbow route were unchanged. | 2.78 Hz, 0.924 on the original paper-envelope audit | 1.75 Hz, 0.964 on the supplemental audit | Maximum within-trajectory one-frame rate change 1.26 Hz |
| `BICshort` | Removed MyoArm's substituted elbow wrap and restored the paper model's pronation-dependent distal point using three polynomially coupled slide joints. | 3.29 Hz, 0.948 on the original paper-envelope audit | 0.91 Hz, 0.991 on the supplemental audit | Maximum within-trajectory one-frame rate change 0.64 Hz |
| `SUPSP` | Restored the paper-model scapular attachment and wrap pose/size while retaining MyoArm's stable side-site to avoid the converted branch switch. | 3.02 Hz, 0.816 on the matched supplemental baseline | 1.43 Hz, 0.925 | Maximum within-trajectory one-frame rate change 0.32 Hz |

`DELT2` now passes the provisional RMSE/correlation/continuity criteria. `PECM1` has a large and smooth improvement and passes the RMSE criterion, but its correlation remains below 0.95, so it is provisionally improved rather than final.

Three literal converter outputs were rejected:

- The converted `PECM1` cylinder side site at the wrap center produced an 11.24 Hz RMSE, 0.450 correlation, and a 161.86 Hz one-frame jump. The stable existing MyoArm cylinder side site was retained.
- Replacing the current `SUPSP` route geometry with the literal converted MoBL point, sphere, and side site produced an 11.57 Hz RMSE, 0.413 correlation, and a 162.22 Hz one-frame jump. The current smooth MyoArm `SUPSP` geometry was restored.
- Replacing the current `TMAJ` approximation with the converter's continuously active three-cylinder/two-moving-point route produced 4.23 Hz RMSE and 0.720 correlation. It was reverted. The OpenSim points are conditional, so a faithful future implementation must reproduce their activation intervals rather than leave both active continuously.

Removing the restored `PEC12hum` cylinder while retaining the two moving points and corrected endpoint left the reported `PECM1` result numerically unchanged. The cylinder does not engage on these supplemental trajectories. The measured improvement therefore comes from the coordinate-dependent points and correct terminal attachment; the stable cylinder route is retained for structural completeness and for postures where it may engage.

Removing the restored `TMINhum` cylinder changed the supplemental score only from 10.32 Hz/0.224 to 10.37 Hz/0.225. This confirms that the residual is not repairable by simply toggling that wrap. The structurally correct cylinder was retained, but no additional TMIN change is recommended without real-trajectory evidence and a branch-aware wrap formulation. The much lower 1.71 Hz result on the original paper-envelope trajectory set also shows that TMIN is unusually trajectory/branch sensitive.

These failures are wrap-branch/topology failures in MuJoCo, not evidence that the paper OpenSim geometry is wrong. A converted side selector can place MuJoCo on a different wrap branch even when the source point coordinates are exact.

## Current disposition after routing experiments

- **Retain as provisional fixes:** `TRIlong`, `DELT2`, `PECM1`, `BIClong`, `BICshort`, and the branch-stable partial `SUPSP` reconstruction.
- **Retain existing MyoArm routing:** `TMAJ` and `TMIN`. Literal converted alternatives were worse or reproduced undesirable topology changes. Their original task-envelope RMSE remains below 5 Hz.
- **Do not presently edit LAT1-3:** their task-envelope RMSE is only 1.23-2.01 Hz. Their known conditional-point differences matter mainly in rotation/extreme postures and should not be changed until real task trajectories demonstrate a failure.
- **No adapter or CNN retuning was used:** all improvements came from correcting the MuJoCo posture-to-musculotendon-length function before the fixed fiber adapter and Mathis spindle layer.

## Unified final-geometry validation

The retained geometry was evaluated for all 25 muscles in one run over six 480-sample trajectories at 240 Hz within the paper's stated joint envelope. The reference was the paper OpenSim model, and both paths used the same MoBL fiber adapter and fixed seed-0 Mathis coefficients.

Across every sample, including OpenSim wrap/path switches, aggregate RMSE was **3.536 Hz**, aggregate correlation was **0.816**, median per-muscle RMSE was **1.754 Hz**, and median per-muscle correlation was **0.962**. These raw aggregate statistics are strongly affected by discontinuities present only in the OpenSim reference: TMIN, LAT1-3, INFSP, TMAJ, and DELT3 showed OpenSim one-frame changes of 134-171 Hz while the corresponding MuJoCo maxima were 0.07-0.47 Hz.

A separately reported continuous-branch score excluded a six-frame neighborhood around any OpenSim reference transition exceeding 10 Hz in one frame. It did not exclude based on MuJoCo error. On the continuous branches, aggregate RMSE was **1.633 Hz**, aggregate correlation was **0.951**, median per-muscle RMSE was **1.302 Hz**, and median per-muscle correlation was **0.963**.

| Muscle | Stable RMSE (Hz) | Stable correlation | OpenSim time excluded |
|---|---:|---:|---:|
| BRD | 0.031 | 1.000 | 0.0% |
| ANC | 0.068 | 1.000 | 0.0% |
| ECRL | 0.088 | 1.000 | 0.0% |
| BRA | 0.100 | 1.000 | 0.0% |
| PT | 0.231 | 1.000 | 0.0% |
| CORB | 0.539 | 0.991 | 0.0% |
| BICshort | 0.910 | 0.991 | 0.0% |
| DELT1 | 1.251 | 0.984 | 0.0% |
| TRIlat | 1.140 | 0.983 | 0.0% |
| TRImed | 0.886 | 0.979 | 0.0% |
| TRIlong | 1.602 | 0.973 | 0.7% |
| BIClong | 1.754 | 0.964 | 0.0% |
| DELT3 | 1.273 | 0.963 | 2.7% |
| DELT2 | 2.162 | 0.962 | 0.0% |
| LAT3 | 1.302 | 0.951 | 1.4% |
| LAT2 | 1.389 | 0.940 | 1.4% |
| PECM1 | 1.688 | 0.932 | 0.0% |
| SUPSP | 1.434 | 0.925 | 0.0% |
| PECM2 | 2.955 | 0.903 | 0.0% |
| TMAJ | 3.650 | 0.863 | 1.5% |
| LAT1 | 1.061 | 0.853 | 2.3% |
| PECM3 | 1.898 | 0.814 | 2.3% |
| SUBSC | 2.135 | 0.781 | 0.0% |
| INFSP | 2.694 | 0.746 | 5.4% |
| TMIN | 1.783 | 0.718 | 14.4% |

No MuJoCo muscle showed a large transition spike; the largest MuJoCo one-frame rate change was 1.91 Hz (`DELT2`). The remaining low correlations generally combine small signal excursion with residual wrap/posture differences. No remaining route presented an obvious correction that improved alignment without either changing anatomical meaning or selecting an unstable MuJoCo wrap branch.

This synthetic unified validation was sufficient to freeze geometry for the selectable backend. The backend also completed an eight-direction center-out pipeline smoke test with finite spindle rates and mean endpoint IK errors of 0.003-0.007 cm. Exact recorded center-out/letter trajectory regression remains recommended before treating the replacement as fully equivalent for every CNN input distribution.

The task-envelope test also gives a fairer MyoArm result than the earlier broad synthetic sweep. Before the retained corrections, the main task-envelope outliers included `PECM1` (4.21 Hz, correlation 0.520) and `DELT2` (4.70 Hz, 0.743). The unified final-geometry table above supersedes those earlier values. Several muscles flagged under broad/extreme sweeps were already modest inside the paper envelope, reinforcing the need to judge routing changes on the intended task distribution.

## Tracked backend implementation

The branch does not vendor a second copy of the approximately 40 MB MyoArm assets. Instead, `mujoco_pipeline/corrected_myoarm.py` composes the right-arm model from pinned `myo-sim==0.2.3`, applies strict source-anchored XML transformations for the retained routing changes, and compiles the resulting MuJoCo model. A missing or changed anchor fails loudly rather than silently applying a correction to an incompatible model revision.

Run the corrected model and adapter with:

```powershell
python -m mujoco_pipeline.pipeline --backend myosuite_corrected
```

The same backend is available in `mujoco_pipeline.generate_training_set`. The legacy `myosuite` and `ms_human_700` modes remain unchanged for comparison.
