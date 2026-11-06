# MyoArm Wang-grid rerun plan

## Purpose

Run the Wang-style localization grid through the corrected MuJoCo/MyoArm
musculoskeletal pipeline, first as a small pilot and then as a complete
100-target by 20-model analysis. The existing full-grid result is retained as
an OpenSim baseline; it is not the requested MyoArm result.

This document is a checklist and stop-gate specification. A later stage must
not begin merely because the preceding script completed. Every required
artifact and acceptance check must pass first.

## Scientific quantities and arrow conventions

For every physical endpoint, retain three separately named locations:

- `requested_hand`: the physical target supplied to IK;
- `actual_hand`: the independently forward-kinematically reconstructed MyoArm
  Handle location after IK; and
- `predicted_hand_xyz`: the network's direct XYZ prediction.

Also retain the network's four predicted proximal joint angles and compute:

- `predicted_hand_joint_fk`: the Handle location obtained by applying the
  predicted joints to a declared musculoskeletal forward-kinematics model.

Never use `requested_hand` as a substitute for `actual_hand`.

Two arrow conventions may be plotted, with the endpoints stated in every
title and legend:

- perceptual-error convention: `actual_hand -> predicted_hand`;
- Wang-style display convention requested for this project:
  `predicted_hand -> actual_hand`.

Reversing an arrow means exchanging these two measured endpoints. Do not keep
one endpoint fixed and construct a reflected or "implied matching" location.

## Network output-head policy

The frozen network predicts XYZ and joint angles simultaneously. These are two
different output heads and are not guaranteed to be mutually consistent.

The direct XYZ head is the most literal answer to "where does the network
predict the hand is?" It will therefore be the primary perceptual-location
readout, after its training-label axes, handedness, units, and origin are
registered to the common anatomical task frame. This registration may use a
declared rigid axis/origin conversion; it may not use a fitted rotation,
reflection, scale, or result-dependent sign change.

The predicted-joint FK location will be a required co-reported readout. It has
the advantage of describing a realizable posture and the same physical Handle
site as the IK result. Before interpreting behavioral agreement, quantify the
distance and directional agreement between the two output heads at every
target. If the two heads disagree in lateral sign at a material fraction of
targets, report the model as internally ambiguous rather than selecting the
head that better matches behavior.

## Stage 0: freeze the baselines and configuration

1. Preserve the existing OpenSim grid data, configuration, figures, and commit
   identifiers without overwriting them.
2. Preserve the old poster data as a historical result, clearly labeled as
   using the older underconstrained IK/extraction pipeline.
3. Create a new experiment name and output root for the MyoArm run.
4. Record exact versions and hashes for:
   - corrected MyoArm XML;
   - OpenSim source model;
   - proprioception checkpoints;
   - spindle coefficient files;
   - experiment configuration; and
   - target-coordinate table.
5. Fix the right-hand anatomical coordinate convention in the configuration:
   positive rightward is outward, positive forward is away from the body, and
   positive vertical is upward.

**Stop gate:** the MyoArm experiment must not point at any OpenSim IK, muscle
extraction, or output directory by default.

## Stage 1: separate workspace reachability from path policy

The previous 7 cm displacement limit is an experiment/training-envelope rule,
not an IK reachability test. Do not use it while mapping static reachability.

1. Construct the complete unscaled Wang target rectangle in the common body
   frame, retaining the published 24.38 mm column spacing and 53.92 mm row
   spacing.
2. Test static endpoints at scales from 0.25 through 1.00, including enough
   intermediate scales to identify the limiting boundary.
3. Test the exact unscaled corners, mid-edges, center, and a dense workspace
   sample before running all trajectories.
4. Evaluate the same endpoints with:
   - the historical OpenSim IK, for diagnosis only;
   - the corrected regularized OpenSim IK;
   - the differentiable/bounded Nimble IK; and
   - the corrected MuJoCo MyoArm IK.
5. Use the OpenSim-equivalent `MOBL_Handle_r` task point for MyoArm, not the
   index fingertip.
6. For every solver and endpoint, use multiple initial postures and record:
   - requested-versus-achieved Handle error;
   - all solved coordinates;
   - joint-limit margins;
   - solver convergence status;
   - solution branch and sensitivity to initialization; and
   - static muscle/tendon continuity diagnostics.

**Static IK acceptance criteria:**

- independent FK Handle error below the declared tolerance (initially 0.5 cm,
  with MyoArm expected to do substantially better);
- no joint-limit violation;
- no coordinate-frame reflection or axis swap;
- no wrap-branch discontinuity at the endpoint; and
- reproducible solution from declared initializations.

**Stop gate:** do not shrink the grid until the limiting targets have been
classified as anatomically unreachable, solver failures, posture-branch
failures, or training-envelope violations. Report the largest passing scale
and why the next larger scale fails.

## Stage 2: choose the movement protocol after endpoint reachability

Static reachability and center-out path feasibility are separate questions.

1. Test the intended common center start to every accepted endpoint.
2. Independently evaluate:
   - path length;
   - full-trajectory Handle error;
   - joint continuity;
   - muscle-path continuity; and
   - whether joint and muscle states remain within the decoder's training
     distribution.
3. If the full Wang grid is statically reachable but long center-out paths are
   outside the training envelope, compare these declared alternatives before
   choosing one:
   - a different common start;
   - more than one preregistered start region;
   - passive placement from the preceding randomized target, closer to Wang;
   - static placement plus a fixed settling history; or
   - a uniformly scaled grid.
4. Do not translate, anisotropically scale, or silently drop individual
   targets to make the result pass.

**Stop gate:** document the selected protocol, its scientific interpretation,
and its difference from Wang and Jones before generating spindle data.

## Stage 3: validate the corrected MyoArm signal pipeline

1. Generate MyoArm kinematics with the corrected right-handed task basis.
2. Assert a positive task-frame determinant and verify finite-difference Handle
   Jacobian signs against OpenSim at the common rest posture.
3. Extract all relevant MyoArm musculotendon lengths.
4. Apply the declared rigid-tendon/fixed-height fiber-length adapter and the
   canonical-rest offset exactly once.
5. Save raw musculotendon length, adapter output, final fiber length, velocity,
   acceleration, and spindle rates so every transformation is auditable.
6. Check muscle and spindle continuity over every trajectory. Flag wrap
   engagement or branch changes rather than smoothing them away.
7. Re-run the existing matched OpenSim-MyoArm regression on representative
   paths to ensure the corrected adapter and task frame have not regressed.

**Stop gate:** no inference if target geometry, muscle geometry, or spindle
signals fail their validation checks.

## Stage 4: MyoArm pilot

Use a preregistered pilot containing the center, all four corners, and
representative near/far and ipsilateral/contralateral edge targets. The pilot
must sample the eventual full-grid envelope rather than only easy central
locations.

1. Cache IK and muscle geometry once per physical target and movement history.
2. Generate spindle signals once per coefficient seed and reuse them only for
   checkpoints trained with that matching coefficient seed.
3. Run all four training seeds for each of the five coefficient seeds, for 20
   matched models total.
4. Save frame-level direct XYZ predictions and joint predictions.
5. Compute both hand-location readouts:
   - registered direct XYZ;
   - Handle FK from predicted joints.
6. Evaluate arrival, +100 ms, +250 ms, +500 ms, and late-hold windows without
   selecting a window based on behavioral agreement.
7. Produce per-model and ensemble summaries, not only ensemble arrows.

**Pilot acceptance criteria:**

- all requested endpoints pass independent geometry validation;
- all 20 model/seed pairings are matched correctly;
- no duplicated target inputs or predictions;
- both output heads have verified coordinates and units;
- model-to-model dispersion is displayed;
- temporal sign stability is reported; and
- any disagreement between direct XYZ and joint-FK readouts is explicit.

**Stop gate:** review the pilot data and figures before authorizing the full
100-target run.

## Stage 5: full cached analysis

After the pilot passes, run every accepted target through all 20 matched
models.

Cache structure:

- one path/IK result per target;
- one muscle-geometry artifact per target;
- one spindle artifact per target and coefficient seed;
- one batched inference pass per checkpoint; and
- one FK evaluation per unique predicted joint posture where safe.

Required saved data:

- requested and actual physical positions;
- complete IK state and endpoint/path error;
- muscle and spindle intermediates;
- direct XYZ prediction;
- predicted joints;
- joint-derived physical Handle prediction;
- both error vectors and magnitudes;
- per-window values;
- per-model values; and
- ensemble means, standard deviations, sign fractions, and confidence
  summaries.

Automated integrity assertions must verify expected row counts, unique target
identities, unique model-target pairs, coefficient/checkpoint matching, cache
provenance, and exact arrow endpoint identities.

## Stage 6: figures and comparisons

Regenerate the body-context and compact figures for MyoArm. For each readout,
produce both conventions:

1. actual hand -> predicted proprioceptive location;
2. predicted proprioceptive location -> actual hand.

Every figure must state:

- MyoArm or OpenSim backend;
- direct-XYZ or predicted-joint-FK readout;
- endpoint window;
- number of models;
- whether arrows are physical scale or magnified; and
- what the dot, tail, and head represent.

Also produce:

- uncertainty or model-dispersion visualization;
- representative individual-model maps;
- requested-versus-achieved geometry map;
- MyoArm versus corrected-OpenSim comparison;
- old-poster versus corrected-pipeline diagnostic comparison; and
- quantitative Wang and Jones comparison tables.

Do not claim replication from arrow appearance alone. Compare bias sign,
magnitude, near/far contrast, ipsilateral/contralateral contrast, spatial
variation, and uncertainty separately.

## Stage 7: completion checklist

The MyoArm experiment is complete only when all of the following are true:

- the backend is demonstrably corrected MyoArm;
- IK and FK agree within tolerance across the declared workspace;
- the movement protocol and reachable scale are documented;
- spindle inputs are coefficient-seed matched;
- both network output heads are saved and compared;
- no convention uses a constructed third endpoint;
- individual-model variability accompanies the ensemble;
- all expensive shared stages are cached with provenance;
- figures are regenerated from saved tables, not notebook state; and
- conclusions distinguish agreement in direction, magnitude, spatial pattern,
  and task from one another.
