# Wang-style proprioceptive localization experiment

This branch develops a body-centered localization experiment inspired by Wang
et al. (2020), *Accuracy of hand localization is subject-specific and improved
without performance feedback* (Scientific Reports 10:19188,
doi:10.1038/s41598-020-76220-0).

The experiment asks whether the spindle-to-position models contain a systematic
spatial bias resembling human proprioceptive localization bias. It does not
attempt to model visual matching, motor planning, or active control. The arm is
moved through prescribed kinematics and the neural model reports its predicted
hand position.

This work is separate from the removed polar-grid distortion experiment. The
target coordinates and acceptance rules are specified in
[`docs/wang2020_localization_grid.md`](docs/wang2020_localization_grid.md).

## Human result being tested

Wang and colleagues measured the localization of the hidden left hand at 100
targets in a 5-row by 20-column rectangular workspace. Participants actively
moved a stylus until their unseen hand felt aligned with a visible target. They
could correct the hand position before confirming the judgment.

The physical matched hand positions tended to lie rightward of the visual
targets. Equivalently, the participants' proprioceptive estimates placed their
left hand leftward of its actual position. For a left hand this is an outward,
away-from-midline perceptual bias. A depth-dependent component was also present
at part of the workspace: physical matching errors tended toward the body,
which corresponds to perceiving the hand farther from the body than it really
was.

This repository models a right arm. The homologous human prediction is therefore
an outward/rightward decoded bias, not a leftward decoded bias. Results will be
reported in both world coordinates and body-relative inward/outward coordinates
to avoid confusing the two hands.

## Primary experimental question

After identical center-out movement histories, does the decoded endpoint
position show a systematic body-relative bias across a Wang-like workspace?

The direct decoder error is

```text
decoder_error = predicted_position - actual_position
```

For the right arm:

- positive lateral error is rightward and outward;
- negative lateral error is leftward and inward;
- positive depth error is forward, away from the body; and
- negative depth error is backward, toward the body.

Plots of decoder error must draw arrows from actual to predicted position. A
predicted visual-matching error, if shown later, is a different quantity with
approximately the opposite sign and must be plotted separately.

## Initial center-out protocol

The first experiment deliberately uses a common center-out movement. This does
not remove the temporal memory of the causal decoder; it standardizes that
memory across target locations.

For every target:

1. hold the arm at the common center posture;
2. passively move it to the target with a fixed-duration minimum-jerk profile;
3. hold the exact endpoint posture for at least 1 second; and
4. return only after all endpoint measurement windows, if a return is needed.

The common center is the center of the target rectangle, not the right shoulder
or the original resting hand location.

The first analysis will compare predictions at several times:

- the final movement/arrival frame;
- 100 ms after arrival;
- 250 ms after arrival;
- 500 ms after arrival; and
- a late-hold window average.

This distinguishes a movement-termination bias from a bias that persists after
dynamic spindle signals settle. A later experiment can vary approach direction
or use randomized target-to-target movements to measure history dependence
directly.

## Target workspace

The intended final workspace is a 5 by 20 rectangle defined relative to the
body midline and torso, using the OpenSim `C7` marker as the horizontal body
reference. It spans approximately:

```text
lateral: -23.161 to +23.161 cm relative to the body midline
depth:    22.671 to 44.239 cm forward of the torso reference
```

Adjacent columns are 2.438 cm apart and adjacent rows are 5.392 cm apart. The
common center-out origin is approximately `(0.000, 33.455)` cm in this
body-centered horizontal frame. See the detailed grid document for all 25 axis
coordinates and the exact Cartesian-product definition.

The full Wang-sized grid will be attempted first. If it cannot be reached and
validated at every point, the rectangle may be uniformly scaled about its
center. The axes may not be scaled independently, and the grid may not be moved
toward the right shoulder to make IK easier.

## Geometry validation gate

No spindle generation or neural inference is considered valid until the
requested movement geometry passes an independent check.

The implementation must:

- lock the declared wrist and forearm coordinates consistently;
- use one fixed regularization policy for all targets;
- solve and cache one canonical endpoint posture per target;
- forward-kinematically reconstruct the OpenSim `Handle` position;
- compare requested and achieved positions frame by frame;
- enforce a strict endpoint error threshold;
- check joint limits, training-distribution bounds, and joint continuity; and
- save a body-centered requested-versus-achieved target plot.

If any target fails, the experiment stops before proprioceptive inference. This
prevents biomechanical or IK distortion from being mislabeled as neural bias.

## Matched model ensemble

The robustness ensemble contains matched spindle/model combinations rather than
twenty predictions from one shared spindle input:

```text
5 spindle coefficient seeds x 4 neural-network training seeds = 20 models
```

Each checkpoint must receive spindle signals generated with its own matching
coefficient seed.

Every model first produces its own 100-target bias map. The primary saved table
will have one row per model, target, and measurement window, including:

- model coefficient seed and training seed;
- target row and column;
- body-centered actual and predicted XYZ;
- lateral, depth, and vertical decoder errors;
- inward/outward error;
- Euclidean error magnitude;
- requested-versus-achieved IK error; and
- measurement time or hold window.

Only after saving the individual maps will results be summarized across models.
Planned summaries include:

- mean and confidence interval of each error component at each target;
- fraction of models showing outward versus inward error at each target;
- consistency of the sign across model seeds;
- near-versus-far and left-versus-right regional comparisons;
- dependence of error on target coordinates and distance from the body; and
- change in bias from reach arrival through the endpoint hold.

The 20 models are robustness replicates, not 20 independent human subjects.
Human-style binomial counts may be displayed descriptively, but should not be
presented as a literal replication of the human participant statistics.

## How the removed experiment differed

The removed experiment generated 72 polar targets: 24 directions at radii of
4, 8, and 12 cm. Every trial moved from a hand-centered common origin to one
target and back. Its main heatmap used the final outbound-reach frame rather
than a settled endpoint hold.

It also had several interpretation and implementation problems:

- targets were centered on a resting hand/shoulder-relative frame rather than
  the Wang body-centered rectangular workspace;
- underconstrained one-marker IK distorted the nominal rings;
- endpoint geometry was not protected by the validation gate above;
- poster arrows were drawn from predicted to true position, opposite the
  natural decoder-error convention; and
- the apparent rightward visual agreement obscured an underlying leftward,
  inward prediction for the modeled right hand.

Those results are not evidence that the model replicated the human bias.

## Planned implementation stages

1. Implement the body-centered rectangular path generator and experiment
   configuration.
2. Solve and validate the common center and all 100 canonical endpoint
   postures.
3. Determine whether the full grid passes or record the required uniform scale.
4. Generate and validate the center-out trajectories.
5. Run one matched checkpoint as an end-to-end smoke test.
6. Run all 20 matched spindle/model combinations.
7. Save individual-model maps before generating ensemble summaries.
8. Produce sign-consistent bias maps and arrival-to-hold temporal analyses.

The initial center-out result is a controlled movement-history experiment. A
subsequent extension should approach each endpoint from multiple directions to
separate persistent posture-dependent bias from trajectory-dependent memory.
