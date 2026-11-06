# Wang 2020 localization-grid experiment

This branch is for a new, validated proprioceptive-localization experiment based
on Wang et al. (2020), *Accuracy of hand localization is subject-specific and
improved without performance feedback* (Scientific Reports 10:19188,
doi:10.1038/s41598-020-76220-0).

This is not a continuation of the removed 24-direction by 3-radius distortion
experiment. The final target locations are a body-centered rectangular grid,
not rings centered on the shoulder or resting hand.

## Coordinate frame

- `x = 0` is the body's sagittal midline.
- Positive `x` points to the model's right. For the right arm, positive lateral
  error is therefore outward and negative lateral error is inward.
- `y = 0` is the torso reference plane; positive `y` points forward, away from
  the body.
- Use the horizontal projection of the OpenSim `C7` marker as the body-centered
  `(x, y)` origin. Do not use `R.Shoulder` as the grid origin.
- All targets share one horizontal `z` value. Determine it from the validated
  KINARM-like center posture; C7 defines the horizontal body reference but not
  the table height.

## Final target coordinates

The paper used 100 targets in 5 rows by 20 columns. Its workspace was 48.76 cm
wide and 26.96 cm deep, with adjacent columns separated by 2.438 cm and rows by
5.392 cm. The target array was approximately centered on the body midline and
began about 20 cm in front of the participant.

Use these 20 lateral coordinates, in centimeters relative to body midline:

```text
-23.161, -20.723, -18.285, -15.847, -13.409,
-10.971,  -8.533,  -6.095,  -3.657,  -1.219,
  1.219,   3.657,   6.095,   8.533,  10.971,
 13.409,  15.847,  18.285,  20.723,  23.161
```

Equivalently, for column `j = 0, ..., 19`:

```text
x[j] = (j - 9.5) * 2.438 cm
```

Use these five depth coordinates, in centimeters forward of the torso
reference:

```text
22.671, 28.063, 33.455, 38.847, 44.239
```

These values preserve the paper's 5.392 cm row spacing and the approximately
2.671 cm margin between the near workspace boundary and first row shown in its
Figure 1. The middle of the target array is therefore:

```text
center = (x=0.000 cm, y=33.455 cm, z=validated table height)
```

Generate the 100 targets as the Cartesian product of the 20 `x` values and five
`y` values.

## Initial center-out protocol

For the first controlled experiment, use the grid center above as the common
starting posture. Each trial should contain:

1. a common pre-reach hold;
2. a fixed-duration minimum-jerk passive movement from the center to one target;
3. an endpoint hold of at least 1 second; and
4. an optional return only after the endpoint measurement window.

Record decoded error as `prediction - actual`. Analyze reach arrival and
multiple times during the endpoint hold so temporal memory is measured rather
than hidden. Draw decoder arrows from actual position to predicted position.
If a predicted Wang-style behavioral matching error is later shown, plot it
separately and label it as the negative of the decoder error.

## Reachability and scaling rule

Attempt the full coordinates first. Before producing spindle signals or neural
predictions, solve and forward-kinematically verify one canonical endpoint
posture for every target with the wrist brace and regularization fixed.

The full grid spans 46.322 cm laterally and 21.568 cm in depth. Its farthest
corner is about 25.55 cm from the common center. If all 100 targets cannot be
reached within the declared IK, joint-limit, and training-distribution
tolerances, apply one uniform scale about the grid center:

```text
scaled_target = center + scale * (target - center)
```

Use the largest single scale that passes every target. Do not scale the axes
independently, translate the grid toward the right shoulder, drop difficult
targets silently, or substitute achieved locations without reporting them.
Record the chosen scale in the experiment configuration and results.

## Required geometry validation

The proprioception stages must not run until the requested and achieved grid is
verified independently. At minimum:

- lock the declared wrist and forearm coordinates consistently;
- use the same rest-posture regularization for every target;
- forward-kinematically reconstruct the `Handle` location;
- enforce a strict Cartesian endpoint tolerance;
- check joint limits, training-distribution bounds, and continuity;
- cache the canonical endpoint solution so every repeat of a target ends in
  the same physical posture; and
- save a body-centered requested-versus-achieved grid plot.

Any failure stops the experiment instead of producing a distorted localization
map.
