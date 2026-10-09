# Corrected MyoArm fiber adapter and muscle geometry

This document describes the two changes that allow the MuJoCo MyoArm model to replace the MoBL OpenSim muscle calculation while retaining the existing Mathis spindle layer and pretrained CNN:

1. converting MuJoCo musculotendon length into the fiber length expected by the spindle model; and
2. correcting the MyoArm paths whose geometry differs materially from the MoBL OpenSim model used by the proprioception pipeline.

The implemented backend is named `myosuite_corrected`. It is built from pinned `myo-sim==0.2.3`; it does not modify the installed package in place.

## A. Musculotendon-to-fiber adapter

### Why an adapter is required

`mujoco_data.actuator_length` is the complete musculotendon path length `L_MT`: the distance along the routed path from origin to insertion, including the tendon and the projection of the pennated fibers along the tendon.

The OpenSim data used to train the spindle/CNN pipeline contains muscle-fiber length `L_f`. These are not interchangeable signals. Passing MuJoCo actuator length directly into the Mathis layer changes both the length scale and its zero point.

### Source model and parameters

The parameters come from the 25 corresponding `Millard2012EquilibriumMuscle` elements in the paper model `MOBL_ARMS_41_seb_writing_pos.osim`:

| Symbol | OpenSim property | Meaning |
|---|---|---|
| `L_0` | `optimal_fiber_length` | Fiber length at which active isometric force is maximal |
| `L_TS` | `tendon_slack_length` | Tendon length below which the tendon develops no tensile force |
| `alpha_0` | `pennation_angle_at_optimal` | Angle between fiber and tendon when `L_f = L_0` |

The same three arrays are present in this repository's `DefaultMOBL_ARMS_fixed_41.osim`; all 25 values and their muscle ordering were verified to be identical. Values are converted from OpenSim meters to the pipeline's millimeters and stored in canonical `MUSCLE_NAMES` order in `mujoco_pipeline/spindles.py`.

Pennation is the angle between the muscle fibers and the tendon. A pennation angle of zero means fibers pull parallel to the tendon. For a pennated muscle, only `L_f cos(alpha)` lies along the tendon direction.

### Derivation

The adapter follows Sec. 2.4, “Rigid-Tendon Musculotendon Model,” of Millard et al., *Flexing Computational Muscle: Modeling and Simulation of Musculotendon Dynamics* (2013). The relevant equations are Eq. 9 and Eqs. 11–12; Fig. 1 illustrates the fixed-height pennation geometry.

Millard Eq. 9 decomposes musculotendon length into tendon length and the fiber component parallel to the tendon:

```text
L_MT = L_T + L_f cos(alpha).                         (Millard Eq. 9)
```

The rigid-tendon assumption in Sec. 2.4 sets tendon length to its slack length:

```text
L_T = L_TS.
```

Therefore, the fiber projection along the tendon is

```text
x = L_MT - L_TS = L_f cos(alpha).
```

The paper uses a fixed-height parallelogram for pennation. Millard Eq. 11 is

```text
L_f sin(alpha) = h,                                  (Millard Eq. 11)
```

and Eq. 12 evaluates that constant height from the muscle's optimal parameters:

```text
h = L_0 sin(alpha_0).                                (Millard Eq. 12)
```

The along-tendon component `x` and perpendicular component `h` are orthogonal components of the fiber. Applying the Pythagorean theorem gives

```text
L_f^2 = x^2 + h^2

L_f = sqrt((L_MT - L_TS)^2 + (L_0 sin(alpha_0))^2).
```

The implementation clamps `L_MT - L_TS` to zero before taking the square root:

```text
along_tendon = max(L_MT - L_TS, 0)
height       = L_0 * sin(alpha_0)
L_f_adapter  = sqrt(along_tendon^2 + height^2)
```

The clamp prevents a negative projected fiber length if a numerical or routing configuration produces `L_MT < L_TS`.

This closed-form equation is derived from Millard's published rigid-tendon and fixed-height equations; the square-root form is an algebraic consequence, not a separately numbered equation printed in the paper. The paper explicitly states after Eq. 13 that Eqs. 9 and 11 are solved for muscle length when tendon length is fixed at `L_TS`.

Reference: [Millard et al. 2013, especially Fig. 1 and Sec. 2.4, Eqs. 9–13](https://nmbl.stanford.edu/publications/pdf/Millard2013.pdf).

### OpenSim rest alignment

The geometric adapter answers: “Given this MuJoCo musculotendon path length, what fiber length would a rigid-tendon, fixed-height Millard muscle have?” The OpenSim baseline answers a slightly different question. Its `Millard2012EquilibriumMuscle` has an elastic tendon and determines how the total length is divided between tendon and fiber by solving muscle–tendon force equilibrium. Tendon length is therefore not always exactly `L_TS`.

Put concretely, the MuJoCo model supplies only the total path length, `L_MT`. The adapter subtracts the source model's tendon slack length and reconstructs the pennated fiber triangle. That is a useful geometric estimate, but `L_TS` is the tendon's *slack* length, not a measurement saying that the tendon is exactly that long in every posture. In equilibrated OpenSim, tendon force can stretch the tendon beyond slack length and the fiber then occupies correspondingly less of `L_MT`. Activation, passive fiber force, and the force–length curves participate in that equilibrium. The closed-form adapter deliberately does not reproduce that stateful force-balance calculation.

This distinction creates two separable kinds of error:

1. **Posture-dependent error.** If attachments, moving points, or wraps are wrong, the change in musculotendon length with posture is wrong. The muscle-path corrections in Section B address this.
2. **Absolute-origin error.** Even with the correct length change, the rigid-tendon approximation can assign a different constant amount of the total length to fiber than OpenSim's elastic equilibrium solution. The rest alignment addresses this.

Why does an absolute offset matter? The Mathis spindle layer receives normalized fiber length as well as fiber velocity and acceleration. A constant fiber-length difference leaves velocity and acceleration unchanged, but changes the tonic length-dependent firing rate during a stationary hold. The CNN was trained on those OpenSim tonic firing patterns and uses them to infer absolute arm posture. Matching only movement-related rate changes is therefore insufficient for an absolute-position decoder.

Normalizing by `L_0` does not remove this problem. `L_0` is a scale—the optimal fiber length—not the actual fiber length at the experiment's rest posture. Dividing two signals by the same `L_0` preserves any difference between their absolute fiber lengths.

### How the need for alignment was identified

The first complete corrected-MyoArm notebook run provided three diagnostic facts:

- MuJoCo IK followed the commanded path with less than 0.05 cm maximum error, so the decoded shift was not an IK failure.
- Changing the tracked point from the MyoArm fingertip to the OpenSim-equivalent `Handle` made the task definition correct but did not materially remove the decoded shift.
- At the common stationary rest pose, the adapter and equilibrated OpenSim fibers differed by 9.65 mm RMS across the 25 muscles. This produced systematic tonic spindle-rate differences and a 5.07 cm decoded starting-position error shared by every movement.

Because every trajectory began at the same physical posture and showed the same decoded displacement, the error had the signature of an input baseline mismatch rather than a path-shape or IK error. That motivated a one-posture alignment test. Its large improvement confirmed that the absolute fiber-length origin was a dominant cause. In the final matched rerun, the residual initial L2 error was 1.82 cm and was numerically the same for OpenSim and corrected MyoArm; it is therefore the frozen CNN's common rest-pose decoding bias rather than a MuJoCo-specific offset.

This is the same logic as zeroing two measurement instruments at a shared reference condition. We did **not** choose offsets by minimizing trajectory error. We first identified a common physical state, asked both muscle formulations what each fiber length was in that state, and used their difference as the change of origin. The subsequent CNN improvement is validation of that independently defined alignment, not the quantity used to fit it.

### What is aligned

For the shared CNN pipeline, the adapter is anchored once at the canonical braced rest pose:

```text
q_rest = (elv_angle, shoulder_elv, shoulder_rot, elbow_flexion)
       = (20, 40, 25, 85) degrees

offset_i = L_f_OpenSim_i(q_rest) - L_f_adapter_i(q_rest)

L_f_used_i(q) = L_f_adapter_i(q) + offset_i.
```

For each muscle, `L_f_OpenSim(q_rest)` is obtained by posing the MoBL OpenSim model at the declared rest posture, equilibrating its muscles, and reading the resulting fiber length. `L_f_adapter(q_rest)` is obtained by posing corrected MyoArm identically, reading MuJoCo musculotendon length, and applying the geometric adapter. Their difference is stored as that muscle's offset.

The two values used for muscle `i` therefore come from different calculations but the same declared posture:

1. **Reference value:** initialize MoBL, set the seven named coordinates (including the wrist brace), realize/equilibrate the model, and read OpenSim's `fiber_length` for muscle `i`.
2. **MuJoCo estimate:** initialize corrected MyoArm, set the corresponding coordinates, read tendon path length `L_MT`, and evaluate the rigid-tendon adapter using that muscle's MoBL `L_TS`, `L_0`, and `alpha_0`.
3. **Stored correction:** subtract step 2 from step 1. At runtime that single per-muscle number is added after the adapter and before differentiation and Mathis normalization.

This requires:

- the same named rest coordinates in both models;
- the same wrist brace (`pro_sup = -30`, `deviation = 0`, `flexion = 0` degrees);
- one equilibrated OpenSim reference fiber length per muscle; and
- the matching 25-muscle ordering.

This additive calibration is project-specific; it is not an equation from Millard et al. It is also not a trajectory fit: no movement samples, decoded positions, CNN outputs, slopes, or scales are optimized. One constant is established per muscle from one declared physical posture. Therefore

```text
dL_f_used/dt    = dL_f_adapter/dt
d2L_f_used/dt2  = d2L_f_adapter/dt2,
```

so length excursions, velocity, acceleration, moment arms, and routing behavior are unchanged. The offset only aligns the absolute length term presented to the Mathis layer. This is why rest alignment cannot repair a bad muscle route; it complements rather than replaces the structural changes in Section B.

The stored values are valid only for the declared canonical rest pose. The code refuses to apply them to a different rest posture. A different rest pose would require recomputing the OpenSim and adapter rest values at that posture, or replacing this approximation with a full elastic-tendon equilibrium calculation.

This alignment reduced the earlier 5.07 cm MuJoCo-specific decoded displacement and brought the final initial L2 error to the OpenSim pipeline's same 1.82 cm baseline. It reduced mean eight-direction wrist RMSE from 3.51 cm to 1.08 cm without changing the CNN or spindle coefficients.

### Signal passed to the Mathis layer

For every frame, the pipeline performs

```text
MuJoCo L_MT
  -> rigid-tendon/fixed-height L_f_adapter
  -> canonical-rest offset L_f_used
  -> numerical velocity and acceleration
  -> normalization by MoBL L_0
  -> unchanged Mathis Ia and II equations
  -> unchanged pretrained CNN.
```

The raw musculotendon length, unaligned adapter length, rest offset, and final fiber length are retained in generated artifacts so this transformation can be audited.

## B. Differences from the original MyoArm geometry

### Task point: OpenSim Handle rather than index fingertip

The original MyoArm reach task tracks `IFtip_r`, the index fingertip. The OpenSim pipeline tracks the `Handle` marker rigidly attached to the hand. These points have different Jacobians, so perfect IK at the fingertip does not produce the same joint trajectory as perfect IK at the Handle.

The corrected model adds `MOBL_Handle_r` to the rigid hand at the converted OpenSim Handle location and uses it as the MuJoCo IK end effector. At the canonical rest pose it represents the same physical task point as the OpenSim marker. Its numeric coordinates need not be identical before registering the two engines' world frames; the validation below compares displacements in the common task convention.

The MuJoCo IK itself was not inaccurate: before and after this correction its maximum target error was approximately 0.05 cm. The correction makes the point being solved anatomically and computationally equivalent to the OpenSim task.

### Retained muscle-path corrections

The table below describes every retained change relative to the original `myo-sim==0.2.3` right MyoArm. Muscles not listed retain their original MyoArm attachments, sites, wraps, and tendon ordering.

| Muscle | Original MyoArm | Corrected model | Why it differs from MoBL |
|---|---|---|---|
| `TRIlong` | Routes around `TRIlonghh` and through a fixed synthetic `P1b` site; the glenoid wrap is omitted. | Replaces that detour with the converted `TRIlongglen` cylinder and side site while retaining the elbow `TRI` wrap. | MoBL routes the long head around the glenoid. The fixed `P1b` approximation produced incorrect shoulder dependence. |
| `DELT2` | Uses a static approximation through the `delt2hum` and `Deltoid2` wrap family. | Adds a shoulder-rotation-dependent moving point using three polynomially coupled slide joints, then routes through the common humeral-head sphere `DELT_TMAJ_LAT_PEC_CORBhh`. | MoBL's middle-deltoid point moves with shoulder rotation and uses the shared humeral-head wrap; a fixed route cannot reproduce its moment arm. |
| `PECM1` | Uses fixed intermediate sites and primarily the `PEC1hh` route. | Adds elevation-dependent humeral and torso-frame points, restores `PEC12hum`, and uses the MoBL terminal clavicular attachment. | MoBL changes the path with shoulder elevation. The fixed MyoArm points changed both length offset and elevation dependence. |
| `BIClong` | Uses only part of the sequential contact path around the humeral-head ellipsoid. | Restores the second and third contacts around the existing `BIClong` ellipsoid. | The omitted contacts shorten and reshape the shoulder portion of the long-head path. |
| `BICshort` | Substitutes an elbow wrap and fixed distal routing. | Removes the substituted wrap and adds the MoBL pronation/supination-dependent distal point using three polynomially coupled slide joints. | MoBL's distal point moves with `pro_sup`; the original fixed approximation cannot preserve that dependence. |
| `SUPSP` | Uses a converted attachment and wrap whose pose differs from the source MoBL path. | Restores the MoBL scapular attachment and wrap pose/size while retaining MyoArm's stable side selector. | The source attachment and wrap geometry improve length and moment-arm agreement; the literal converted side selector selected an unstable MuJoCo wrap branch. |
| `TMIN` | Leaves the relevant humeral wrap commented out, so the tendon has no active equivalent contact there. | Activates the structurally corresponding `TMINhum` cylinder with an explicit side site. | MoBL contains active humeral wrapping for TMIN. This restores the missing structure, although residual TMIN behavior remains sensitive to wrap-branch selection. |

### How moving OpenSim points are represented

OpenSim permits path points whose coordinates are functions of a joint angle. MuJoCo tendon sites are normally fixed in their parent body. The corrected model represents each moving coordinate using a low-mass helper body with orthogonal slide joints. MuJoCo equality constraints couple those slides polynomially to the relevant anatomical hinge coordinate. The path point therefore moves with the same coordinate while remaining inside MuJoCo's differentiable kinematic model.

MuJoCo wraps are differentiable within a fixed contact branch but only piecewise smooth when a tendon engages, disengages, or changes wrap side. The retained routes use side sites that stayed on a continuous branch over the task envelope.

## Validation summary

Against the paper OpenSim model over six smooth trajectories within its stated arm envelope, the retained geometry plus the geometric adapter produced the following continuous-branch spindle agreement before applying the later canonical-rest offset:

| Metric | Result |
|---|---:|
| Aggregate spindle-rate RMSE | 1.633 Hz |
| Aggregate spindle-rate correlation | 0.951 |
| Median per-muscle RMSE | 1.302 Hz |
| Median per-muscle correlation | 0.963 |
| Largest MuJoCo one-frame rate change | 1.91 Hz |

The continuous-branch score excludes only neighborhoods around abrupt OpenSim-only wrap/path transitions greater than 10 Hz in one frame. Across all samples including those OpenSim transitions, aggregate RMSE was 3.536 Hz and correlation was 0.816.

In the complete eight-direction notebook regression, adding the OpenSim-equivalent Handle target and canonical-rest alignment produced:

| CNN output metric | Result |
|---|---:|
| Initial wrist-position L2 error | 1.82 cm (the same as OpenSim) |
| Mean wrist RMSE | 1.08 cm |
| Mean wrist L2 error | 1.81 cm |
| Mean shoulder-elevation RMSE | 2.02 degrees |
| Mean shoulder-rotation RMSE | 0.80 degrees |
| Mean elbow-flexion RMSE | 1.49 degrees |

### Matched OpenSim–MuJoCo decoded-path check

We also ran the same eight nominal 10 cm center-out commands, rest posture, timing, pretrained CNN, coefficient seed, and padding through both complete pipelines. Error below is the CNN output against that pipeline's analytic-FK label convention:

| Mean over eight reaches | OpenSim + Nimble IK | Corrected MyoArm + adapter |
|---|---:|---:|
| Wrist RMSE | 1.02 cm | 1.08 cm |
| Mean wrist L2 error | 1.74 cm | 1.81 cm |
| Shoulder-elevation RMSE | 1.89 degrees | 2.02 degrees |
| Shoulder-rotation RMSE | 0.68 degrees | 0.80 degrees |
| Elbow-flexion RMSE | 1.51 degrees | 1.49 degrees |

These close values show that, once given each pipeline's own joint trajectory, the corrected MuJoCo muscle signals drive the frozen CNN about as accurately as the OpenSim signals.

#### Task-frame correction

The first matched plot initially showed a systematic lateral reflection: right and left were exchanged while forward and backward were preserved. This was traced to the task basis in `mujoco_pipeline/reaching.py`, not corrected by relabeling paths or transforming CNN outputs.

The former MuJoCo basis used the horizontal shoulder-minus-torso vector, followed by `cross(right, up)`, and then `up`. Those columns have determinant `-1`, so the basis was left-handed. The OpenSim `S2W` transform used by the path generator has determinant `+1`. A finite-difference check at the common rest posture confirmed the consequence: OpenSim and MuJoCo Handle Jacobians had comparable forward/up derivatives but opposite lateral derivatives for all four driven coordinates.

The MuJoCo basis now uses the OpenSim task-X direction from shoulder toward torso and constructs forward as `cross(up, task_x)`. This produces a right-handed basis with determinant `+1`. The code asserts positive determinant at runtime. This is a frame-registration correction derived from anatomical landmarks, cross-product handedness, and the end-effector Jacobian; it is not an empirical path swap or fitted output transform. After the change, the MuJoCo Handle Jacobian's lateral signs agree with OpenSim and the same-name directions align directly.

For the final comparison, each predicted wrist path was translated to start at zero, then compared sample-for-sample with the same named OpenSim path across all eight reaches. No reflection, direction reassignment, rotation, scaling, or fitted alignment was applied:

| Same-direction predicted-path metric | Final result |
|---|---:|
| Coordinate RMSE across X/Y/Z | 0.419 cm |
| RMS three-dimensional point separation | 0.727 cm |
| Aggregate centered-path correlation | 0.983 |
| X correlation | 0.988 |
| Y correlation | 0.990 |
| Z correlation | 0.671 |

The lower Z correlation occurs because these are nominally planar reaches with relatively little Z excursion, so small absolute vertical differences have a large effect on correlation. The three-dimensional RMSE includes that component. These results establish close same-direction decoded-path alignment while retaining the unchanged CNN and Mathis layer.

## Implementation map

- `mujoco_pipeline/corrected_myoarm.py`: reproducible MyoArm XML transformations.
- `mujoco_pipeline/models.py`: corrected backend and `MOBL_Handle_r` selection.
- `mujoco_pipeline/spindles.py`: MoBL parameters, adapter equation, and canonical OpenSim rest fibers.
- `process/mujoco_ik.py`: arbitrary-path MuJoCo IK.
- `process/mujoco_extract.py`: musculotendon extraction, adapter, and rest alignment.
- `process/mujoco_spindles.py`: unchanged Mathis Ia/II calculation interface.
- `run_pipeline.py` and `run_pipeline.ipynb`: CONFIG-selected OpenSim or corrected-MyoArm execution.
