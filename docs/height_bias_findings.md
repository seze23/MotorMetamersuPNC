# Workspace-height bias investigation

## Current conclusion

Workspace height causally changes the lateral localization bias in the OpenSim
pipeline. This is supported by two paired manipulations that keep horizontal
target coordinates fixed and change only the endpoint-height plane.

| Test | Lower plane | Higher plane | Paired change |
|---|---:|---:|---:|
| Wang grid (target-mean outward bias) | -0.673 cm at -23.2 cm | +0.749 cm at +4.0 cm | +1.422 cm; 80/98 targets shift outward |
| Static 80-target multiring set | -0.076 cm at -23.2 cm | +0.714 cm at +4.0 cm | +0.790 cm; 80/80 target means shift outward |

The multiring comparison is the cleaner causal test: the same 80 horizontal
endpoints are held statically at each height, avoiding reach-path and speed
differences. Across individual model-target pairs, 1407/1600 shifted outward.
All physical inverse-kinematics solutions passed (maximum endpoint error about
0.00004 cm), and the tested muscle-length range stayed above TMIN.

## What the heights mean

Heights are Handle endpoint heights relative to the model's C7 origin. The
archived poster trajectories do not contain an explicit `+4 cm` target-setting
line. Direct forward kinematics of their final Handle positions gives a mean
of +3.861 cm, median +4.427 cm, range -2.857 to +7.911 cm; 70/80 endpoints are
above C7. `+4 cm` is therefore a rounded representative plane, not a literal
constant in the original generator.

## Readout and interpretation limits

The arrows in the retained figures run from the physical Handle endpoint to
the CNN's decoded analytic-wrist XYZ position. The CNN was trained on that
analytic label, so this is not exactly a Handle-to-Handle comparison. Height is
demonstrated to affect the model's decoded bias, but the present experiments do
not by themselves identify which muscle/posture feature mediates the effect.

The lower and higher planes also differ in their forward component. The
multiring mean changes from +1.12 cm forward at the lower plane to +0.12 cm at
the higher plane. Thus the result is not merely a uniform lateral offset.

## MyoArm replication status

The paired MyoArm full-grid run is incomplete and is not included as a finding.
Physical artifacts for 200 conditions were generated; 88 paired targets passed
the matching filter. Ensemble inference completed for three of twenty models,
with a fourth partial. These results are preserved locally for resumption.

