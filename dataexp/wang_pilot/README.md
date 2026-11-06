# Five-target Wang pilot

This directory contains the first geometry-gated right-arm pilot for the Wang
localization experiment. `pilot_model_0_9_bias.csv` has one row per target and
measurement window. Its primary errors are decoded-joint OpenSim Handle minus
true OpenSim Handle in C7-relative anatomical coordinates:

- positive `error_outward_cm` is rightward/outward for the right arm;
- positive `error_forward_cm` is forward/away from the body; and
- positive `error_up_cm` is upward.

The five retained targets are exact Wang-grid locations: center, near-center,
far-center, x = -6.095 cm at middle depth, and x = +18.285 cm at middle depth.
Every endpoint passes the configured 0.5 cm requested-versus-achieved gate;
the maximum is 0.451 cm. Attempts at the exact -23.161, -18.285, and -10.971
cm middle-depth columns missed by 1.059, 0.853, and 0.600 cm respectively and
were rejected before neural interpretation. This indicates that an unscaled
full-width grid is not valid for this right-arm setup under the fixed IK policy.

At the late hold, model 0_9 has outward errors at four of five retained targets:
center +0.570 cm, near +0.842 cm, far +0.967 cm, crossed-midline +1.260 cm, and
right +18.285 cm -0.514 cm. Forward/depth error changes with location: +1.131
cm at the near target, -0.428 cm at the far target, and small negative errors at
the two lateral targets. The lateral pattern is therefore mostly consistent
with the homologous right-hand behavioral prediction, but it reverses at the
rightmost pilot target and the depth effect is not globally consistent.

The other three valid coefficient-seed-0 checkpoints were run on the identical
spindle signals. Across these four models, 14 of 20 target-level late-hold
errors are outward. All four models are outward at center, far-center, and the
retained crossed-midline target; all four are inward at the +18.285 cm right
target; and the near target splits two outward versus two inward. Every model's
five-target mean is outward (+0.625 to +0.941 cm). The ensemble-mean depth bias
is location-dependent and much less coherent. See
`pilot_valid_seed_ensemble_summary.csv` for every target and time window.

`old_model_0_9_summary.csv` is a provenance snapshot calculated from
`main:dataexp/centerout/multi_seed_results_matched.csv`. In the old analytic
coordinate convention, positive X was labeled rightward. Model 0_9's mean raw
X error across the 72 polar-grid targets was -0.679 cm, with negative X at
69.4% of targets. Across the four valid coefficient-seed-0 checkpoints, the
mean pointwise X error was negative at all 72 targets. Thus the underlying old
right-arm decoder error was predominantly leftward/inward, opposite the
homologous Wang lateral prediction. The poster plotted arrows from predicted
to true, which visually reversed that decoder-error direction. Old raw analytic
errors and new physical Handle errors are retained as separate quantities and
must not be numerically pooled.

Reproduce the pilot with:

```powershell
python run_pipeline.py --config experiments/wang_pilot.yaml --through inference
$env:MOTOR_META_CONFIG=(Resolve-Path experiments/wang_pilot.yaml).Path
$env:MOTOR_META_EXPERIMENT='wang_pilot'
python process/analyze_wang_pilot.py
```
