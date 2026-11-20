## 7. Research-meeting audit: paper, posture, spindle validity, and averaging

This audit distinguishes **what the paper reports**, **what our saved simulation actually does**, and **what remains unvalidated**. The original Sections 1–6 still show the single `0_9` checkpoint; the checkpoint comparison below is separate. No weights, original trajectories, OpenSim geometry, or original predictions were edited for this audit.

### 7.1 What the human experiment actually tested

The source is **Viviani, Baud-Bovy & Redolfi (1997)** (not Vivaldi), *Perceiving and Tracking Kinesthetic Stimuli: Further Evidence of Motor-Perceptual Interactions*. Our comparison concerns **Experiment 1**. Experiment 2 asked participants to reproduce the right-arm movement with the left arm; this notebook does not simulate that bilateral task.

**Participants and sensory conditions.** Experiment 1 involved 20 right-handed adults (8 women, 12 men). They sat in an adjustable chair, were blindfolded, and heard masking noise. A six-degree-of-freedom robot passively displaced their **right hand**. Their left hand pressed buttons to answer **“flat” or “tall.”** There was no explicit “circle” response and no drawing of a perceived circle in this experiment.

**Arm and grip (printed p. 1234, Figure 1 and Method).** Participants held an anatomical grip connected by a three-degree-of-freedom ball-and-socket joint. The hand could remain **palm down throughout motion**. The authors describe almost complete arm extension, with movement mainly involving shoulder rotation. Figure 1 is a schematic, not calibrated motion-capture data. It does not supply exact shoulder angles, elbow angles, segment lengths, scapular motion, or hand-orientation time series for the participants. Exact participant-by-participant posture cannot be recovered from this paper alone.

**Path placement and size.** The robot moved in a vertical **frontoparallel plane**, roughly 45 cm from the chest, centered on the right shoulder. All paths had perimeter **160 cm** and period **2.5 s**, giving mean tangential speed **64 cm/s**. A circle therefore had diameter **50.93 cm**. At absolute eccentricity .95, full major/minor diameters were approximately **75.44 × 23.55 cm**. The paper's quoted values 25.46–37.72 and 11.77–25.46 cm correspond to semiaxis lengths; they are not circle diameters. Signed eccentricity was positive for horizontal elongation, negative for vertical elongation, zero for a circle.

**Correction to the supplied findings notes.** The notes correctly describe overestimation of the horizontal dimension in B, but then call the resulting perceived circle a “tall” ellipse; those labels conflict. Horizontal stretching means a circle tends to appear **flat/horizontal** in B. The analogous C example is reversed as well: vertical stretching tends toward **tall/vertical**. The primary paper is used to resolve these contradictions. In the notes' simple power-law form, V ∝ R^(1−β) = curvature^(β−1); writing curvature^(−β) with β=2/3 is inconsistent with the accompanying R^(1−β) expression. The notebook's implemented path generator, not those summary equations, defines the actual stimuli.

**Kinematics (printed p. 1235).** A used constant tangential speed. B slowed near the left/right extremes; C slowed near the top/bottom. The paper states a reference kinematic eccentricity of .9 and also a maximum/minimum speed ratio of 3.205. These are mathematically inconsistent: .9 implies 1/sqrt(1−.9²) = **2.294**, whereas .95 implies **3.203** (approximately the reported 3.205). The current generator explicitly uses .9. We match the reported slowing locations, but cannot claim unambiguous exact modulation-depth replication. This ambiguity should be tested as a prespecified sensitivity analysis, not resolved by choosing whichever gives better human agreement.

**Trial sequence.** The robot brought the hand to the participant-leftmost path point. Motion was counterclockwise from the participant's viewpoint, with a smooth half-cycle acceleration. Participants chose how many cycles to experience before answering. A quarter-cycle deceleration and slow transport followed the answer. The study used adaptive, interleaved ascending/descending staircases with .05 eccentricity steps, starting at ±.70, until at least 15 reversals per staircase. Blocks generally required 40–90 trials. A/B/C block order was randomized across participants. The last 10 reversals from each staircase contributed to participant PSE estimates.

**What was measured.** The PSE is the physical shape at the flat/tall boundary; JND measures discrimination sensitivity. A PSE is not a reconstructed hand path, and a JND is not trajectory RMSE. The mean participant PSEs transcribed from Table 1 are A −.09095, B −.26695, C +.02605. The qualitative finding is perceived stretching along the direction of slowing: horizontal stretching in B and vertical stretching in C. Baseline A already has a small bias. Table-1 means and paired contrasts are not identical to the paper's Figure-3 pooled-median estimator.

### 7.2 What our notebook models, and what it does not

| Feature | Human Experiment 1 | Sections 1–5 | Section 6 |
|---|---|---|---|
| Limb | Human right arm, passively driven | One fixed OpenSim right-arm anatomy | Same model anatomy |
| Path plane | Vertical, frontoparallel | Vertical pipeline X–Z plane | Vertical X–Z circle at each right/forward grid location |
| Center | Right shoulder, ~45 cm from chest | Right-shoulder reference, ground X=.45 m | Centers distributed relative to C7, at reference shoulder height |
| Chest reference | Participant's chest | Torso X=0 approximation; no chest-surface calibration | Grid referenced to C7, not a 45-cm chest plane |
| Size | 160-cm perimeter | 32-cm perimeter, one fifth | 32-cm perimeter; 10.19-cm circle diameter |
| Period / mean speed | 2.5 s / 64 cm/s | 2.5 s / 12.8 cm/s | 2.5 s / 12.8 cm/s |
| Conditions | A, B, C | A, B, C | A only |
| Start / direction | Participant-left / counterclockwise | Matches these geometric conventions | Shared hand start, then approach to each participant-left circle point; counterclockwise |
| Onset and exposure | Smooth half-cycle ramp; participant-selected duration | Three cycles beginning immediately; no paper-matched onset ramp | .5-s hold, 1-s approach, .5-s settle, three cycles, .25-s continuation buffer |
| Analysis | Adaptive flat/tall responses | Last cycle sampled at 150 phases; fixed geometric readout | Third cycle sampled at 150 phases; reach excluded from shape readout |
| Arm extension | Almost complete extension | Substantial elbow bend | More substantial, location-dependent bend |
| Hand orientation | Palm-down maintained | Neutral forearm/wrist coordinates, not verified world palm-down | Same limitation |
| Input to perceptual system | Human afference plus other sensory/central information | Synthetic firing rates from 25 muscles | Same |
| Output | Binary flat/tall decision | Seven regression outputs; predicted joints mapped through OpenSim; separate width/height classifier | Same, plus center/ellipse-fit summaries |

**Coordinates must not be conflated.** Pipeline X is negative participant-right, pipeline Z is up, and pipeline Y is negative forward. OpenSim ground uses X forward, Y up, Z participant-right. “X–Z movement plane” here means the pipeline frame, not native OpenSim ground X–Z. Section 6's body-context map is top-down: its vertical page axis is **forward distance**, not hand height. C7 is the model's base-of-neck reference, not the hand's height or the chest surface.

**Geometry versus dynamics.** IK finds arm joint angles to follow a prescribed hand path. Muscle equilibrium and a coefficient-based transfer function generate synthetic spindle rates. This is not a forward simulation of a robot pulling a relaxed human arm, and does not reproduce grip/contact forces, participant-specific activation, fusimotor control, cutaneous feedback, effort, or a perceptual decision process. The model was trained for movement reconstruction using synthetic spindle inputs derived from human movement recordings, not on these human flat/tall judgments.

**Smaller does not mean fully equivalent.** Shrinking the path fivefold while retaining the period reduces speed and acceleration magnitudes and changes joint excursions and muscle signals. Keeping the same relative slowing locations does not preserve peripheral input statistics or guarantee the same illusion.

### 7.3 Arm audit: what has actually been checked

The saved angle arrays, rather than only the configured posture prior, show:

- Main 21 ellipse trials: elbow flexion **70.49–73.99°** across all frames.
- Fifteen refinement trials: **71.81–74.00°**.
- One hundred grid-circle trials, including reaches: **86.35–117.52°**.

Zero elbow flexion is the model's extended reference. These values establish a mismatch with the paper's almost extended arm. A configured rest angle is an IK regularization target, not an assertion that the arm holds that angle at all times. The model solves shoulder and elbow motion throughout each path.

All active configurations fix `pro_sup`, `deviation`, and `flexion` at 0°. **Zero is a model coordinate setting; it is not evidence of a palm-down hand in world coordinates.** Muscle extraction drives four proximal coordinates, leaving distal coordinates at model defaults. Previous matching of the IK and extraction defaults fixed a coordinate inconsistency but did not validate the human palm-down posture.

The cached maximum hand-path errors are **1.084 mm** in the main experiment, **0.889 mm** in refinement, and **1.233 mm** over the full grid sequences. Fiber lengths are finite and within the engineering screens, without clipping/replacement. Good endpoint tracking does not validate anatomical posture, wrapping at every possible pose, or physiological spindle signals. The earlier multi-meter teres-minor wrapping failure was real; smaller paths avoided that failure but are not a literal reproduction of the study.

**What reproducing the paper's position requires next.** Build a separate experiment with: calibrated chest-to-plane distance; right-shoulder centering; near-extended elbow over the entire path; hand orientation verified from a hand-frame normal and held palm-down; consistent distal coordinates in IK, muscle extraction, and physical decoder readout. Test the full 160-cm path and all eccentricities for reachability, joint limits, wrapping continuity, fiber lengths, and hand orientation before inference. The paper does not specify enough detail to claim an exact participant posture; document anthropometry and any chosen joint-angle targets explicitly. Do not force a visually correct hand path through a failed muscle route. The audit does **not** silently replace the current validated smaller-path experiment with an unvalidated full-size run.

### 7.4 Spindle audit: numerical checks versus physiological validity

The input tensor contains **5 Ia + 5 II responses per muscle × 25 muscles = 250 synthetic signals**, sampled at 240 Hz. Length is normalized as L/Lopt − 1; velocity and acceleration are divided by optimal fiber length. Velocity is smoothed with a 31-sample, first-order Savitzky–Golay filter before the existing transfer function is applied. It combines length, signed velocity raised to a coefficient exponent, acceleration, and a constant offset. Rates are rectified at zero and clipped at channel-specific maxima.

The fresh audit checks every saved rate in all **136 trajectories**, both the full sequence and final-cycle readout. It checks finite values, negative/upper-limit violations, zero occupancy, upper-cap occupancy, quantiles, and distance from the checkpoint's channel-wise training mean/std. It uses the exact coefficient-seed-0 channel ordering. The tables below are computed from cached artifacts, not guessed physiological limits.

Full-sequence rates:

| Experiment | Ia range | II range | Mean zero occupancy Ia / II | Upper-cap occupancy |
|---|---|---|---|---|
| Main ellipses | 0–32.15 Hz | 0–6.26 Hz | 7.76% / 5.23% | 0% |
| Refinement | 0–31.73 Hz | 0–7.40 Hz | 7.69% / 5.39% | 0% |
| Grid circles + reaches | 0–90.75 Hz | 0–8.25 Hz | 18.91% / 11.15% | 0% |

No saved rate is nonfinite or exceeds its channel cap. The selected caps range from **16.17 to 180.31 Hz**, but caps differ by channel and do not constitute a universal physiological reference interval. Zero occupancy is the fraction of channel-time values equal to zero, averaged equally over trajectories; it is not the proportion of muscles permanently silent.

**Training-statistics warning:** in the analyzed grid-circle cycles, about **6.17% of Ia channel-time values are more than 3 saved training SD from their channel mean**, and the largest absolute deviation is about **35.85 SD**. In the main ellipse cycles the corresponding fraction is about **0.106%**, with a maximum of **18.18 SD**. These are descriptive normalization diagnostics, not Gaussian probabilities. They reveal potentially important channel mismatches that an Hz-range check misses. For example, a grid-trial DELT2 Ia channel reaches **90.75 Hz**, compared with saved training mean **2.23 Hz** and SD **2.46 Hz**; a main-ellipse LAT2 Ia channel reaches **28.42 Hz**, versus mean **2.04 Hz** and SD **1.45 Hz**. These examples occur during movement, not only at onset. Channel parametrization and posture therefore require inspection; this is not just evidence of upper saturation. Very small training SD can amplify standardized deviations in other channels as well. Consequently the inputs cannot currently be described as uniformly within the model's training range.

The separate robustness run also recomputed and checked all **five coefficient groups × 136 trajectories = 680 synthetic-input sets**. Every value was finite, nonnegative and within its own channel limit; no group reached upper saturation. The highest rate over these groups was **122.23 Hz**. This extends the numerical cap check to every input parametrization used in the new comparison. The detailed channel-by-channel training-deviation analysis above concerns the original `0_9` input mapping; it is not silently generalized to the other coefficient groups.

**Verdict: finite/cap checks pass; training compatibility and physiological validity remain unresolved.** In particular, the low II rates deserve attention. These checks cannot establish that the signals match human spindle recordings in this passive, palm-down, nearly extended posture. We lack matched human afferent recordings and the full training distribution. Training mean/std comparisons can flag unusual channel values, but cannot establish distributional equivalence or normality. Being within 3 SD is not a physiological acceptance criterion; the distribution is rectified, heterogeneous, and potentially non-Gaussian. Zero output may result from lower rectification; absence of upper saturation does not prove a correct baseline. Audit unrectified transfer values, coefficient units and offsets, static holds, and matching reference movements before declaring rates “in good physiological range.” Do not clip or rescale inputs merely to obtain desired CNN predictions.

### 7.5 Seeds, averaging, and which bias this can reduce

A seed controls a random selection or training process. Repeating inference with identical inputs and the same evaluation-mode checkpoint does not create independent trials. Arbitrarily changing a runtime seed does not create new learned weights or meaningful confidence bars.

There are **five spindle-coefficient groups (0–4)** and **four saved checkpoint replicate folders (0, 1, 2, 9) per group**, hence 20 archived matched variants. Coefficient groups select different spindle parameter/channel combinations. Their CNNs must receive the corresponding channel definitions. Feeding a coefficient-group-1 signal to a group-0 CNN is a channel-identity mismatch, not a clean robustness test.

**Provenance caution:** all four group-0 checkpoint configurations record `seed: 0` internally, although their weight hashes differ. Folder endings are therefore reported as **replicate IDs**, not verified independent training random seeds. Archive provenance is required to establish exactly which training randomness changed. The five coefficient choices and four checkpoints form a crossed comparison; they are not 20 independent human participants.

**New comparison policy.** Use every available matched variant with equal weight, identical saved arm trajectories, the existing `replicate` temporal padding, the same preprocessing, the same final-cycle windows and the same geometric classifier. Keep the original `0_9` result separate. Check that the repeated `0_9` predictions reproduce the cached outputs. Do not choose variants based on human agreement. Report each variant, average continuous metrics, SD/range across variants, and label frequencies. For PSE, calculate each variant's crossing first, then summarize valid crossings and count missing/ambiguous ones. Averaging trajectories before applying a nonlinear shape label answers a different question and can hide disagreement; it is not used as the default result.

This reduces dependence on a selected checkpoint and selected spindle parametrization. **It does not guarantee less systematic prediction bias**, correct the bent-arm mismatch, make synthetic rates physiologically valid, or turn model spread into a human population confidence interval. Report coefficient-group means and within-group spread so that pooling does not hide structure. Human participants remain the statistical units for human CIs. Repeated phase samples are correlated, and repeated deterministic cycles are not independent subjects.

**Temporal-history clarification.** The full reach is supplied to the CNN, but the selected architecture has four stride-1 temporal convolutions of width 7 followed by a per-time-step linear readout: a 25-frame (~104-ms) receptive field, not persistent recurrent memory. The reach is many seconds before the third circle. Under the current feed-forward model it is outside that readout's direct receptive field; preprocessing adds local filter context, not multi-second memory. The earlier conversational suggestion that the approach might directly influence third-cycle predictions through long-term CNN history was too broad. Physical posture at the circle still matters. Endpoint padding can affect the final samples of the ellipse runs, which lack the extra grid buffer; keep this limitation separate from a reach-history effect.

### 7.6 Interpretation and next experimental decisions

The original single CNN has a strong baseline vertical shape bias, and the velocity-dependent PSE shifts are much smaller than human shifts. Its grid circles show position-dependent distortion and rightward decoded center bias. These are model outputs, not observations of human experience. A wider checkpoint comparison tests robustness, not biological truth.

Priority for reducing avoidable experimental bias: (1) make the passive-arm posture and hand orientation defensible; (2) verify synthetic spindle baselines and training compatibility without tuning to the desired perceptual answer; (3) report all matched checkpoint variants rather than a favorable subset; (4) define the classifier and uncertainty measures before comparing human data; (5) test size, speed, and modulation-depth mismatches separately. Matching human results by changing all factors simultaneously would not identify the cause.

Reproduction status: **plane/start/direction partly aligned; full size/speed, almost extended arm, maintained palm-down posture, adaptive human judgments and onset protocol are not reproduced.** This audit establishes those limits; it does not claim they have been corrected.

### 7.7 Completed all-checkpoint results: what survives averaging?

All **20 matched variants × 136 trajectories = 2,720 predictions** were evaluated with unchanged saved arm motions. Checkpoint `0_9` reproduces the cached predictions within float tolerance; its grid labels and physical RMSEs match exactly. No variant produced predicted proximal joints outside the model's limits in the analyzed samples. This is another numerical screen, not proof of physiological correctness.

**Grid location bias is not robust.** The selected `0_9` variant has mean rightward center bias **+0.635 cm**. The other **19 variants each have a negative mean rightward bias**, i.e. a leftward mean bias. Across all 20, the equal-weight mean is **−2.332 cm**, with across-variant SD **1.243 cm**. This is the average of the variants' mean signed errors over the 100 locations, not proof that every location shifts left in every variant. Even the same-coefficient group 0 changes bias direction across checkpoints, so the discrepancy is not solely caused by changing spindle coefficients. Agreement of `0_9` with the supplied reaching figure cannot be generalized to the entire checkpoint set.

**Averaging does not remove reconstruction error.** Mean per-variant RMSE over the main 21 ellipse trials is **1.421 ± 0.400 cm** (across-variant SD). Over the 100 grid circles it is **4.325 ± 1.097 cm**, with per-variant grid means ranging **2.057–7.199 cm**. These are averages of individual models' error metrics, not error of a trajectory obtained by averaging predictions. The latter is a distinct ensemble decoder and is not silently substituted for the existing CNN.

**Velocity effects are not robust either.** Mean PSEs across variants with a unique tested crossing are A **+.364** (20/20), B **+.391** (19/20), C **+.358** (19/20). One B and one C crossing could not be uniquely estimated from the tested grid. Do not replace them with zero or silently ignore the missing count. Across 19 variants with each paired contrast available, mean shifts are B−A **+.030** (SD .131) and C−A **−.0003** (SD .085). These do not reproduce the human mean shifts B−A **−.176**, C−A **+.117**. The original `0_9` directional tendency is therefore checkpoint-dependent.

**What “less bias” means here.** Using all available variants makes the analysis less dependent on choosing one favorable checkpoint. It did not make the model's signed errors vanish or improve its agreement with humans. We should report that disagreement, then test posture, size/speed, and spindle compatibility in separate controlled experiments. Existing evidence does not establish a human-like perceptual illusion in this CNN family under the current simulation settings.

**New-figure guide.** Thin shape curves are individual variants; bold curves and shaded bands show equal-weight means ± SD. PSE dots show each valid variant; red summaries use SD across variants and black human bars use 95% confidence intervals. These are intentionally different uncertainty measures. The RMSE panel shows each variant's mean error. The grid-bias panel averages signed center error at each location: positive means predicted too far right. Gray zero lines indicate no signed bias, not a physiological threshold. None of the new model-spread bars are participant confidence intervals.

**Reproducibility:** run `process/audit_viviani_setup.py` for cached signal checks (Torch environment), `process/audit_arm_snapshots.py` for pose samples (OpenSim environment), and `process/plot_viviani_audit.py` for the audit figures (Torch environment), then `process/checkpoint_sensitivity.py inference` (Torch), `process/checkpoint_sensitivity.py readout` (OpenSim), and `process/summarize_checkpoint_sensitivity.py` (Torch). Saved CSVs, checkpoint hashes, preprocessing choices and figures are under `outputs/research_audit/`. All original single-checkpoint artifacts remain in their existing folders. Checkpoint comparison is restricted to the available archive; it is not a random sample of all possible trained models.
