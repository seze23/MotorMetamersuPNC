from pathlib import Path
import nbformat
root=Path.cwd();n=nbformat.v4.new_notebook();cells=[]
def md(s):cells.append(nbformat.v4.new_markdown_cell(s))
def code(s):cells.append(nbformat.v4.new_code_cell(s))
md('''# Horizontal elliptical trajectories: Soechting & Poizner (2005)

Separate experiment based on **The use of motion cues in the haptic sense of circularity**, *Experimental Brain Research* 165:413–421, DOI **10.1007/s00221-005-2316-8**. Source: [supplied paper](/Users/macbookair2026/Downloads/s00221-005-2316-8%20(1).pdf).

**Scope warning:** the published 15-cm circle radius is retained. Eighteen attempted trials at α=−1, −.75 and −.5 failed muscle-continuity screens and are excluded before inference. The active set is **48 trajectories × 20 variants = 960 predictions**, α=−.25 to +1 (AR=.8–2). This restricted shape coverage is not a complete replication.

**Purpose:** test whether the 20 available matched CNN variants reproduce the paper's lack of a systematic speed-induced circularity bias. Original Viviani notebooks and results are preserved. No CNN weights are trained or changed. This is a kinematic/spindle-only simulation, not a complete human haptic model.
''')
md('''## 1. Human experiment: setup, procedure, and findings

- **Participants:** six per experiment; eight unique people in total, with four participating in both. The robot moved the **dominant arm**; all but two participants were right-handed. Our model is a single right-arm anatomy, not eight simulated participants.
- **Plane:** horizontal. Paper x = medio-lateral (wide); paper y = anterior–posterior (narrow). Here plots use participant-right and forward. Pipeline X–Y is horizontal, with both signs reversed to obtain these participant coordinates; pipeline Z is height. This is not native OpenSim's X–Y plane.
- **Arm/elbow position:** both shoulder and elbow contributed to movement. The paper does **not** report exact elbow/shoulder angles, hand height, torso-to-center distance, seating geometry, or a palm-down constraint. Do not copy Viviani's nearly extended arm or 45-cm chest distance into this protocol as if specified here.
- **Apparatus:** a lightweight, two-joint robot moved a grasped handle (10 cm × 4 cm diameter), free to rotate about the vertical axis. Subjects closed their eyes. A spring/damper force field generated passive displacement; stated stiffness was 10 N/cm and damping 0.5 N per (cm/s). Actual position, velocity, force, response and response time were recorded at **200 Hz**.
- **Geometry:** x=A sin θ, y=B cos θ; AR=A/B (operational definition in Methods, which can exceed one). **Area was fixed**, not perimeter: AB=15² cm². A circle has radius **15 cm**, diameter **30 cm**, circumference **94.25 cm**. The attempted shape grid covered AR=.5–2; the active, screened grid covers AR=.8–2; its extreme diameters are about 21.21 × 42.43 cm, swapped between orientations.
- **Shape measure:** α=AR−1 for AR≥1, α=1−1/AR for AR<1. Positive = wide, negative = narrow, zero = circle. This is **not Viviani's signed eccentricity**; their numerical values and thresholds cannot be compared directly.
- **Starting position:** each trial was preceded by a slow **5-s transport** to x=0, y=B, followed by a warning tone. Equations 2–3 begin there, moving toward positive x. With the right/forward convention used here this is the forwardmost point, initially moving right, clockwise from above. The paper does not separately describe a body-relative common home position.
- **Conditions:** **N (Normal):** constant phase rate; conforms to the two-thirds power law, so circles—but not noncircular ellipses—have constant speed. **FS:** faster on the sides; a wide ellipse is approximately constant-speed. **FT:** faster at top/bottom; a narrow ellipse is approximately constant-speed. They are not Viviani's A/B/C profiles: here angular modulation is held fixed across shape, while tangential speed depends on the ellipse.
- **Experiment 1:** period **4 s**, three cycles (12 s); Fourier coefficients c₁=±.0991, c₂=.0118. Approximately constant-speed reference ARs are 1.5 and 2/3. For a circle, distorted min/max speed ≈.68; normal circle speed =23.56 cm/s.
- **Experiment 2:** period **2.5 s**, five cycles (12.5 s); c₁=±.1558, c₂=.0321. Reference ARs are 2 and 1/2; distorted circle min/max speed ≈.54; normal circle speed =37.70 cm/s. The Fourier approximation does not produce perfectly constant speed at every reference ellipse.
- **Speed equation:** θ=ωt+c₁sin(2ωt)+c₂sin(4ωt). For our x=A sinθ, y=B cosθ convention, c₁ is negative in FS and positive in FT. We verify this using speed extrema, not the condition name alone.
- **Responses:** wide versus narrow, not a three-class task. Three blocks of 60 trials per participant; randomly interleaved ascending/descending staircases started at AR=2 and .5. Initial α steps were 1/8; direction and step size adapted after response changes. The final 30 trial α values were averaged to estimate circularity bias. Movement could end after the prescribed cycles or the response. We simulate complete trajectories, not this adaptive decision process.
- **Human bias findings:** pooled across both experiments, mean α was N=.001, FS=.028, FT=.012. Explicit experiment-2 means were **N=−.019, FS=.030, FT=−.002**; the condition effect was not significant (reported F(2,15)=.159, p=.85). This is evidence of no detected effect in this design, not proof that speed can never affect proprioception.
- **Sensitivity:** pooled bias-corrected choices were fitted with a logistic function. The paper defines difference threshold as **2β**, spanning probabilities .27–.73, not a .25–.75 JND. Figure 4 β values are experiment 1: .077/.054/.029 and experiment 2: .040/.113/.061 for N/FS/FT. Experiment-2 FS sensitivity was worse than N; the paper reports bootstrap intervals for β. These are not trajectory errors.
- **Other distortions:** participants informally described axes tilted by about 30° and asymmetric shapes. These were **anecdotal reports, not quantified illusion measurements**. The authors related them to tilted/asymmetric interaction-force patterns. Our spindle-only CNN does not receive tactile grip forces or simulate the robot's viscoelastic field.
- **Source cautions:** the abstract's minor/major AR wording differs from the signed wide/narrow operational AR=A/B in Methods; we implement Methods. One Results sentence repeats “FS” for the third condition; the condition order/Figure 4 identify it as FT. Human individual numerical responses are unavailable; no fabricated human trials or confidence intervals are used below.
''')
cells[-1].source += '\n\n'+(root/'notebooks/data/soechting_alpha_explanation.md').read_text() + '\n\n### Side view: shoulder, elbow, torso, and hand\n\nThe solid arm below is the saved **starting pose of the Experiment-2 normal-speed circle**, not a drawing of a measured participant. Faint arms show later positions. The neck reference is C7. The circle center is assumed to be **35 cm forward and 30 cm below the reference shoulder**. The hand begins at the circle’s forwardmost point, another 15 cm forward of its center. From the side, the horizontal circle looks like a flat line. The torso outline is schematic; the paper does not provide these body-relative distances.\n'
code('''from pathlib import Path
import os,sys,json,subprocess,hashlib
import numpy as np
import pandas as pd
from IPython.display import display,Image
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'run_pipeline.py').exists())
OUT=ROOT/'outputs/soechting_2005'
OSIM=ROOT/'.venv-opensim/bin/python'
TORCH=ROOT/'.venv-mujoco/bin/python' # Used for Torch/plots; no MuJoCo arm is used.
CONFIG=ROOT/'experiments/soechting_2005.yaml'
RUN_BIOMECHANICS=False
RUN_ALL_CNNS=False

def run(python,args,log):
    env=dict(os.environ,MPLBACKEND='Agg',MPLCONFIGDIR=str(OUT/'.matplotlib'),CENTEROUT_WORKERS='4',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (OUT/log).open('w') as handle:
        subprocess.run([str(python),*args],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
if RUN_BIOMECHANICS:
    run(OSIM,['-c',"from run_pipeline import Pipeline; Pipeline('experiments/soechting_2005.yaml').run_stage('path')"],'path.log')
    run(OSIM,['process/soechting_ik.py'],'ik.log')
    run(OSIM,['-c',"from run_pipeline import Pipeline; p=Pipeline('experiments/soechting_2005.yaml'); [p.run_stage(s) for s in ['motion','muscle-lengths','muscle-signals']]"],'biomechanics.log')
    run(TORCH,['process/soechting_analysis.py','quality'],'quality.log')
    run(OSIM,['process/soechting_arm_setup.py','extract'],'setup_schematic_extract.log')
    run(TORCH,['process/soechting_arm_setup.py','plot'],'setup_schematic_plot.log')
stimuli=pd.read_csv(OUT/'stimuli.csv')
print('Shape/speed trials:',len(stimuli),'— published radius retained; no downscaling.')
display(stimuli[['study_experiment','condition','alpha','AR','A_cm','B_cm','period_s','cycles']].drop_duplicates().head(12))
display(Image(filename=str(OUT/'figures/stimuli.png')))
display(Image(filename=str(OUT/'figures/speeds.png')))
display(Image(filename=str(OUT/'figures/side_arm_setup.png')))
''')
md('''## 2. OpenSim → muscle lengths → synthetic spindle firing: checks and limits

**Pipeline:** prescribed horizontal hand path → bounded OpenSim IK → standard `.mot` motion → the existing 25-muscle extraction/equilibrium routine → the existing coefficient-based spindle transfer function. Input arrays are sampled at **240 Hz**, matching the CNN; the human recordings were at 200 Hz. This is deliberate sample-rate compatibility, not resampling human recordings.

**Placement assumption, not a fact from the paper:** center at participant-right 0 cm and forward 35 cm relative to the reference right shoulder, **30 cm below shoulder height**. The torso is fixed. Wrist and forearm model coordinates remain neutral (0°), consistent in IK, extraction and decoder readout. Neutral coordinates do not establish any specific world palm orientation. Shoulder/elbow angles are solved and reported below, not borrowed from Viviani. A shoulder posture prior and a minimum 10° shoulder elevation select a continuous IK branch; neither was fitted to human circularity data or CNN agreement.

**Why the IK implementation is explicit:** four proximal coordinates can place one hand marker in multiple postures. Initial unconstrained solutions showed branch changes and muscle-length failures. We solve a periodic, spatial joint-angle loop using the same OpenSim model and exact couplers, regularizing joint curvature and posture, then interpolate it to each speed profile. Every resulting hand frame is rechecked against the requested path. The original motion, muscle-equilibrium and firing-rate stages are retained; no wrapping surfaces or weights are edited. This phase-cache solver is an alternative IK implementation, not an unchanged run of the default IK tool.

**Exposure differences:** the model receives three/five full cycles plus a .25-s continuation buffer. The last full revolution is read out every fourth sample. The paper's 5-s preparatory transport is not modeled because its origin is unspecified; onset and early-response termination are not reproduced. The late readout is outside the finite CNN receptive field of the onset, and the continuation buffer protects it from end-padding/filter artifacts.

**Rejected inputs:** of the 66 attempted full-size trajectories, 18 (α=−1, −.75, −.5 in all six experiment/condition combinations) had abrupt fiber-routing jumps exceeding 18,000 mm/s. Their endpoint tracking and absolute length bounds alone looked acceptable. A representative rejected TMIN sequence alternates around 213, 61, 214, 215 and 66 mm while the joints change smoothly; it is not credible muscle motion. They were rejected **before CNN inference**, with the same rule across conditions. A second lower-workspace attempt failed hand tracking and was not used. The final accepted data retain 30-cm circles; nothing is silently downscaled. Failed inputs remain available in `attempted_quality.csv`, separate from the active manifest. A model PSE outside the remaining tested interval cannot be estimated and is not extrapolated.

**Acceptance screens:** finite positions, positive finite fiber lengths; ≤2-mm hand error at every frame; normalized fiber length .1–3; peak raw fiber speed <500 mm/s; intended horizontal shape and height inspected. These are conservative engineering checks, **not human normative intervals**. The saved input joint-angle limits are checked, and actual angles/ranges are displayed. Lengths are not clipped, repaired or substituted to pass.

**Spindles:** 25 muscles × (5 Ia + 5 II) =250 synthetic signals. The existing optimal-length normalization, 31-sample first-order Savitzky–Golay velocity smoothing, acceleration term, zero rectification and channel-specific upper clipping are retained. All five matched coefficient groups are audited. We report zero/cap occupancy and each checkpoint's channel-wise standardized input deviations. Bounded rates alone do not demonstrate human physiological plausibility. Force/contact, skin afferents, voluntary activation and fusimotor control are not independently modeled.

**Interpretation rule:** numeric simulation failures stop inference. Large deviations from training statistics are disclosed as an extrapolation warning; they are not hidden by arbitrary rescaling. Passing all engineering screens does not certify the signals as human spindle recordings. Very small tracking residuals reflect numerical consistency inside the model, not micrometer-level knowledge of a human hand or physical robot. The paper supplies no simultaneous spindle data for validation.
''')
code('''attempted_path=OUT/'attempted_quality.csv'
attempted=pd.read_csv(attempted_path if attempted_path.exists() else ROOT/'notebooks/data/soechting_2005_rejected_input_audit.csv')
print('Attempted input screens:',attempted.passed.value_counts().to_dict())
display(attempted[~attempted.passed][['id','max_tracking_error_cm','max_normalized_fiber','peak_fiber_speed_mm_s']])
quality=pd.read_csv(OUT/'quality.csv')
assert quality.passed.all(), 'Stop: failed simulation input screens.'
display(quality[['id','max_tracking_error_cm','max_height_drift_cm','min_normalized_fiber','max_normalized_fiber','peak_fiber_speed_mm_s','elbow_min_deg','elbow_max_deg']])
display(Image(filename=str(OUT/'figures/arm_and_tracking.png')))
rates=pd.read_csv(OUT/'spindle_quality.csv')
display(rates.groupby(['coefficient_seed','type']).agg(min_Hz=('min_Hz','min'),max_Hz=('max_Hz','max'),mean_zero_fraction=('zero_fraction','mean'),mean_upper_cap_fraction=('upper_cap_fraction','mean')))
training=pd.read_csv(OUT/'training_range.csv')
display(training.groupby('variant').agg(mean_fraction_abs_z_over_3=('abs_z_over_3_fraction','mean'),max_abs_z=('max_abs_z','max')))
display(Image(filename=str(OUT/'figures/quality.png')))
''')
md('''## 3. All 20 matched CNN weight sets and the overall prediction

- Use **all five coefficient groups × all four saved replicate IDs (0,1,2,9)**. Every CNN receives its matching spindle-channel definitions and its own saved training normalization. No model is chosen because it agrees with the paper.
- These share a network design but have different learned weights. The archive metadata does not fully verify independent training seeds; “20 variants” is more accurate than “20 independent subjects.” Variation is structured by coefficient group and replicate.
- Each network regresses seven quantities; it does not directly classify wide/narrow. We map its four predicted proximal angles through the **same OpenSim arm**, then measure the physical horizontal path. Predicted angles outside model limits are flagged; no angle clipping is used.
- **Overall prediction:** convert each variant to physical XYZ first, align identical time samples, then average the **20 hand trajectories point by point**. We do not average spindle arrays across unmatched channel definitions or average joint angles before the nonlinear physical transform.
- Report both the mean-path shape and each variant's shape/PSE. Shape of the averaged path can differ from average shape across variants; cancellation can make a mean path look better than its component predictions.
- Thin lines below are individual decoded paths; bold color is the mean of 20; black is the actual circle. All paths use the same actual-center reference, so location offsets remain visible. Model-spread SD is not a human confidence interval or a trial-noise estimate.
''')
code('''if RUN_ALL_CNNS:
    run(TORCH,['process/soechting_analysis.py','inference'],'inference.log')
    run(OSIM,['process/soechting_analysis.py','readout'],'readout.log')
    run(TORCH,['-m','process.soechting_report'],'report.log')
provenance=json.loads((OUT/'provenance.json').read_text())
assert provenance['n_models']==20
assert provenance['config_sha256']==hashlib.sha256(CONFIG.read_bytes()).hexdigest(), 'Stale output: configuration changed.'
base=ROOT/'trained_models/experiment_causal_flag-pcr_optimized_linear_extended_5_5_letter_reconstruction_joints'
for variant in provenance['variants']:
    checkpoint=base/f"spatiotemporal_4_8-8-32-64_7171_{variant['variant']}"/'model.ckpt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==variant['checkpoint_sha256']
predictions=pd.read_csv(OUT/'all_predictions.csv')
ensemble=pd.read_csv(OUT/'ensemble_predictions.csv')
assert len(predictions)==len(stimuli)*20
print('Completed predictions:',len(predictions),'from',provenance['n_models'],'matched weight sets.')
print('Fraction of decoded frames outside proximal joint limits:',predictions.outside_joint_fraction.mean())
display(ensemble[ensemble.alpha==0][['study_experiment','condition','predicted','predicted_AR','ensemble_rmse_cm','right_bias_cm','forward_bias_cm']])
display(Image(filename=str(OUT/'figures/circle_predictions.png')))
''')
md('''## 4. CNN results, human results, and comparison

**Main comparison:** human circularity bias uses the paper's α units. The model proxy is the input α at which a decoded path's right/forward extents become equal. We interpolate a unique zero crossing and retain the adjacent tested-α bracket; missing or multiple crossings are reported as unavailable. A width/height ratio alone does not prove a smooth ellipse or circle.

**Human numbers:** experiment-2 means are explicitly reported. Experiment-1 mean markers are **derived from rounded published numbers** as 2×pooled mean − experiment-2 mean, assuming the equally sized experiments contribute equally; they are approximate, not participant-level reanalysis. We do not construct human PSE CIs without individual data. Four people overlap across experiments, so there are not 12 independent people.

**Graph guide:**
- Shape curves: x = actual α; y = mean individual decoded α; band = across-variant SD. Dashed diagonal = accurate geometric reconstruction. Each experiment is separate.
- Circularity panels: black = human mean; colored dots = each variant's valid PSE; red diamond = PSE of the pointwise mean physical trajectory; its red bar is the tested-α crossing bracket, **not a confidence interval**. These are distinct estimators.
- Sensitivity: published **human 2β** only; available experiment-2 N/FS bootstrap intervals are doubled from the reported β intervals, and missing intervals are not invented. No matched CNN psychometric JND has been established; variation across models is not an interchangeable substitute.
- Reconstruction error: 3D RMSE of the **mean decoded trajectory**, not average individual RMSE. It includes location, shape, timing and out-of-plane error. The summary reports both types of RMSE.
- Tilt: fitted major-axis orientation for physically wide α=.5 paths with fitted axis ratio >1.1. Fits with large residuals remain unreliable; the paper's informal “about 30°” report is not a numerical population benchmark.
- Height-error panels: actual motion is horizontal; departures from the dashed zero line are reconstruction errors, not measured human illusions. Horizontal shape projections can hide these errors.
- Center arrows: **dot = actual center; arrow tip = mean CNN prediction**. Right-pointing means rightward prediction bias. There is no positional grid in this experiment.

A lack of speed-dependent human bias does not mean no human sensitivity or no haptic illusion at all: the paper separately reports sensitivity changes and anecdotal tilt/asymmetry. Matching one near-zero contrast alone does not establish a match of baseline shape, posture, sensory signals, or decision process.
''')
code('''comparison=pd.read_csv(OUT/'human_model_comparison.csv')
display(comparison[['study_experiment','condition','mean_alpha','mean_source','ensemble_pse','ensemble_bracket_low','ensemble_bracket_high','mean_variant_pse','sd_variant_pse','n_unique_crossings']])
display(Image(filename=str(OUT/'figures/human_comparison.png')))
display(Image(filename=str(OUT/'figures/sensitivity_error_tilt.png')))
display(Image(filename=str(OUT/'figures/center_bias.png')))
display(Image(filename=str(OUT/'figures/out_of_plane.png')))
print('Mean decoded out-of-plane excursion across individual paths (cm):',round(predictions.height_excursion_cm.mean(),2))
summary=json.loads((OUT/'summary.json').read_text())
print('Mean RMSE of individual variants (cm):',round(summary['mean_individual_rmse_cm'],3))
print('RMSE averaged across mean-path reconstructions (cm):',round(summary['ensemble_mean_rmse_cm'],3))
''')
md("### Do speed conditions affect the CNN? Exploratory grouped tests\n\n**Correction to the earlier qualitative interpretation:** similar mean-trajectory curves do not establish that the model is insensitive to speed. Averaging paths can cancel individual changes. The following analysis uses the **circularity proxy of each checkpoint**, a different estimator from the circularity point of the averaged physical trajectory.\n\n**Question:** is there a systematic N/FS/FT difference in the checkpoint circularity proxies? Experiment 2 is the primary comparison of interest; Experiment 1 is secondary. The 20 checkpoints are not treated as 20 independent human subjects. Within each of the **five spindle-coefficient groups**, average the four checkpoint replicate PSEs. Then enumerate all **6⁵ = 7,776** within-group permutations of the three condition labels. The statistic is the sum of squared, group-centered condition means. Each group is permuted as a unit, retaining its four-checkpoint average. Exact p is the fraction of permutations at least as extreme as the observed statistic, including the observed assignment.\n\n| Protocol | Exploratory condition-effect p | Holm adjustment if testing both protocols as one family |\n|---|---:|---:|\n| Experiment 1 | .87963 | .87963 |\n| Experiment 2 | .01929 | .03858 |\n\n**Interpretation:** conditional on this test's assumptions, Experiment 2 shows evidence of speed-condition sensitivity. Experiment 1 does not provide detected evidence of a shift; that is not proof of equivalence. We should not claim “no CNN speed effect” from the mean curves alone.\n\n**Effect sizes, checkpoint PSE estimator:** in Experiment 2, FS−N is **−.01746 α**, and FT−N is **+.02621 α**, averaging all 20 paired checkpoint differences equally. Their across-checkpoint SDs are .05111 and .03411. By contrast, the *mean-trajectory* PSE shifts are +.00638 and +.00470. Nonlinear reconstruction/shape extraction means these estimators need not agree, even in direction. Small mean-trajectory shifts can conceal checkpoint sensitivity.\n\n**Pairwise tests:** two-sided exact sign-flip tests on the five group-mean differences give Experiment-2 p=.3125 for FS−N and p=.125 for FT−N; Holm values for those two contrasts are .3125 and .25. With only five groups, a two-sided sign-flip p cannot be smaller than .0625. An omnibus permutation test over all three conditions and a low-resolution pairwise sign-flip test are different tests; the omnibus result does not establish either individual contrast at p<.05.\n\n**Baseline is a different hypothesis:** testing normal-condition circularity proxy against zero gives group sign-flip p=.0625 in Experiment 1 and p=.375 in Experiment 2. These do not erase the observed positive mean proxies; they reflect a small, heterogeneous model set and the test assumptions. They are not the requested speed-condition test.\n\n**Material limitations:** exchangeability of N/FS/FT labels within groups and independence of the five groups are assumptions, not actual randomized model trials. Coefficient selections can overlap, all models share a training source/architecture, and independent training seeds are not verified. Consequently these are **conditional exploratory model-sensitivity p-values**, not confirmatory human-population p-values. The deterministic averaged decoder has no repeated-trial error distribution. The alpha grid is coarse, asymmetrically restricted and uses interpolated crossings; the tested brackets are not statistical CIs. Do not compare these p-values directly with the paper's human ANOVA p=.85. No equivalence margin was specified, so this analysis cannot establish that an effect is negligibly small.\n\nNo new trajectories, spindle inputs, learned weights or CNN predictions were generated for this analysis. Code: `process/soechting_statistics.py`; cached tables and method notes: `outputs/soechting_2005/statistical_*.csv` and `statistical_method.json`.\n")
code("# Conditional exploratory statistics on existing per-checkpoint circularity proxies.\nrun(TORCH,['process/soechting_statistics.py'],'statistical_tests.log')\nspeed_tests=pd.read_csv(OUT/'statistical_tests.csv')\npaired_tests=pd.read_csv(OUT/'statistical_contrasts.csv')\ndisplay(speed_tests[['study_experiment','n_coefficient_groups','n_checkpoints_per_group','speed_condition_p','holm_p_if_both_experiments_family']])\ndisplay(paired_tests[['study_experiment','contrast','mean_paired_shift','sd_shift_across_20_variants','paired_sign_flip_p','holm_p_two_contrasts']])\nprint('Exploratory model tests: exchangeability and independent-group assumptions are not established by the archive.')\n")
md('''## 5. Conclusions, possible biases, and what remains uncertain

- **Mean curves are insufficient to establish insensitivity:** mean-path PSEs cluster around +.074 to +.083, but the separate grouped analysis of checkpoint PSEs flags Experiment-2 speed sensitivity (exploratory p=.01929; p=.03858 if Holm-adjusting both protocols). Experiment 1 gives p=.87963. These depend on unverified exchangeability/independence assumptions and coarse interpolated PSEs. They do not establish a human-like null effect or equivalence.
- **Baseline model bias:** actual circles are classified as slightly **narrow/forward-elongated** in all six mean-path cases. The model needs a slightly wide physical ellipse (AR about 1.08) to yield equal decoded widths. Human mean circularity points lie closer to zero; exact agreement is not established.
- **Location bias:** the mean decoded circle centers are about **1.3–1.6 cm too far right** in this assumed workspace. The paper did not measure circle-center localization, so this is not a corroborated human illusion.
- **Averaging helps reconstruction but can hide disagreement:** mean-path RMSE averaged over the accepted trials is about **2.88 cm**, versus **3.45 cm** for mean individual-model RMSE. Individual variants remain visibly different.
- **Out-of-plane error matters:** individual decoded paths have an average vertical excursion of about **6.36 cm**, although the actual hand stays in a horizontal plane. Good-looking horizontal projections therefore do not demonstrate correct 3D perception.
- **No validated tilt illusion claim:** fitted tilts are model diagnostics; the human tilt reports were anecdotal, and contact-force cues are absent from this simulation.
- **Scope:** 18 failed extreme-narrow trials are excluded; accepted inputs pass engineering checks, not physiological certification. About .63% of accepted input values exceed 3 saved training SD on average; maximum deviation is about 8.34 SD. All analyzed decoded proximal joint angles remain within model limits.

The following generated conclusion reflects the saved results, rather than assuming that the CNN reproduces the paper. A decoded shape or center bias is a **model prediction**, not evidence that humans experience the same percept.

This is **not a controlled change of plane alone** relative to the Viviani notebook: path size, speed law, placement, timing and human task details also differ. Differences between notebook results cannot be attributed to the movement plane without a separate matched-factor experiment.

Unresolved limits: unknown human posture/height/workspace; one right-arm anatomy versus dominant-arm human testing; synthetic spindle signals and training-distribution mismatch; no robot interaction-force/skin cues; no human-like adaptive decision rule; approximate geometric PSE readout; sparse and asymmetrically restricted α-grid interpolation, with missing-crossing counts reported; structured model variants rather than independent human samples; qualitative rather than quantitative human tilt reports. The model's quarter-second end buffer and fixed complete-cycle analysis also differ from response-terminated human trials.

The 20 checkpoints are not retrained or calibrated to this paper. Better agreement of the mean trajectory can arise through cancellation of individual errors; it should be assessed alongside the individual traces and spread. A flat condition contrast is not enough to claim “no bias” if all conditions share a substantial baseline bias.
''')
code('''print('Input engineering checks passed:',summary['all_input_screens_pass'])
print('Maximum hand error (mm):',round(summary['max_tracking_error_mm'],3))
print('Fiber range (mm):',summary['fiber_range_mm'])
print('Maximum firing rate (Hz):',round(summary['max_rate_Hz'],2))
print('Mean fraction of input values >3 training SD:',round(summary['mean_abs_z_over_3_fraction'],4))
for exp in [1,2]:
    c=comparison[comparison.study_experiment==exp].set_index('condition')
    print(f"Experiment {exp}: human circularity α N/FS/FT =",c.loc[['N','FS','FT'],'mean_alpha'].round(3).tolist())
    print('  Mean-trajectory CNN circularity α =',c.loc[['N','FS','FT'],'ensemble_pse'].round(3).tolist())
    if c.ensemble_pse.notna().all():
        print('  CNN shifts relative to N: FS−N =',round(c.loc['FS','ensemble_pse']-c.loc['N','ensemble_pse'],3),'FT−N =',round(c.loc['FT','ensemble_pse']-c.loc['N','ensemble_pse'],3))
    else:
        print('  At least one model circularity crossing is not uniquely estimable on this grid.')
print('Circle labels of the mean trajectory:')
display(ensemble[ensemble.alpha==0][['study_experiment','condition','predicted','predicted_alpha','right_bias_cm','forward_bias_cm']])
print('Read the individual-variant spread and input warnings before interpreting agreement as human-like perception.')
print('All artifacts:',OUT)
''')
n.cells=cells;n.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}}
nbformat.write(n,root/'notebooks/soechting_2005_horizontal_ellipses.ipynb')
print('Built',len(cells),'notebook cells')
