# Results and limitations

## 31 October 2026 — Shared noise does not reliably improve the full stream

All eight seed pairs complete (16 full269-frame expert-input streams). Source/config/array hashes verify and every stream metric recomputes exactly in the original model environment. Prior generated references match with maximum difference0.

Across seeds, independent→shared near-contact position mean is20.28→21.57mm (3 improve,5 worsen); expert-relative increment mean is21.10→20.90mm, only0.95% lower on average. Boundary increment residual is23.89→23.34mm; within-chunk residual is20.17→20.09mm. Mean per-run full position p95 rises161.36→167.42mm, jaw errors4.50→4.625frames and jaw transitions2.0→2.5. Orientation p95 improves3.73→3.42degrees on average but worsens in3 pairs. Native decoder rejection is zero throughout. Startup0..39 position mean is141.62→144.50mm. These are predicted-command statistics, not robot motion.

Do not adopt shared noise or select a favorable seed for a success claim. This particular correlation scheme is not a reliable fix. The evidence does not rule out other sampling methods or prove the cause of physical failure; stop extending this sampling sweep and retain the original10-step independent-noise behavior.

A read-only supervision audit confirms uniform observation-anchor sampling with20 future targets, while live execution consumes four before replanning. Over one pass through all269 anchors, the first four relative offsets contribute1,070 of5,190 valid label occurrences (20.62%). Absolute action0 appears once, action19 twenty times. These are structural label counts, not exact optimizer-gradient weights, and long-horizon supervision is intentional rather than a code defect. Next proposed bounded training ablation emphasizes the executed prefix while preserving longer-horizon learning, and audits startup/near-contact, jaw and pose performance together. It is a hypothesis, not an established remedy; no new training or physics is launched here. [Source-hashed evidence](../../results/summaries/2026-10-31-ball-noise-replanning-result.json).


## 31 October 2026 — Noise replanning implementation

The [full-episode diagnostic](BALL_NOISE_REPLANNING.md) has CPU checks for provenance, exact model normalization, terminal padding, paired seeds and increment/jaw accounting. It reuses the verified10-step sampler and checks previous generated references during GPU execution. GPU results remain pending; no prediction or grasp improvement is established.

## 31 October 2026 — More denoising steps worsen this matched probe

The fixed-input diagnostic completes at12 anchors with8 matched draws per condition. All recorded hashes verify and pooled position decompositions/action metrics recompute exactly. Identical-noise repeats and archived parent predictions match exactly (maximum difference0). This supports genuine noise sensitivity rather than nondeterminism for these checked calls.

For the20 sampled target offsets in frames80..119,10-step empirical mean bias/spread/total RMS are14.97/17.93/23.36mm;30-step values are15.61/21.56/26.62mm. Total RMS increases13.97%; mean synchronized inference time rises89.9→211.7ms (2.36x, first-call warmup included). Both have zero native decoder rejects. These sparse RMS statistics differ from earlier dense-window means. Do not promote30 steps.

Both bias and variability remain. Startup anchor0 has212.73mm empirical mean bias RMS over its first four targets under10 steps; it warrants attention independently of contact-stage noise. At anchor140,31 of32 predictions close the jaw while the expert remains open. These are all31 observed jaw mismatches in the sampled384 targets, under both integration counts. This does not establish full-episode timing or directly explain closed-loop chatter.

Next proposed bounded check: compare independent-noise versus shared-noise replanning across all269 expert-observation frames with predeclared seeds. Check target increments, startup/near-contact errors, orientation, mode validity and jaw timing together; never select a noise seed using task success. Fixed noise may reduce variation but cannot remove the demonstrated mean bias. No default inference changes, new optimizer steps or physics rollout are justified yet. This remains one exposed development episode. [Source-hashed result](../../results/summaries/2026-10-31-ball-sampling-result.json).


## 31 October 2026 — Sampling diagnostic implementation

The [fixed-input sampling diagnostic](BALL_SAMPLING_DIAGNOSTIC.md) is implemented. CPU checks cover the noise-copy requirement, actual denoiser count, configuration restoration and empirical error decomposition. GPU sampling remains pending; this adds no evidence of improved predictions or learned grasp success.

## 31 October 2026 — Precision training completed; parent retained

Run `20261031_144208_499175` completes1,000 additional optimizer steps and samples all269 training frames. All ten saved generated streams reproduce their reported metrics exactly; all four checkpoint selections correctly reject. The selected weights are byte-identical to the3,000-step parent. No new rollout has been run or is justified by an unchanged checkpoint.

For seeds90000/190000, the last candidate reduces approach means61.26→59.23mm /55.97→50.88mm (3.31%/9.10%), near-contact means20.33→18.42mm /21.23→18.26mm, and increment residuals25.36→22.68mm /21.18→18.76mm (10.55%/11.42%). Required approach/increment reductions are10%/20% in both probes. Seed90000 orientation p95 rises3.95→4.07degrees. Jaw errors/transitions are unchanged at step1,000. The conservative engineering screens reject modest improvement; they are not physical laws defining grasp success.

With identical expert observations, the two noise draws differ by24.25mm (parent) and21.58mm (last candidate) on average over frames80..119. This motivates separating inference variability from persistent prediction bias, not claiming randomness alone causes failures. Only two draws were observed. The last candidate's generated arrays remain, but its weights were not saved because it failed selection. Next use the retained parent for a bounded fixed-observation sampling diagnosis before choosing another training or inference change. Source-hashed evidence: `results/summaries/2026-10-31-ball-precision-training-result.json`.


## 31 October 2026 — Precision-oriented training prepared

Saved generated predictions with expert training observations still show20.33mm mean XYZ error and24.95mm mean target step over frames80..119; teacher mean target step is3.84mm. Closed-loop drift is therefore not the only unresolved issue. After XYZ-only postprocessing advanced contact, return to training supervision rather than promoting that filter.

`train_hgwind_precision.py` strictly starts from the selected3,000-step lineage checkpoint and caps additional training at1,000 steps, batch2, learning rate1e-5, frozen backbone, unchanged269-frame episode, controller and29D contract. The separate objective increases normalized flow weights on XYZ and jaw to4 (others1), and adds0.1-weight physical XYZ endpoint and0.1-weight demonstrated-increment Huber residuals, both scaled by10mm. These coefficients are an unvalidated engineering hypothesis, not proven optimal settings. The upstream equation is `x_t=t*noise+(1-t)*action`; its endpoint estimate is `x_t-t*v_t`. A hook observes the existing output projection without editing upstream code and checks its residual against the upstream loss. Endpoint reconstruction during training is not equivalent to a fully sampled inference rollout. Temporal penalties compare predicted versus expert increments, preserving correct rapid motions, and exclude padded endpoints. They act within chunks; full generated-stream probes also measure changes across replans.

Every250 steps, generate the complete stream with prefix4 and two fixed seeds90000/190000. Both probes must improve approach and frames80..119 position means by at least10%, and expert-relative increment error by20%. Full position p95, orientation p95, jaw error count/transition count and all recorded mode accuracies must not regress from the parent; native decoding rejects must be zero. Select the lowest combined near-contact error among passing candidates. If none passes, `checkpoint_final` retains the parent, `selected_checkpoint_step=0`, and no physics rollout is recommended. Fixed seeds are repeated training-case diagnostics, not independent validation data; results do not establish generalization. Selection thresholds are screening rules, not task-success thresholds.

Actual CPU strict load, normalizer identity, output-hook/upstream residual equivalence and backward gradient checks pass without CUDA. Synthetic gradient checks exclude padding and padded dimensions and give zero increment penalty for a correct expert discontinuity. Portable metric tests guard both seeds and prevent pose/jaw regressions from hiding behind position improvements. GPU training remains user-run and pending. Existing checkpoints/source training files are preserved; rollback is to use the parent run and stop the new entry point. Full report metadata includes projection delta so the existing checkpoint inspector can reload the selected result.


## 31 October 2026 — Position smoothing advances premature contact

The user-run pair `20261031_142217_528773` completed both269-interval/807-callback episodes with unchanged native safety checks. Both fail independent grasp verification and have zero bilateral contact. Raw commands first contact the ball with the right finger at frame120 (2.000s); XYZ-smoothed commands first contact at97 (1.617s), 23frames/0.383s earlier. Jaw commands are identical, with seven transitions; both first contacts precede a requested sustained closure. The alpha0.25 position filter is not promoted.

All six compared initial state arrays are identical between cases. Raw execution reproduces the original first120frames exactly in joint position/velocity and object position/quaternion/linear/angular velocity. This makes the fixed-stream comparison more interpretable than the earlier live height intervention. It remains one exposed command-reexecution case, not live-policy evaluation or a success-rate study. Position filtering changes timing/path while velocity and acceleration feedforward remain unchanged; these observations do not isolate which remaining prediction error causes collision or rule out every smoothing method.

Source-hashed evidence: `results/summaries/2026-10-31-ball-smoothing-result.json`. Both slow-motion MP4 files contain269 checked frames. Next inspect pre-impact position, orientation and feedforward against the successful demonstration before choosing another intervention; default runtime and training remain unchanged.


## 31 October 2026 — Early contact and fixed-command smoothing diagnostic

The recorded-height intervention completed 269 intervals / 807 physics callbacks without a safety abort, but failed grasp verification. First right-finger contact occurred at frame120, before the height intervention138..179. No bilateral contact occurred. Its initial state/camera poses matched the previous refined run, but initial RGB and first model output differed. Consequently this comparison cannot attribute the changed contact outcome to height substitution. See `results/summaries/2026-10-31-ball-height-diagnostic-result.json`.

Before contact (frames80..119), adjacent model XYZ target changes average30.7mm versus3.8mm for the teacher. Actual end-effector steps average2.5mm; target jumps are not measured instantaneous robot motion. Within-chunk jumps are also present. This supports testing command continuity, without proving it is the sole cause of impact.

New bounded experiment: `scripts/run_hgwind_smoothing.py --run` runs raw and causal XYZ-smoothed versions of one saved model stream sequentially. Alpha0.25 initializes at the first raw target; only channels0:3 change. Remaining26 channels, including orientation, feedforward, masks and jaw, are identical. No online inference or expert decision loop executes. Native controller limits and target-blind geometry checks remain active. This is command-reexecution diagnosis, not learned-policy success. RGB is captured and a two-view slow-motion MP4 is exported for each saved case, including failed grasps.

The selected source is the latest hybrid diagnostic, but the input is its raw `model_action`, bound exactly to saved model IPC responses. Recorded expert height is excluded. Later raw predictions still reflect the hybrid run's observations; this limitation is explicit. Both cases execute the identical frozen stream apart from the XYZ filter. The comparison checks matched initial states and whether raw execution reproduces the original first120 frames before attributing an original failure. Filter lag is material: over first120 commands, average displacement from raw targets is38.8mm and maximum218.3mm, including large startup transients. Retaining velocity/acceleration fields isolates position smoothing but may leave inconsistent feedforward. Physical benefit is pending; do not promote this filter by default.

CPU validation: all538 raw/filtered requests decode; seven new tests cover causal filtering, channel isolation, lag, invalid parameters, hash rejection and motor execution without IPC. A temporary fixture built from an actual recorded trace exercises the complete analyzer and rejects changed action data; it is not a new simulation result. Two GPU simulations remain user-run. Existing live/model code is unchanged; rollback is to stop using the separate diagnostic entry point.


Snapshot: **2026-10-31, Asia/Hong_Kong**. These are development observations, not a benchmark leaderboard or a paper's final test results.

## Refined live rollout: near miss, no finger contact

The selected refinement completes all269 control intervals/807 physics callbacks without a safety abort, but independent grasp fails and neither finger contacts the ball. Nine jaw transitions remain. Pinch-to-ball-center distance reaches8.17mm at138; this is not contact clearance. Over138..156 the finger midpoint rises15.47mm while the teacher descends15.03mm. A [fixed target-height-only diagnostic](BALL_VERTICAL_DIAGNOSTIC.md) is prepared to test this clue; no causal repair or policy success has yet been demonstrated. [Evidence](../../results/summaries/2026-10-31-ball-refined-live-result.json).

## Bounded refinement result — 31 October 2026

The user completed 1,000 additional optimizer steps from the saved 2,000-step checkpoint, at learning rate 1e-5. The saved baseline and all four generated-action probes were recomputed in the model environment and match the report. The selected checkpoint is additional step 1,000 (3,000 lineage steps).

On the same 269 recorded observations with prefix4 execution, mean approach target error improves from 66.696 to 61.256 mm (8.16%). Full-stream mean improves from 44.747 to 38.778 mm, but position p95 **worsens** from 146.775 to 158.527 mm. Orientation p95 improves from 4.457 to 3.954 degrees. Wrong jaw frames decrease from 6 to 5; transitions decrease from 5 to 3, at frames 138/139/140 versus expert closure at 144. Native request decoding rejects no frames; this is not a physical feasibility result.

Step 750 has the cleaner jaw sequence (one transition at 140, four wrong jaw frames) and approach error 62.219 mm. The predetermined selector chooses step 1,000 because approach error is lower and its guardrails compare with the original parent, not the previous candidate. The selected model is not best on every metric. Only the selected full checkpoint was retained; candidate predictions are retained for audit.

CPU downstream loading exposed a missing `action_projection_l2_change` report field. Preserve the byte-identical original report, recompute this metadata from saved tensors (0.3136768937), and annotate it as a post-run audit with report/model/official-weight hashes. No model weights or training history were changed. Strict CPU checkpoint loading, normalization checks and live preprocessing now pass without CUDA. The refinement writer remains frozen to preserve recorded source hashes; another run using that writer requires the same explicit metadata completion before downstream loading.

Next: one matched 269-frame live simulation using this selected checkpoint, legacy camera protocol, prefix4, original seed/initialization and unchanged safety/controller settings. Compare first contact, approach tracking, jaw transitions, bilateral contact and independent grasp evidence with the parent. This remains training-case development; no grasp improvement or generalization is established. [Source-hashed result](../../results/summaries/2026-10-31-ball-refinement-result.json).

## Camera control outcome and bounded model refinement

The optional formal capture completes 8 intervals / 24 callbacks without a safety abort, with exactly matching initial robot state and camera poses. It is **not promoted to default**. Initial wrist RGB MAE against training increases from 13.92 to 24.35/255; external increases 8.36 to 10.34/255. External images at frames 4/7 improve, but wrist images do not. The measured pinch discrepancy at frame 7 is 13.49 mm legacy versus 13.78 mm formal, so there is no observed short-motion improvement. Eight frames do not evaluate grasp; later pixel comparisons also include differing robot poses. [Source-hashed result](../../results/summaries/2026-10-31-ball-camera-control-result.json).

Next, refine the existing 2,000-step model with a separate entry, at most 1,000 additional steps at learning rate 1e-5 (previously 1e-4). This loads the parent weights strictly and retains its normalizers, data and backbone freezing. The optimizer restarts because no optimizer state was saved. It does not restart model weights from the official checkpoint. Old scripts/checkpoints and runtime defaults are retained.

Every 250 steps, generate the complete 269-frame stream on recorded expert observations using the same four-action queue and 90000+frame noise. The archived batch4 parent reference has 66.69 mm mean approach target error (before expert closure at 144), six wrong gripper frames, five transitions, orientation p95 of 4.46 degrees and zero native request rejects. Select a candidate only when approach mean improves versus the current best, orientation p95/gripper error count/transitions do not worsen versus the parent, and native rejects remain zero. Selection recomputes the parent baseline with batch1, matching live inference, and verifies the available eight batch1 teacher-input references. Stored batch4 predictions differ slightly and are not used as an exact replay gate. This avoids selecting by a noisy flow loss alone. It remains same-episode development selection, **not a held-out validation metric or grasp success**.

The new run retains one selected full checkpoint under `checkpoint_final` and all candidate predictions/metrics. `selected_checkpoint_step` records when that checkpoint was selected; it can differ from the total number of optimizer steps. If no candidate passes, step 0 retains the parent and `selected_improved=false`. The original parent directory is never overwritten.

```bash
.venv_dynamicvla/bin/python scripts/refine_hgwind_development.py --run
```

CPU preflight and strict actual parent reload pass without CUDA; 13 focused selection/prediction checks pass. Training completion is reported above; physical improvement remains pending. The user runs GPU jobs; this entry launches no Isaac Sim or robot commands.

## Input sensitivity result and optional render-protocol control

All 32 predictions completed, and the all-live 20-action chunks match saved execution responses exactly at eight anchors. Across their first four actions, image-only replacement changes predicted target position by 19.28 mm on average (max 43.31 mm); state-only replacement changes it by 6.30 mm (max 11.96 mm). These are input sensitivities, not additive causal failure contributions. At frame 0 states match exactly; replacing only images changes the first target by 18.50 mm. At later anchors the hybrid inputs can be inconsistent. Even teacher images/state retain first target errors 403.28 mm at 0,143.85 mm at 8 and95.27 mm at 40, so image consistency alone is not an established cure.

**Reporting correction:** one field ending `_mm` in the original input-probe report accidentally contained metres. The original report and prediction arrays are retained. `verified_input_metrics.json` recomputes the correct units directly from those arrays, and the diagnostic now multiplies by 1000. Existing first-target-error metrics were already correct. A 3-4-5 mm regression checks conversion. [Verified summary](../../results/summaries/2026-10-31-ball-input-sensitivity.json).

Code inspection finds an additional concrete mismatch: the formal recorder resets DLAA history around each four-subframe capture; the original live helper renders once before capture and does not reset that history. The opt-in `formal-temporal` boundary removes the extra prerender, brackets capture with the same history-reset key, and applies only the two renderer intensity clamps saved with this training scene. It does not treat probe ISO as exposure, load saved images, change model weights, use object state, alter scene physics or relax safety. It leaves the native timeline running while requiring zero simulation-time advancement. Camera images are still measured live. Exact appearance matching and improved grasp remain unverified; this combines the formal capture boundary components, not a reset-only causal ablation.

```bash
data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_closed_loop.py --run --frames 8 \
  --camera-protocol formal-temporal
```

The original `legacy` mode remains the default and rollback option. Eight frames test the new capture boundary and matched initial-frame input/output differences, not grasp. Tests: 23 focused CPU checks pass, including zero-time/reset cleanup, invalid calibration rejection and corrected units; the actual scene/model-input CPU preflight passes without CUDA. GPU runtime result pending.

## Full live ball rollout: integration passes, grasp fails

The user-run eight-frame integration and subsequent full 269-frame rollout both pass the live-chain audit. The full run completes 807 physics callbacks with matching initial conditions and unchanged native safety checks. **The learned grasp fails**: zero bilateral-contact frames, no retained lift. This is one exposed training case, synchronous simulation, not held-out or real-time evaluation.

Raw evidence locates the first right-finger contact at frame 104 while the fingers are still open (about 0.040 m per finger). The measured EE differs from the expert by 60.73 mm at that time. The ball remains on the expert path through that pre-action frame, then differs by more than 1 mm at frame 105. The model's 15 gripper-threshold transitions begin later, at frame 106, ending at 130. The expert closes at 144, first contacts at 162, and achieves bilateral contact at 163. Live right-finger contact occurs in only one control interval; left contact is absent throughout.

This supports prioritizing approach deviation and premature single-finger impact over a gripper-only fix. It does not identify why the learned approach differs. At the identical initial state, camera poses agree within 0.001 mm and 0.001 degrees; live/train RGB differs (mean absolute 8.36/255 external and 13.94/255 wrist). Brightness/rendering differences are a candidate input shift, not a demonstrated cause.

A new diagnostic keeps the checkpoint, noise seed, history and prefix fixed and crosses teacher/live images with teacher/live measured robot states at eight pre-contact/contact anchors. Frame 0 has identical states, isolating the image input contrast there. Later hybrid inputs can be inconsistent and are sensitivity probes, not realistic rollouts or counterfactual-success claims. Each all-live prediction must reproduce its saved response within explicit numerical tolerance before interpretation. No object ground truth enters the model; source object states are used only by the CPU evaluator.

```bash
.venv_dynamicvla/bin/python scripts/diagnose_hgwind_live_failure.py --run
```

This runs 32 model input combinations on GPU; it does not train or launch Isaac Sim. Without `--run`, it saves the CPU chronology and optionally a plot. No matplotlib installation is required in the model environment. The actual 32 input combinations, checkpoint contract, frame 0 state equality and saved request bindings have passed CPU preparation; GPU inference is pending. [Compact verified result](../../results/summaries/2026-10-31-ball-live-failure.json).

## Ball startup audit: target semantics and remaining response error

A [source-checked CPU audit](BALL_STARTUP_TARGET_AUDIT.md) identifies initial falling/rebounds and dynamic pregrasp braking clearance as contributors to the expert's large startup targets. At frame18 the safe standoff is about363mm while the ball is about30mm high; this is an upper target, not a one-frame displacement. The targets already occur in the teacher and match the training labels.

At recorded robot states, the 2,000-step model's first actions differ from the expert's native bounded nominal translation requests by a mean0.569m/s in frames0–41 (p951.470m/s). This excludes collision projection, orientation, IK/effort and actual dynamics. The200-step model produces19 native-request semantic rejections; the2,000-step model produceszero across269 first actions. Structural decoder acceptance was insufficient to establish controller validity.

**Follow-up:** live RGB/state30/absolute29 integration is now implemented and CPU checked (above), with the physical run pending; do not interpret this audit as learned grasp or justification for smoothing labels. [Hashed evidence](../../results/summaries/2026-10-31-ball-startup-target-audit.json).

## Ball 2,000-step comparison: improved fit, startup errors remain

The matched 2,000-step fit reduces fixed-probe flow MSE from the 200-step value 2.0003 to 0.3066. Only the step budget changes; the first 200 sampled batches, losses and gradient norms match exactly. Strict CPU checkpoint reload passes. Offline first-action mean position error improves 127.05→46.42 mm, p95 290.69→187.17 mm, mean orientation error 6.19→2.02 degrees, and gripper accuracy 79.18→97.03%. Thresholded first-action jaw transitions drop 78→1, but closure is frame136 versus expert144, 133.3ms early. Prefix4 still has five transitions; this is not a guarantee for queued execution.

Errors are concentrated early: first30-frame mean163.02mm, frames30–143 mean45.06mm, closure-to-contact mean25.46mm, and contact-onward mean18.70mm. All25 worst frames occur within frames0–35. Captured expert upper targets themselves contain12 consecutive-frame position jumps over100mm, all before frame42; the measured robot pinch path is smoother. These are high-level targets, not actual robot displacement. The jumps are present in the teacher capture and do not alone prove a label defect, cause of model error, or poor source grasp.

All source/archive hashes and raw prefix metrics were rechecked; the adapted pretrained baseline reproduces prior metrics exactly. Motion-reference mean errors improve over200 steps, but tail errors do not uniformly improve. No labels were smoothed, removed or replaced. This is one exposed training episode with one fixed noise realization per observation, not held-out success or learned closed-loop control.

**Follow-up:** the startup target and nominal-response audit is now complete (above); bounded live-feedback integration remains next. Do not immediately increase steps again or discard the startup frames. Preserve current controller and safety settings. [Source-hashed comparison](../../results/summaries/2026-10-31-ball-2000step-prediction-result.json).

## Ball offline predictions: adaptation improves pose, fit remains poor

The 200-step model has first-action position mean127.05mm/p95290.69mm and orientation mean6.19degrees on its own training episode. Gripper accuracy is79.18%, with78 thresholded offline transitions versus one expert transition. Before-close position error is129.63mm, so this is not only a contact-stage issue. Velocity/acceleration mean errors worsen relative to the adapted baseline. All source hashes and raw prefix metrics are independently rechecked. Next is a matched2,000-step fit, CPU-preflight passed; efficacy and learned grasp are unverified. [Result](../../results/summaries/2026-10-31-ball-offline-prediction-result.json).

## Ball development fine-tune: optimization and checkpoint verified

The user-run 200-step update finishes on `local/traj_00018`, sampling198 distinct anchor frames. All six paired flow probes improve: mean4.8705→2.0003, a58.93% reduction. Checkpoint weights change as reported and strictly reload with exact source-derived normalizers on CPU. This establishes a working optimization/checkpoint path, not pose accuracy or learned grasp. Actual29D action prediction is prepared next; no new simulation or GPU inference was started by the agent. [Result](../../results/summaries/2026-10-31-ball-development-finetune-result.json).

## Current priority: one passing ball case through the model pipeline

The development bridge for `local/traj_00018` now verifies269 synchronized60Hz frames, bit-exact successful float32 action labels, source/teacher/render state equality and calibrated30D robot state. Official pretrained weights load with action29/state30 on CPU; normalization, image/language preprocessing and loss-padding gradients pass. These are data/model-interface checks, not trained-policy results. GPU fine-tuning subsequently completed as reported above; generated-action prediction and live policy integration remain pending. Use this deliberately selected passing case to advance the pipeline while retaining the following failures as known defects. [Details](BALL_DEVELOPMENT_TRAINING.md).

## Single-drive control: early motion reduced, full grasp still fails

The [single-drive run](MIMIC_DRIVE_CONTROL.md#observed-result--31-october) verifies the intervention and all214 issued decoded fields, but completes213/319 intervals before a joint5 velocity/braking-feasibility abort. Bilateral contact lasts0.533s; sustained lift fails. Contact is delayed200ms, so reanalysis uses each run's first150ms after bilateral contact: maximum absolute joint5 velocity2.032→0.637rad/s and secant acceleration328.58→34.91rad/s². Later measured velocity-2.625rad/s still differs from model-predicted-0.770rad/s. This one exposed pair does not establish a robust fix or mechanism. Keep default settings; inspect the remaining contact-stage response mismatch. [Source-hashed result](../../results/summaries/2026-10-31-single-drive-result.json).

## Matched precision result: float64 reference grasps successfully

The matched float64-label run completes all 319 intervals and 957 physics callbacks safely. Initialization and full decoded motor-boundary audits pass. Independent raw HDF5 recomputation verifies 2.483 s continuous bilateral contact, at least 57.768 mm final lift, 0.211 mm relative-position jitter and no grasp assistance. All 15 core arrays match the successful teacher through the first 180 recorded frames, including the previous float32 failure window. Selected arm/EE/object state arrays first differ exactly at frame 217; this is not whole-episode exact replay.

The pinned float32 run stops after 179 complete intervals on motion feasibility. The matched pair supports sensitivity to upper-label quantization in this exposed execution chain. It does not establish repeatability, tolerance to model errors or a general benefit from changing model precision. Source labels, decoder, quaternion hemispheres, initialization, fixed gains, posture, safety callbacks and collector patch were controlled; original float64 labels round exactly to the failed float32 input.

The collector still prints `status=failed` and stores `failed_00000`: its unused native task FSM records `episode_timeout: Grasp not started` with zero skill steps. The packet executor intentionally bypasses online task/skill decisions. Preserve that original label and distinguish it from the independent contact/lift/retention pass; this is not official benchmark success or a VLA result.

**Next:** inspect contact-stage feedback and effort response to small state/action differences, then establish stable float32 execution before cohort conversion or training. Float64 success is a diagnostic reference, not the deployment solution. No further simulation, rendering or controller tuning was launched for this analysis.

[Source-hashed result](../../results/summaries/2026-10-31-precision-control-result.json). The 353 CPU checks belong to the implementation milestone; this result milestone recomputes raw evidence, boundary audits and input/dependency hashes.

## Initialized29D surrogate: initialization passes, contact-stage execution fails

The initialized29D surrogate now verifies startup force/gain alignment but stops after179 complete intervals (180 recorded frames) of319. All180 issued command-boundary fields and jaw targets match the decoded action prefix; full-stream interface flags remain false because the run aborts. First contact166 and bilateral contact170 match the successful teacher. EE error exceeds0.01mm at172 and grows from0.0031mm at171 to4.3217mm at179. Native safety rejects the next command after joint5 reaches-2.8248rad/s against the2.61rad/s limit; required braking also conflicts with acceleration/jerk bounds. Independent grasp fails without sustained lift. This is an expert-labelled interface diagnostic, not learned-policy inference.

Read-only recomputation checks the original raw HDF5 contact evidence, pinned source/action hashes and every issued motor input. The reported `interface=false` remains a full-stream rejection, not evidence of a discovered codec field mismatch: the legacy auditor requires319 events and has only180. Prefix correctness does not establish safe or successful full execution.

Through the observed prefix, nonnumeric settings/callbacks/posture match teacher inputs; numeric differences are small float representation changes (maximum position-component difference0.000237mm), with equivalent quaternion sign changes. That excludes a large input-field discrepancy in this prefix, but does not identify a unique cause of contact divergence. The native fifth-joint limit remains2.61rad/s and has not been relaxed.

**Next proposed control:** preserve initialized packet execution and use original double-precision upper labels with the same quaternion hemisphere convention. This tests sensitivity to float32 conversion without changing gains, force, safety limits or scene. It is now implemented with353 passing CPU checks, SDK parsing and actual native-entry/label-loading preflight. The user-run physical result now passes; see the matched precision result above. An exact-label replay success would still not establish robustness to learned-policy errors.

[Source-hashed result](../../results/summaries/2026-10-31-initialized-contract-result.json). No new simulator/render job or runtime code change by agent. The previously passed338 CPU checks belong to interface preparation; this milestone rechecks existing artifacts.

## Fresh expert grasp restored after measured gain reapplication

User run `20261030_022121_219398_fresh60_omit1_gains` completes319control intervals/957native callbacks safely. The single setup-force omission and post-setup finger-gain reapplication both verify; gains persist at the recorded checks. Independent grasp passes:2.483s continuous bilateral contact, at least60.349mm final lift,0.321mm relative-position jitter and no grasp assistance.

A separate read-only recomputation verifies the saved raw hashes, original source/dependency hashes, all15 full core arrays, and **all958 native ball transform/velocity rows** against original successful attempt60. These match exactly. Contact/lift evidence is recomputed directly from HDF5 and equals the saved report. This is stronger than equal initial conditions or a visual impression.

The previous omission-only fresh60 control fails after190frames; adding the measured warm gain reapplication restores original success under the retained omission. This supports a causal contribution from the measured gain mismatch in this exposed case. It does not identify the exact internal native overwrite mechanism, establish a cohort success rate, certify dual active gains with mimic coupling, or evaluate a learned policy. Official benchmark success remains unverified; independent engineering criteria pass. Modified-force provenance and training exclusion remain explicit.

**Follow-up:** the initialized29D surrogate has now been tested; its failed post-contact execution is reported above. The existing `--teacher-run` preparation can inherit diagnostic fields into its plan, but its standard packet entry does not install the history/force/gain hooks. Do not assume those fields execute or run an unmatched comparison. Avoid more collection/training until this interface check is meaningful.

[Source-hashed measured result](../../results/summaries/2026-10-30-gripper-gain-reapplication-result.json). No new GPU/render run or runtime code change by agent in this result milestone.

## Observed native gripper initialization mismatch

The passive drive pair completes with `comparison_ready=true`. Independent rechecks confirm15 original core arrays for all three history episodes, and15 exact arrays against the previous fresh-omission failure. Target history60 succeeds; fresh60+omit1 still aborts at190recorded frames. The observer did not change these known arrays.

At frame 0 before `ensure_gripper_force`, both have native Kp[10000,0]/Kd[1000,0]. Both call `ensure_gripper_force(70, speed=.05, reconfigure=True)` exactly once. In the fresh episode it returns true and records requested10500/140, but native gains stay unchanged. In history59/60 the same call yields Kp[10500,10500]/Kd[140,140], retained at first close frame154. At closure native caps are equal:70N and0.05m/s on both fingers; applied targets and USD/mimic properties also match in the compared snapshots. The7.2N/0N fresh cap is a **pre-configuration frame 0** reading, not the closure cap.

The frozen method sets native gains before authoring USD maxForce and adding a missing second-finger drive schema. That first-episode USD addition and the post-call gain mismatch are measured; an internal native overwrite mechanism remains inferred. Both warm active finger gains coexist with mimic coupling, so copying them is diagnostic reproduction, not certification of mechanically correct production actuation.

A bounded one-attempt `--restore-gripper-gains` control is implemented and CPU validated: preserve the verified force omission, reapply only the measured requested finger gains after original setup, require native readback and unchanged arm gains/caps/targets/USD/state, then observe persistence.314 portable checks, SDK Python3.10 parsing and real scene/entry-isolation preflight pass. No new GPU/render run by agent and no successful correction yet. [Measured evidence](../../results/summaries/2026-10-30-gripper-drive-result.json) · [protocol and command](GRIPPER_DRIVE_DIAGNOSTIC.md#one-explicit-gain-reapplication-control).

## Observed result: startup ball mismatch removed, finger response differs

User run `20261030_005620_189348_fresh60_omit1` verifies exactly one pinned omission and three remaining native setup force returns. No native startup abort occurs in this launch. Independent full-prefix rereading finds ball native transform and linear/angular velocity **exactly equal to successful history for rows0..500**; row501 first differs at2.7833334s. This is evidence that the specific force submission contributed to the earlier startup mismatch in this case. It does not establish an internal accumulation mechanism or a general reset correction.

The episode still fails:190 recorded frames/189 completed intervals, then motion safety stops joint offset4 at-2.6516rad/s against the absolute2.61 limit. Bilateral contact lasts0.2833s without sustained lift.

Both runs first command zero finger-position targets at frame154; those recorded targets match through166. At166, the omission run's finger joints remain2.192/2.490mm farther open while EE position differs by only0.0886mm. Positive raw right-finger/ball impulse begins at native step500 in history versus516 in omission (about88.9ms later); left-finger steps are511 versus520. Thus the remaining visible discrepancy includes finger closure before contact, not just a different free-moving ball path. Equal saved position targets do not establish equal native velocity targets, gains, force/speed limits or solver/mimic state.

`Expert.ensure_gripper_force` can change live drive gains/caps and USD drive attributes, while its free-travel branch preserves some authored properties. That is a code-supported reset/drive-history hypothesis, not a measured difference in gain values. Existing records lack the full actual drive properties at closure. Next inspect those properties at episode initialization and before closure in matched conditions, before changing parameters. Zero explicit finger entries in `get_applied_joint_efforts` are not measurements of the position drive's actual force.

This is an expert development experiment with modified forcing history, excluded from training. No VLA, new GPU run or rendering by the agent. [Source-hashed result](../../results/summaries/2026-10-30-setup-force-result.json).

## Historical preparation of the force-history experiment

The [setup-force control](SETUP_FORCE_CONTROL.md) now supports one explicitly modified fresh60 attempt: omit only the pinned second pre-recording wind submission. All other native submissions, controller settings, limits and scene parameters are retained.287 portable CPU checks, final56 focused checks, existing SDK Python3.10 parsing and actual scene/collector preflight pass. Physical result and hypothesis efficacy remain unverified; no GPU run or rendering by agent.

## Matched fresh60 result: history contrast persists

The user run `20261030_003531_182273_fresh60` passes the watcher admission probe and stage-load barrier, then records 169 frames / 168 completed control intervals. No native startup abort occurs in this launch; this is not a general stability guarantee. The independent engineering grasp check fails: bilateral contact lasts 0.4667 s without sustained lift. Joint offset 4 (the fifth arm joint) reaches 2.6393 rad/s against a 2.61 limit, and required braking conflicts with acceleration/jerk bounds. Safety stops the worker with exit 1; limits are retained.

Independent raw HDF5 rereading confirms all 15 core arrays exactly match the previous fresh60 failure. The same guarded-startup history58/59/60 reference instead reproduces all original arrays and successful attempt60 (319 frames). First native ball velocity diverges at row 1: fresh minus history is approximately (0.004831869, 0, -0.000151388) m/s. Initial recorded state matches. Thus the scene-load barrier alone does not explain recovery; worker history/reset has an observed effect on initial physical evolution in this bounded comparison. No unique hidden solver/force/cache mechanism has been isolated.

Fresh startup records four pre-recording wind calls versus three for history60. The extra observed call has x-force 0.67059 N. Equal clock readings and API call counts do not establish an extra applied impulse. Next proposed work is a bounded episode-initialization investigation of this difference, with controller and safety limits frozen. No new simulation or rendering was launched by the agent.

The analysis now reports `comparison_evaluable=true` separately from `complete=false`, `worker_exit=1` and grasp failure. A saved safety abort remains valid startup-comparison evidence; it is not successful execution. The old report is preserved locally. 23 focused CPU regressions pass, including aborted, failed-startup and malformed-evidence cases. [Source-hashed summary](../../results/summaries/2026-10-30-matched-fresh60.json).

## Matched fresh control unavailable; startup resource shortage measured

Fresh60 run `20261030_000853_155886_fresh60` repeats the same Omniverse Client mutex abort during the first guarded update, before any attempt. The loading barrier has not reliably fixed startup. No new grasp result exists; the earlier exact three-episode history reproduction stands.

A separate no-physics/no-renderer CPU USD loader reports30 inotify ENOSPC watch-creation failures; direct resource probing allocates only26/512 watches. Host limit is65536 and VS Code holds approximately64927 watch references. Disk availability is357G. Watch exhaustion is confirmed, while its causal relationship to the Client abort remains unknown. An admission probe prevents simulator launch when insufficient. After VS Code reload, summed watch references fall to 581, the allocation probe passes 512/512, and the same CPU-only USD loader opens the scene with zero watch errors. Matched fresh60 CPU preflight and exact collector/history checks pass. Resource recovery is verified; native simulator stability and crash causation remain unestablished. The next step is one matched fresh60 run without rendering. No GPU was launched by the agent. [Evidence and next action](NATIVE_STARTUP_RESOURCES.md).

## Original worker history reproduced; matched-startup fresh control pending

User run `20261029_234931_549595` executes original attempts58/59/60 in one process and reproduces failed720frames, failed363frames and successful319frames. Independent rereading verifies15 full-length action/observation arrays per episode exactly and the first7 native ball-velocity rows; all initial-state/safety checks pass. Attempt60 independently passes physical grasp evidence:2.4833s bilateral contact and at least60.349mm final lift. The two earlier failures were original failed demonstrations and reproduce as expected.

This supports a startup/process-history explanation for the earlier fresh60 mismatch, but the stage-load barrier also changed startup scheduling. One matched-startup fresh60 control is prepared to distinguish those factors. Its native collector assembly matches the successful history, runtime dependencies are pinned, and263 CPU plus real scene/native-entry checks pass. That new simulator result is pending. No learned policy, formal training or render ran. The native crash did not recur once; no general crash fix or unique force/cache mechanism is established. [Verified hashes and limits](../../results/summaries/2026-10-30-worker-history-reproduction.json), [control command](WORKER_HISTORY_DIAGNOSTIC.md#observed-reproduction-and-one-matched-startup-control-30-october).

## History launch blocked by native client abort

The user’s history launch `20261029_232920_958901` aborts during scene loading with no observed/saved attempt. GDB locates `std::system_error/Invalid argument` in `libomniclient.so` on `OC Tick`; matching-binary inspection identifies the failed mutex-lock path. The underlying resource/ownership failure is unknown. This is unavailable history evidence, not zero successful grasps out of three.

The optional [stage-load barrier](WORKER_HISTORY_DIAGNOSTIC.md#current-blocker-and-optional-startup-workaround) now defers script application and waits for stopped-timeline attachment before native robot USD edits. It is CPU validated, with260 portable checks and actual scene/native-entry preflight; its ability to avoid the native crash is untested. It changes startup scheduling, so any future result must state that condition. The default patched collector is hash-identical to the previous launch, and rendering remains optional. [Source hashes and scope](../../results/summaries/2026-10-29-startup-client-abort.json).

## Worker-history experiment prepared; physical result pending

The [new bounded command](WORKER_HISTORY_DIAGNOSTIC.md) executes original attempts58/59/60 in one native worker, then analyzes every saved result. Rendering is now optional at the user’s request (`--render`); skipped video generation does not imply a failed diagnostic. Original outcomes were failed720frames, failed363frames and successful319frames. An unchanged safety exit can truncate this sequence; no automatic worker restart is allowed. Compare startup physics and full arrays with originals and the previously measured fresh attempt60 before interpreting a grasp outcome.

251 CPU checks, actual scene/three-source preflight and native-entry assembly verification pass. No GPU sequence has been run by the agent and no history cause or physical improvement is claimed. See [preparation provenance](../../results/summaries/2026-10-29-worker-history-preparation.json).

## Startup force measurement: reproduced failure, observed setup calls

User run `20261029_225526_640181` reaches simulation under GDB, records169frames/168completed intervals, and renders five30Hz views of85frames each. The15 compared action/observation/native arrays match the prior uninstrumented fresh teacher1 failure exactly. This establishes observed noninterference for those arrays in this run, not a general claim about logging/debugger timing.

The joint-offset4 speed is2.6393rad/s versus the2.61 bound; required braking conflicts with acceleration/jerk bounds. Bilateral contact lasts0.4667s, but sustained lift fails. The diagnostic completing means that a measurement exists; grasp still fails. No learned policy ran.

Ten force API calls are observed: four before control-frame recording at the same0.033333334s clock reading, followed by six marked control frames. Setup force x-components are0.670589983,0.670589983,1.668164968,1.668164968N. All calls return normally and report configured dt1/60s. This configured value is not the native integration-step record. No zero-dt call was observed in coverage, so the earlier zero-dt hypothesis remains unsupported. Equal clock readings do not prove force accumulation or absence of internal stepping; original force-call histories are unavailable.

The two earlier C++ scene-load crashes do not recur in this launch. No native crash stack was saved and no crash fix is established. Proposed next step: compare original worker history58/59/60 with the already recorded fresh attempt60, preserving settings and every outcome. This experiment is not implemented/run yet; even a history effect would not isolate a unique fan/cache/solver cause. [Source hashes and measured values](../../results/summaries/2026-10-29-startup-force-measurement.json).

## Repeated scene-load abort: native stack capture prepared

The bounded retry `20261029_224044_493811` reproduces the earlier C++ `std::system_error: Invalid argument` at the same stage. Both runs stop before completed script inspection, save zero episodes and record zero force calls. No wind, grasp or learned-policy conclusion follows. Keep the two failures as execution evidence; the original native crash cause is still unknown.

A [native backtrace option](STARTUP_PHYSICS_DIAGNOSTIC.md) now runs the same entry/SDK child under the installed system GDB, saving the fault stack without modifying forces, scene or control parameters. 243 CPU checks and real scene preflight pass; CPU-only debugger tests preserve 0/7 exit statuses and capture a controlled abort stack. No GPU diagnostic was launched during preparation. This is a new measurement, not a demonstrated native-crash fix. See [source-hashed preparation](../../results/summaries/2026-10-29-repeated-startup-abort-backtrace-preparation.json).

## Passive startup measurement: native scene-load abort, no force evidence

User run `20261029_223230_505509` exits while opening/loading the Task stage: C++ `std::system_error`, `Invalid argument`, then native abort. No episode or video is saved; `startup_force_calls.jsonl` has zero rows. This occurs before completed script inspection and the first force proxy installation. It is not a grasp failure or evidence that the fan applied no force. The native crash's cause is not established; the startup-force hypothesis remains untested.

The coordinator now emits `STARTUP_DIAGNOSTIC_FAILED status=no_evaluable_episode`, records observation availability separately and exposes native fatal messages. 240 CPU checks and diagnostic scene preflight pass. No GPU retry or control/physics/SDK change was made during this correction. One manual retry of the same command is proposed; a repeated native abort calls for startup investigation, not repeated grasp tuning. Original logs and the initial diagnostic JSON are preserved. See [source-hashed execution diagnosis](../../results/summaries/2026-10-29-startup-native-scene-abort.json).

## Earliest divergence isolated to startup physics

CPU analysis of raw 180 Hz evidence places ball velocity divergence in cases1/7 at the first physical step, about5.56ms, before meaningful arm divergence. Initial ball mass/inertia/COM/transform, first motor action and next control-frame EE positions match. The first three contact-step records report zero ball contact impulse. Original cases1/7 were third/second attempts within a worker; original case18 was first and reproduces exactly in a fresh probe. This process-history association and the fan's force-calling startup path are hypotheses, not an identified root cause.

A [single passive force-call diagnostic](STARTUP_PHYSICS_DIAGNOSTIC.md) is prepared, with all saved outcomes automatically rendered. It changes recording only, forwarding native forces unchanged. 235 CPU checks and actual scene preflight pass; GPU execution is pending. Initial loading/force-accumulator history is partly outside observer coverage. No new grasp or VLA result follows. See [source-hashed diagnosis](../../results/summaries/2026-10-29-startup-physics-diagnosis.json).

## Bounded batch completed: one passing surrogate, two teacher safety aborts

Batch `20261029_215343_401088` finishes without the earlier recording error. The four new saved runs have five verified RGB videos each (20 new videos); both existing references add ten videos. All are saved-state review renderings, not new physics, formal training pixels or completed human review.

| Development case | Native expert regeneration | Frozen 29D surrogate |
|---|---|---|
| local/traj_00001 | Safety abort after 168 completed intervals; no sustained lift | Skipped because teacher fails |
| local/traj_00007 | Safety abort after 176 completed intervals; no sustained lift | Skipped because teacher fails |
| local/traj_00018 | Pass; 269 intervals, all 15 core arrays exactly reproduce source | Pass; 269 intervals / 807 physics/OSC updates, decoded-action/live-state/safety checks pass |

For traj_00018 the surrogate retains bilateral contact for 1.767 s, at least 57.277 mm final lift and 0.006 mm relative-position jitter; teacher lift is 59.468 mm. No grasp assistance or learned policy is used. This adds one compatible development case while preserving the earlier traj_00000 retention failure. It does not establish broad reliability or training readiness.

The two teacher aborts report hard next-step velocity/position bounds conflicting with acceleration/jerk. Both match source initial states exactly but first depart by more than 0.01 mm at frame 3, before contact. Seeds, sampled variations, robot-start/inertial metadata, simulator settings, task plan and finger material match. Collision-LOD settings/fingerprint match (session-layer paths differ); physical-limit differences are online results. Root cause of early expert divergence remains unresolved. Their missing surrogates cannot be counted as failed surrogate grasps. Neither 1/3 nor 1/1 is a supported generalization or learned-policy success-rate claim.

**Next decision:** keep the frozen controller, inspect teacher early-reference divergence and human videos before more collection or training. Do not loosen safety limits or tune friction to explain failures that occur before the surrogate is run. See [source-hashed results](../../results/summaries/2026-10-29-bounded-contract-batch-result.json).

## Integrated action contract — interface verified, physical retention fails

User run `20261029_210114_208078` completes 209 intervals and 627 physics/OSC updates, with exactly matching initial robot/object states, native safety passed, no grasp assistance, decoded 29D commands verified, live posture matched to raw robot states and collision callbacks always enabled. The combined physical gate is **false**: bilateral contact spans 137–162 (0.433 s), final retained lift is approximately zero and final relative-position jitter is 55.786 mm. The preserved fixed-gain/latched-posture reference retains 1.2 s bilateral contact, about 72.595 mm lift and 0.160 mm jitter.

The initial report crashed on three quaternion differences up to 2.22e-16 between CPU and simulator normalization. Native `_orientation_projection` also normalizes its input axis in place, producing six post-call trace differences. The corrected **analysis-only** verifier allows 1e-14 absolute roundoff exclusively on quaternion/axis fields; every other input remains exact. Raw artifacts and the strict raw equality result remain unchanged. The normalization-aware execution audit passes; physical retention still fails. See [contract](BALL_ACTION_CONTRACT.md) and [hashed result](../../results/summaries/2026-10-29-ball-action-contract-result.json).

Before first contact at 135, EE difference from the passing reference is at most 0.00217 mm. Object positions first diverge at 136, and EE displacement exceeds 0.01 mm at 139, reaching about 3.65 mm at 150. The callback activation difference starts only at 173. This excludes late callback activation as the explanation for the earlier onset; it does not identify a unique cause among float32 conversion, normalization, sign representation and subsequent contact dynamics. Native robot posture-source modes match throughout.

21 runtime dependency hashes and unchanged runtime encoder/decoder functions verify. 210 portable CPU checks pass; no additional simulation, model training or inference was launched for this analysis. Recorded expert upper commands replay without fresh target-conditioned decisions, so this exposed replay failure does not demonstrate that a learned reactive policy cannot succeed. Freeze the controller and run a bounded 3–5 distinct development-episode diagnostic before deciding label generation/training. Existing 40 processed 9D recordings still lack the complete upper labels.

## First bounded batch: execution errors, no new grasp evidence

Batch `20261029_213626_774692` renders both existing passing/failed references completely: five RGB views each. The three new teachers all stop on `KeyError: frame` inside the recording callback before any evaluable episode is saved. No surrogate is executed. These are **three execution errors and zero new evaluable grasps**, not a 0/3 grasp success rate. Preserve the existing genuine 209-frame physical-retention failure separately.

The native segment dictionary has no recording frame; the recorder now uses its explicit callback index on a copy and rejects conflicting frame metadata. Added coverage calls the actual `record_segment` entry with an unframed segment, unlike earlier fixtures. All 194 reference links from the existing 209-frame native capture pass without modifying original inputs. 227 portable CPU checks pass. The coordinator now stops immediately on an unsaved execution error. Controller parameters/inputs and physical criteria are unchanged; no GPU rerun was launched during this fix. That fresh batch is now completed above; retain the previous execution errors and their logs. See [source-hashed diagnosis](../../results/summaries/2026-10-29-native-capture-frame-fix.json).

## Wind-driven spherical-object cohort

Task: `grasp_wind_driven_motion_1s1p_hgprgew`.

- 619 saved attempts: 40 accepted successes and 579 failures. Collection was intentionally stopped at 40 successes; its original target was 100.
- Main-machine group: 21 successes; remote group: 19. The groups used different approach-depth settings and must retain their provenance.
- 12,154 control frames, recorded at 60 Hz; 40 distinct numeric trajectory fingerprints.
- Raw/core HDF5 consistency, finite actions/states, timing, scene identity and contact indexing checks passed.
- Independent recording analysis found positive impulses on both fingers and no attachment assistance in these 40 episodes. These checks complement, rather than replace, task verification and visual review.
- All 40 revised-camera episodes are now on the main machine and passed canonical render-completeness and frame-index/timestamp checks at 60 Hz, totaling 12,154 frames. All 19 remote episodes completed pinned-revision Hub download and installation with file-checksum verification.
- Across five RGB views and three sampled frames per episode/view (600 samples), none were flagged as blank. Source raw-file hashes and the physical scene signature remained unchanged. Sampling does not certify target visibility or grasp quality throughout every video.
- A temporary TLS interruption delayed cloud downloads after upload had succeeded; sync-only resumption completed delivery without rerendering.
- Manual quality/visibility review, ball-task action conversion and training remain pending; no held-out split is claimed yet. See the [render-completion evidence](../../results/summaries/2026-10-28-render-complete.json).

A [descriptive quality screen](QUALITY_SCREEN.md) now covers all 40 episodes, with nine event sheets spot-checked. It finds mixed source-group advantages, frequent wrist-view truncation and one substep contact-alternation case requiring review. No source-based training weight or success label has been changed.

A 40/619 collection ratio describes this adaptive development collection only. It is neither a policy success rate nor an unbiased estimate of a frozen expert's generalization.

## One native expert reproduction

A fresh run of the original frozen expert configuration reproduced all 15 core numeric arrays of local/traj_00000 exactly over 209 frames, with passing runtime safety and independent contact/lift/retention checks. It recorded 627 physics updates and 582 native OSC calls. This establishes reproducibility for one exposed case; it does not establish converted-action or VLA success. The raw control goals and exported actions differ by up to 0.046696 rad. Complete-control physical replay now also passes independent grasp checks and exactly matches all 15 core arrays. It uses original cached Jacobians/constraints; the live policy action contract is still unverified. See [the complete-control replay summary](../../results/summaries/2026-10-28-control-reference-replay.json). See [the probe protocol](HGWIND_EFFORT_PROBE.md) and [source-hashed summary](../../results/summaries/2026-10-28-native-expert-reproduction.json).

The first current-Jacobian component diagnostic aborted after 150 completed intervals. The added adapter omitted the original angular task scaling (0.03), pairing an unweighted Jacobian with weighted task velocity. With the metric corrected, the rerun completes 174 intervals but still aborts; first contact/bilateral frames match the source, and appreciable motion divergence begins after contact. The subsequent native-geometry comparison succeeds, as detailed below (latest runtime CPU regression suite: 123 passes). It retains the remaining logged expert references and must not be counted as a deployed policy. See [the metric diagnosis](../../results/summaries/2026-10-28-task-metric-correction.json) and [retained dependencies](../../results/summaries/2026-10-28-live-kinematics-readiness.json).

The subsequent native-geometry live-Jacobian run passes all 209 intervals, safety and independent grasp checks. Its 194 recomputed Jacobians and all 15 core numeric arrays match the source exactly. This certifies the live geometry component for one exposed case, with other logged command references and shadow bookkeeping still retained. See [the live-geometry success evidence](../../results/summaries/2026-10-28-live-native-geometry-success.json).

The task-acceleration feedforward ablation completed 209 intervals safely but produced no finger/object contact; EE difference reached 50.122 mm by frame 15. The audit confirms 110 changed nonzero commands and otherwise unchanged requested joint references and control/scene fingerprints. Direct velocity differencing does not recover the original acceleration. Retain the passing reference while defining the complete Cartesian feedback contract; the result does not independently establish a required VLA action dimension. See [the dependency audit](../../results/summaries/2026-10-28-task-feedforward-result.json). Latest runtime CPU regression suite: 129 passes.

The read-only [upper Cartesian target capture](CARTESIAN_TARGET_CAPTURE.md) succeeds over 209 intervals, records 194 upper calls and preserves exact equality of all 15 core arrays. The initial identity-based association check was incorrect because native consumption copies references; corrected exact-content matching recovers all 194 links offline, with raw records unchanged and the original report preserved. `cartesian_target_capture_verified=true`; 141 CPU checks pass. Standalone feedback execution remains pending. See [verified capture evidence](../../results/summaries/2026-10-29-cartesian-target-capture-result.json).

The upper-target feedback reexecution now passes: 209 completed intervals, 627
physics updates, 582 OSC callbacks and all 194 upper calls aligned. Regenerated
references and all 15 core arrays exactly match the exposed source. Independent
grasp evidence measures 1.2 s continuous bilateral contact, at least 74.861 mm
retained lift and 0.123 mm relative-position jitter; initial state and runtime
safety also pass. Original source files are unchanged. Native expert timing,
live safety callbacks, gripper, source settings schedule, controller memory and
15 terminal joint-only intervals remain explicit dependencies. This is one expert
component result; a standalone policy contract and VLA success remain unverified.
See [the protocol](CARTESIAN_TARGET_CAPTURE.md) and [source-hashed result](../../results/summaries/2026-10-29-cartesian-target-reexecution-result.json).
Latest runtime CPU suite remains 149 passes; no runtime code changed in this result update.

The next [explicit packet probe](CARTESIAN_PACKET_INTERFACE.md) bypasses online task/skill
decisions, commands jaws explicitly and replaces the terminal joint fallback with
a Cartesian hold. Native geometry helpers and fixed actuator initialization remain;
target truth access is blocked at their safety-world boundary. Numeric settings and
motion still come from the recorded stream. 160 portable CPU checks and real
preflight pass; its physical result and learned-policy contract remain pending.

The first packet physical attempt stops at frame 0 because the new safety wrapper
omitted an optional emitted-geometry query and falsely classified the fan mesh as
missing. This was reproduced with frozen native code and corrected without
loosening geometry or robot limits. All 163 CPU checks and scene preflight pass;
the corrected physical packet run remains pending. This run provides no grasp
performance evidence. See [the protocol diagnosis](../../results/summaries/2026-10-29-packet-geometry-protocol-fix.json).

The corrected geometry-protocol packet run reaches bilateral contact, then aborts
after 174 completed intervals. The first command divergence occurs at frame 151
while robot state, Jacobian and whole-arm constraints still match the passing
reference. Explicit CLOSE began at frame 118, but the new adapter left the native
terminal geometry context in fully-open mode. That omission is fixed and audited;
166 CPU tests and scene preflight pass. The corrected full physical result is
pending. See [the jaw-context diagnosis](../../results/summaries/2026-10-29-packet-jaw-context-diagnosis.json).

With jaw context synchronized, the packet rollout safely completes 209 intervals
and passes request alignment but loses final retention. The first command
changes at frame 156, before any recorded robot/object state differs. Native
terminal safety was still missing measured robot linear/angular velocity context
normally provided by the skipped skill tick. That robot-only context is now
refreshed, logged and causally audited; 170 CPU checks and scene preflight pass.
The corrected physical grasp result is pending. See [the motion-context diagnosis](../../results/summaries/2026-10-29-packet-robot-motion-context-diagnosis.json).

## Explicit Cartesian packet grasp: one exposed case

The corrected packet executor passes 209 complete intervals / 627 physics and
OSC updates, with initial-state identity, runtime safety, explicit jaw mode and
causal measured-robot-motion context verified. The online task/skill decision
loop is bypassed and the final 15 intervals use Cartesian hold rather than an
expert joint tail. First 194 references and all 15 core arrays exactly match
the source; the remaining hold intervals are intentionally different.

Raw contact-evidence recomputation confirms 1.2 s continuous bilateral contact,
at least 74.861 mm final retained lift and 0.131 mm relative-position jitter,
without assistance or limit changes. Thirty source/execution hash checks pass.
The collector's unchanged FSM success remains false because its decision loop
is bypassed; no saved labels are relabeled. The separate packet/grasp gates pass.

This validates a recorded numeric request boundary, not a learned VLA. Recorded
motion and settings remain dependencies; this single exposed case provides no
success-rate or held-out physical-generalization estimate. The next gate is the
policy-facing settings/motion contract before ball-task training. No runtime
code changes in this result update; the prior 170 CPU passes remain applicable.
See [protocol](CARTESIAN_PACKET_INTERFACE.md) and [hashed evidence](../../results/summaries/2026-10-29-cartesian-packet-success.json).

## Prepared fixed-feedback-gain diagnostic

An opt-in probe now fixes only pose gains at 9/8 rather than switching to 4/4
at frame 173. Exact comparison changes 36 of 209 requested frames and preserves
every other packet field and native limit. Seven regressions bring the portable
suite to 177 passes; actual scene preflight and saved passing-packet re-audit
pass. No new physical result or learned-policy improvement is claimed. Remaining
posture/orientation/motion schedules are still recorded dependencies. See
[preparation evidence](../../results/summaries/2026-10-29-fixed-gain-probe-preparation.json).

## Fixed-feedback-gain grasp: one exposed case

The fixed-9/8 physical probe passes all 209 intervals and independent grasp
checks with 627 physics and OSC updates, matching initial conditions and unchanged
native limits. Raw packet/contact recomputation and 37 provenance/execution
checks pass. Only gain fields differ over frames 173–208; first 173 core frames
are exactly equal to the recorded-gain packet baseline.

Bilateral contact lasts 1.20 s, minimum final retained lift is 72.577 mm and
relative-position jitter is 0.160 mm. The recorded-gain comparison was 74.861 mm
and 0.131 mm. The fixed-gain case passes the same criteria but does not improve
these measurements. The gain schedule is unnecessary for this single exposed
case; broader reliability and the remaining controller/motion contract are
unverified. Recorded requests still drive execution, not VLA predictions. The
original collector labels and raw reports remain unchanged; prior 177 CPU checks
apply, with no runtime changes. See [hashed result](../../results/summaries/2026-10-29-fixed-gain-probe-result.json).

## Prepared measured-posture diagnostic

The next probe uses the passing fixed-gain configuration and removes only
`posture_target`, selecting native current-state redundancy handling. Exact
comparison changes 173 requests and preserves all other input fields and limits.
Current measured q and the native resolved posture/source are explicitly audited.
All 186 portable CPU checks, actual scene preflight, old fixed-gain preflight,
saved baseline re-audit and 209 native-branch CPU fixtures pass. No new physical
or learned-policy result is available. See [preparation evidence](../../results/summaries/2026-10-29-measured-posture-probe-preparation.json).

## Measured-posture result: failed final retention

The comparison completes 209 intervals with safe, correctly audited native
current-state posture handling. Raw measured q matches every logged resolved
posture, and only the posture-reference field differs from the passing fixed-gain
baseline. Thirty-eight read-only checks and raw contact recomputation pass.
Bilateral contact lasts 0.433 s (frames 137–162), but final retention/lift fails;
relative-position jitter is 55.677 mm. Both the independent grasp and combined
measured-posture gates fail. The fixed-gain/recorded-posture baseline remains valid.

For this exposed case, original native code reconstructs the source posture anchor
exactly from initial robot q and logged sampled IK/homotopy offsets. This supports
testing posture as explicit robot-side initialization/configuration; it does not
justify removing the term, adding a required seven-dimensional model output or
claiming universal cross-task compatibility. No runtime code changes in this
update; prior 186 CPU checks apply. See [hashed result](../../results/summaries/2026-10-29-measured-posture-probe-result.json).

## Prepared robot-side latched posture

A new comparison generates a posture anchor once from live initial robot joints
and explicit sampled robot variation, then retains it for all 209 requests.
Configuration contains no saved initial q/anchor or object state. The first 173
requests exactly match the passing fixed-gain case; only 36 later posture fields
change. All 193 portable CPU checks, actual scene and old fixed-gain preflight,
initializer fixture, baseline re-audit and Python 3.10 parsing pass. No physical
gripping result or universal controller compatibility is claimed. See
[preparation evidence](../../results/summaries/2026-10-29-latched-posture-probe-preparation.json).

## Live-initialized posture: two same-condition successful repeats

The user ran the latched-posture probe twice with the same seed and initial
condition. Both complete 209 intervals / 627 physics and OSC updates safely and
pass independent grasping. Forty read-only checks per run verify requests,
initialization/raw robot state and contact evidence. All 15 core arrays and
request/execution payloads are exactly equal across the two repeats.

Each retains 1.20 s bilateral contact, about 72.595 mm final lift and 0.160 mm
relative-position jitter. Live initialization replaces per-frame saved posture
execution and remains in effect through lift/hold. This supports a passing
exposed-case controller baseline, not cross-trajectory reliability, a success-rate
estimate or learned VLA success. Other recorded target motion/settings inputs
remain explicit dependencies. Raw reports/limits are unchanged; prior 193 CPU
checks apply. See [hashed evidence](../../results/summaries/2026-10-29-latched-posture-probe-result.json).

## Earlier DynamicVLA integration: irregular egg-object task

Task: `grasp_sliding_motion_on_flat_surface_1s1p_g9wvkk1`. Earlier shorthand called the object a ball; the target is an egg. This task remains a difficult development case.

A 2,000-step run used 40 training episodes and eight same-task validation episodes. Normalization was fitted on training data only. The fixed validation flow-MSE probe decreased from **12.1492 to 0.4197**. This is a flow-prediction diagnostic, not a physical distance or grasp-success score.

The saved final offline first-action evaluation covered 1,848 validation frames: mean position error **9.16 mm**, p95 **17.36 mm**, mean orientation error **2.49 degrees**, and gripper-sign accuracy **88.53%**. These measurements condition on recorded observations. They do not establish performance on states visited by the live policy, and they are not directly comparable to evaluations with different sampling noise or execution prefixes.

Subsequent closed-loop debugging has not established reliable learned grasping. Camera consistency, action timing, velocity references, stochastic action continuity and visited-state distribution remain distinct sources of failure. Better offline fit is insufficient evidence of closed-loop success.

## What cannot yet be concluded

- No established VLA robustness gain under physical disturbances.
- No independent physical-generalization test score.
- No evidence that a 2 mm depth bias alone caused better demonstrations.
- No automatic Franka-to-UR10e transfer and no completed real-robot experiment.

The compact [evidence snapshot](../../results/summaries/2026-10-28.json) records source-report hashes and selected metrics. Full records remain outside Git; hashes establish provenance, not public reproducibility without access to those records.
