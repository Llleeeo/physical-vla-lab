# Changelog

## 2026-10-31 — Optional RGB approach-distance filter prepared

- Add default-off robot-finger height projection with coherent vertical motion references, latched missing-view pose hold and two-estimate reacquisition.
- Keep the original 0.5 jaw decoding and raw model predictions. Runtime perception uses RGB; sphere size is an explicit static prior.
- Verify scene geometry, strict CPU model load and regression checks. Independent grasp improvement remains untested; label this VLA plus visual control.

## 2026-10-31 — Withdraw jaw confirmation; inspect approach from RGB

- Remove the0.8 confirmation flag/module and restore pre-experiment live runtime. Preserve all historical results.
- Add a CPU-only two-view orange-ball localization/geometry prototype with explicit missing/ambiguous-view rejection and evaluator-only object truth.
- Verify complete precontact coverage on two exposed runs, with worst observed3D error12.3mm. Runtime distance protection remains unimplemented pending uncertainty/missing-view handling.

## 2026-10-31 — Gripper confirmation does not prevent open-finger collision

- Verify all269 consumed raw predictions and the optional gate against saved execution; independent grasp fails without safety abort.
- All jaw outputs remain below0.8, so the gate never closes. Right-finger contact occurs at100, before the first suppressed raw closure102.
- Record horizontal misalignment at descent and historical rendering confounds. Export existing RGB to a verified two-view video; retain defaults.

## 2026-10-31 — Optional causal gripper confirmation prepared

- Add explicit three-frame closure confirmation and reopening hysteresis, preserving raw predictions and all other action channels.
- Audit consumed jaw commands against native execution; keep the switch off by default.
- CPU preflight passes; saved-stream13 transitions reduce to1, with no live success claim. Document height protection as a separate proposal requiring verified perception.

## 2026-10-31 — Per-step live replanning result verified

- Bind all269 executed first actions and verify premature single-finger contact with no grasp.
- Record13 jaw-command transitions and confounds in the historical prefix4 comparison; retain existing defaults.
- Preserve existing-data timing-improvement scope; export saved camera review without another simulation.

## 2026-10-31 — Distinguish fixed orientation targets from executed tilt

- Verify constant expert orientation labels despite changing measured wrist tilt.
- Prioritize native angular-command audit before fixed-attitude expert changes; defer the prepared repeat pair.

## 2026-10-31 — Candidate grasp failure verified; repeatability plan fixed

- Verify latest live contact chronology and artifact hashes; distinguish integration pass from failed grasp.
- Correct visible-motion versus jaw-command transition interpretation for the retained-parent near miss.
- Fix two identical-config parent trials before further training or controller interventions; GPU execution pending.

## 2026-10-31 — Translation-objective result verified

- Verify both 500-step checkpoints, paired training draws, and all 24 prediction streams.
- Record modest nominal velocity improvement alongside worse near-contact position versus parent and unchanged jaw errors.
- Retain the default; prepare a bounded exploratory candidate rollout without claiming grasp improvement.

## 2026-10-31 — Conditional translation training ablation prepared

- Add a separate matched training pair with endpoint command supervision and unchanged full-action flow loss.
- Verify native velocity-mapping parity, corresponding future-state supervision, and gradient/padding behavior.
- Evaluate full predicted speed contracts without silently discarding rejected requests; no automatic promotion or simulation launch.

## 2026-10-31 — Native upper-target provenance audited

- Rebuild269 labels exactly and verify frozen expert dependencies.
- Separate startup ingress hold/release and2mm depth adjustment from actual arm motion.
- Audit all eight retained-parent streams through native pre-collision translation mapping; preserve labels, checkpoint and safety limits.

## 2026-10-31 — Prefix ablation completed; parent retained

- Verify matched500-step training branches, both checkpoints and24 full generated streams.
- Record modest prefix-weighting tradeoffs versus uniform continuation, without near-contact improvement over parent; preserve both candidates without promotion.
- Flag startup upper-goal discontinuity for provenance review, explicitly distinguishing it from physical arm motion or proven label corruption.

## 2026-10-31 — Matched executed-prefix training ablation prepared

Add two matched500-step continuations from the same parent: uniform temporal weights versus4x first-four weights, with valid-weight normalization. Match batches and explicit flow noise/time hashes; preserve all29 channels and later-horizon supervision. Save both final candidates and evaluate all eight fixed seeds without promotion or physics.

## 2026-10-31 — Shared-noise comparison verified; no adoption

- Verify all16 complete prediction streams and source hashes; shared noise worsens mean near-contact position and jaw switching while barely changing increment error.
- Retain existing inference and reject best-seed selection. Record structural20-target/four-executed-action supervision counts as a hypothesis for a bounded training ablation, not a proven cause.

## 2026-10-31 — Full-episode noise comparison prepared

Add independent/shared Gaussian noise replanning across269 expert-input frames and eight predeclared seeds, retaining10 integration steps. Verify prior predictions, terminal padding and source provenance. Report boundary versus within-chunk increment errors, startup bias and jaw timing; no model/controller change or automatic promotion.

## 2026-10-31 — Sampling diagnostic verified; retain10 steps

- Verify source hashes, exact repeatability and recomputed prediction metrics.
- Record13.97% higher sampled near-contact RMS and2.36x mean latency with30 steps; no sampler promotion.
- Identify startup bias and sampled premature jaw closure; propose shared-noise replanning diagnosis without claiming grasp success.

## 2026-10-31 — Fixed-input sampling diagnostic prepared

Add paired10/30-step, eight-draw inference checks at12 fixed expert-observation anchors. Protect shared initial noise from upstream in-place mutation; audit actual solver counts and restore runtime settings. Separate empirical XYZ bias/spread and report decoder/mode errors. GPU results pending; no training or controller change.

## 2026-10-31 — Precision training completed; no checkpoint promoted

- Recompute all saved prediction metrics and four rejected selections; verify selected weights equal the parent exactly.
- Record modest position/continuity improvement and the remaining screening failures. No new physics or success claim.
- Record two-noise sensitivity and propose a bounded fixed-observation inference diagnosis; candidate predictions are retained, rejected weights are not.

## 2026-10-31 — Precision-oriented training prepared

- Add isolated weighted flow and physical endpoint/expert-increment objectives; preserve upstream code, existing weights, dataset and controller.
- Verify actual CPU forward/backward and source-equivalent flow residuals; mask padding from all loss components.
- Add two-seed generated-action screening every250 steps with parent fallback, complete checkpoint metadata and a1,000-step cap. No GPU training or task-success claim yet.

## 2026-10-31 — Smoothing comparison completed, no grasp improvement

- Verify both269-frame episodes, native safety, contact evidence and269-frame videos against saved hashes.
- Confirm exact paired initial states and raw reproduction of the original first120frames.
- Record earlier contact under XYZ smoothing (97 versus120), zero bilateral contact in either case, and reject default adoption. No runtime or training change.

## 2026-10-31 — Fixed-model command smoothing diagnostic prepared

- Record failed height intervention and pre-intervention RGB/action confound; no causal height claim.
- Add separate raw/smoothed command reexecution, exact source/model-response checks, native motor auditing and independent contact/grasp evaluation.
- Preserve default online policy and native safety limits. Record filter displacement and verify raw-prefix reproduction; label all outcomes diagnostic.
- Export saved two-camera RGB to checked slow-motion videos, including failed grasps. GPU pair remains pending user execution.

## 2026-10-31 — Refined near miss and explicit height diagnostic

- Verify complete refined live integration but no finger contact; near-center alignment precedes upward departure. Preserve raw evidence and exported review videos.
- Add opt-in recorded target-z substitution on frames138..179, raw/executed action logging and independent provenance/command audit. Default model-only branch and native limits retained.
- Handle zero-contact failure diagnostics without indexing a nonexistent contact frame. Diagnostic simulation remains pending and cannot count as unassisted policy success.

## 2026-10-31 — Bounded refinement completed; matched rollout ready

- Verify all saved generated-action probes and selection. Approach mean improves 8.16%; position p95 worsens. Step750 has a cleaner jaw sequence than selected step1000.
- Preserve the original report and complete missing projection-change metadata from unchanged saved weights. Strict CPU reload and live preprocessing pass; matched full rollout remains user-run and pending.
- Record source hashes and limits; no new physical success or held-out result.

## 2026-10-31 — Repair CPU test import isolation

- Actions runs 71 and 72 failed during test collection with the same missing diagnostic-script import. Local checks had included an extra `scripts` search path.
- Move the unchanged position-distance helper into a NumPy-only source module and import it from the diagnostic and its unit test. Keep CI search paths and checks intact.
- Require the complete workflow suite in a clean tracked-file export before publication; all 404 CPU tests, dependency checks and staged publication audit pass.

## 2026-10-31 — Camera control reviewed; bounded refinement prepared

- Verify 8 intervals/24 callbacks in optional formal camera mode, but no short-motion improvement; retain legacy default. Initial wrist image mismatch worsens.
- Add a separate strict parent-weight refinement entry: at most 1,000 extra steps, 1e-5 learning rate, generated 269-frame prefix4 probes every 250 steps, explicit non-regression selection and parent fallback. No training launched by agent.
- Actual parent CPU reload/preflight and 13 focused selection/prediction tests pass. Training-case selection is not generalization evidence.

## 2026-10-31 — Input sensitivity measured; formal render control prepared

- Reproduce all saved live chunks at eight anchors; image-only target changes average 19.28 mm versus 6.30 mm for state-only changes. These are sensitivity probes, not success estimates.
- Correct a metres-to-millimetres report field from saved arrays, preserving original evidence; add a unit regression.
- Add optional formal temporal-history reset/clamp boundary, keeping legacy capture as default. 23 focused CPU checks and actual scene/model-input preflight pass. Eight-frame physical test pending; no training or safety changes.

## 2026-10-31 — Full learned rollout and contact diagnosis

- Verify 269 control intervals/807 physics callbacks without expert actions or safety abort; independent grasp fails with zero bilateral frames.
- First right-finger contact 104 precedes jaw chatter 106–130; expert first contact 162. Preserve failure evidence and unchanged controller.
- Add CPU chronology/plot and opt-in 32-condition GPU image/state sensitivity diagnostic, with matched noise and saved-response fidelity checks. GPU diagnostic pending.

## 2026-10-31 — Live absolute29 ball-policy integration prepared

- Add isolated simulator/model entries using live calibrated RGB and the same measured state30/preprocessing as training. Consume model absolute29 chunks through the existing native packet/effort controller; no online expert decisions or recorded action fallback.
- Preserve live inputs, model chunks, consumed commands, timing and raw physics evidence; audit input/output binding and motor-boundary execution separately from task grasp. Default eight intervals; synchronous inference pause, no real-time claim.
- 389 CPU tests pass. Actual preprocessing and strict checkpoint reload match on CPU; real scene/native patch checks pass. GPU rollout is pending, not a learned grasp result. [Protocol and command](docs/project/BALL_LIVE_POLICY.md).

## 2026-10-31 — Ball startup target mechanism and native-request audit

- Bind captured targets to float32 labels and startup telemetry; verify dataset/prediction hashes and frozen control sources. Initial rebounds and dynamic braking standoff contribute to large expert target jumps; these are not measured robot displacements.
- Compare generated first actions through the native nominal translation function at recorded states. The 2,000-step model still differs by 0.569 m/s on average in the first42 frames. Its269 first actions have zero nominal rejects; the200-step model has19 inconsistent relative-speed/reference requests despite structural decoding acceptance.
- Publish a reproducible CPU analysis and scoped phase results. No training, labels, controller parameters or scene changes. Next prepare bounded live29D integration; no new live adapter or learned grasp is claimed. [Audit](docs/project/BALL_STARTUP_TARGET_AUDIT.md).

## 2026-10-31 — Longer ball fit improves predictions; startup limitation isolated

- Verify identical configuration except training duration and identical first200 training updates. Fixed probe mean2.0003→0.3066; strict CPU reload passes.
- Recompute offline first-action error127.05→46.42mm, orientation6.19→2.02degrees, gripper79.18→97.03%, transitions78→1. Closure is133.3ms early; prefix4 retainsfive transitions. Largest errors are in startup, where captured upper targets already jump. Preserve this distinction from measured motion.
- Save a local comparison plot and publish hashed aggregate evidence. No runtime/model/controller changes or agent GPU jobs; next audit startup target semantics. [Result](docs/project/BALL_DEVELOPMENT_TRAINING.md#observed-2000-step-result).

## 2026-10-31 — Ball predictions verified; bounded longer fit prepared

- Recompute raw offline metrics and verify archive/source hashes: first-action mean position error127.05mm, p95290.69mm, gripper accuracy79.18%,78 predicted transitions versus one expert transition. Errors span approach and contact; velocity/acceleration errors do not uniformly improve. Decoder acceptance does not imply physical feasibility.
- Existing 2,000-step entry passes CPU preflight. It changes training duration only, starts from the same pretrained weights and preserves the 200-step result. GPU run and improvement are pending; no runtime/controller change. [Protocol](docs/project/BALL_DEVELOPMENT_TRAINING.md#verified-offline-result-and-bounded-next-experiment).

## 2026-10-31 — Ball fine-tune completed; action prediction prepared

- Verify 200 optimizer steps, 198 sampled anchor frames, changed checkpoint weights and a 58.93% reduction across six fixed training-episode flow probes (4.8705 to 2.0003). Strict CPU checkpoint reload and exact new normalizers pass. No learned grasp or held-out claim follows.
- Add independent absolute29D generated-action comparison, same noise/normalizers before and after fine-tuning, prefixes1/4/20 and explicit decoder/mode/clipping diagnostics. CPU regressions cover quaternion signs, units, padding-tail execution and invalid outputs. Prediction is user-run; no GPU/simulation launched by agent. [Details](docs/project/BALL_DEVELOPMENT_TRAINING.md#observed-fine-tuning-result-and-next-command).

## 2026-10-31 — Passed ball case connected to development training

- Prioritize a complete learned-policy pipeline using exposed `local/traj_00018`, whose expert and float32 surrogate already pass. Keep the other case's contact/safety defects unresolved; no controller limits or dynamics changed.
- Add a source-hashed269-frame RGB/state30/absolute-action29 adapter and bounded pretrained fine-tuning entry. Verify teacher/source/render identity, causal frame mapping, calibrated pinch state, exact successful action labels, endpoint padding and model normalizers on CPU.
- Existing official weights load with the new dimensions; image/language preprocessing and29-channel masked gradients pass without CUDA. GPU optimization, action prediction and learned simulator grasp remain pending. [Run and limitations](docs/project/BALL_DEVELOPMENT_TRAINING.md).

## 2026-10-31 — Single-drive result inspected

- Verify exact float32 actions, runtime fingerprints, initial setup, follower-drive intervention, all214 issued action/jaw fields and independent raw contact/lift evidence. Execution reaches213/319 intervals before a joint5 speed/braking-feasibility abort; grasp fails without sustained lift.
- Contact is delayed; compare the first150ms after bilateral contact rather than only absolute time. Early joint5 motion is smaller, but later measured/model velocity mismatch remains. Keep defaults unchanged; single-drive alone is insufficient. Save source-hashed evidence and analysis script. No new GPU/render/training or runtime edits.

## 2026-10-31 — Single-drive mimic control prepared

- Raw native records show alternating contact-stage velocities in both precision conditions and a large measured/model velocity mismatch before the float32 abort. Driving both fingers while retaining the mimic follower is a candidate contributor, not a proven cause.
- Add one opt-in float32 control with follower native stiffness/damping zero and leader10500/140 retained; require the original authored mimic relationship, unchanged caps/arm/USD properties, exact baseline actions and matched source/controller hashes. Active grip authority changes. Full safe execution and physical grasp remain separate gates; 369 CPU checks and actual-source/SDK preflight pass; no GPU/render/training launched. [Experiment](docs/project/MIMIC_DRIVE_CONTROL.md).

## 2026-10-31 — Matched float64 labels pass physical grasp

- Independently verify 319 complete intervals/957 callbacks, full action-boundary and initialization gates, 2.483 s bilateral contact, 57.768 mm retained lift, 0.211 mm relative jitter and no assistance. All 15 core arrays through the first 180 frames match the successful teacher exactly.
- Preserve the unused native FSM failure label; independent physical checks pass, official benchmark/VLA success remains unverified. One matched pair supports quantization sensitivity, not model robustness. Next inspect contact-stage control sensitivity before float32 training. Analysis/docs only; no GPU, rendering or runtime changes.

## 2026-10-31 — One matched precision diagnostic prepared

- Add opt-in `--precision-control`, reconstruct original float64 labels and preserve baseline quaternion hemispheres; require exact roundtrip to the pinned failed float32 stream. Verify source/controller/patch hashes and original-label reconstruction again at runtime. Default float32 execution remains unchanged.
- Label diagnostic precision in outputs and report issued-prefix validity separately from full execution/grasp.353 CPU checks, SDK Python3.10 parsing and actual default/diagnostic entry assembly and label loading pass. Physical outcome pending; no GPU, rendering or training by agent.

## 2026-10-31 — Initialized surrogate failure localized after bilateral contact

- Verify both initialization interventions and recompute raw contact evidence. All180 issued action/jaw inputs match the decoded prefix; full319-frame execution fails after179 intervals. Contact timing matches the successful teacher, followed by rapid motion divergence and a joint5 velocity/braking-feasibility abort.
- Preserve original reports and distinguish prefix-field correctness from full-stream interface acceptance. Source upper fields differ only by small numeric representation changes and equivalent quaternion signs in this observed prefix; this does not prove precision causes failure. Propose one matched float64-label control before retuning. Analysis/docs only; no new GPU, rendering, training or runtime edits.

## 2026-10-31 — Matched initialization carried into one29D surrogate

- Add a dedicated fresh60 entry that explicitly installs the verified force omission, post-setup gain reapplication, passive drive observation and stopped-timeline barrier around packet execution. Reuse the reset boundary for one attempt only; no online task/skill decision or prior episode runs. Preserve the frozen29D representation, existing controller/limits and raw/frozen sources.
- Pin the verified teacher and generate319×29 float32 actions. Reject inactive inherited hooks in the standard entry. Report initialization, interface decoding, safe completion and independent grasp separately; preserve failures and support analysis-only finish.338 portable CPU checks,108 focused checks, SDKPy310 parse and actual scene/native-entry assembly pass. User-run physical outcome pending; no GPU/render/training by agent.

## 2026-10-30 — Fresh expert grasp restored exactly

- User run with verified setup-force omission plus one post-setup native finger-gain reapplication completes319 intervals/957 physics callbacks safely and passes independent grasp. Independently recomputed all15 core arrays and all958 native ball transform/velocity rows match original success exactly; bilateral contact2.483s, final lift at least60.349mm, no assistance.
- This supports a causal contribution of the measured gain mismatch in one exposed case. No general initialization fix, production mimic certification, learned-policy result or official benchmark success claim. Next carry initialization explicitly into29D surrogate execution; existing standard packet entry does not install these diagnostic hooks. Analysis/docs only; no new GPU/render by agent.

## 2026-10-30 — Actual gripper gain mismatch measured

- Verify passive pair and recheck all original history/fresh control arrays. First episode reports requested10500/140 gains but native readback remains10000/1000 and0/0; later history episodes retain10500/140 on both fingers. Closure caps match70N/0.05m/s. No corrected grasp result yet.
- Add opt-in one-attempt post-USD initialization gain reapplication, with native readback, arm/property invariance and later persistence checks. Reproduce measured warm actuation while retaining mimic coupling; diagnostic only, excluded from training.314 CPU checks, SDK parsing and real scene/entry isolation pass. Original/frozen sources and data remain unchanged; no GPU/render by agent.

## 2026-10-30 — Passive gripper-drive pair prepared

- Add opt-in native/USD drive snapshots at reset, initialization and closure, plus before/after original force-configuration calls. Preserve original arguments/results/exceptions; no simulator setter or extra physics update added. Missing/nonfinite readings remain errors.
- One bounded coordinator runs history58/59/60 then fresh60+the existing force omission, serially without rendering or retries. Require unchanged known motion and valid observations; stop before the second job if the first cannot establish the baseline.300 portable CPU checks, SDK Python3.10 parsing and real scene/entry preflight pass. Physical properties/effects are not yet measured.

## 2026-10-30 — Setup force control removes initial ball mismatch

- User-run omission verified: exactly one suppressed submission, three remaining native setup force calls. Ball pose and linear/angular velocity match successful history exactly for native rows0..500; first difference moves to row501 at2.7833s. A specific startup-force contribution is supported for this case; no general reset fix or accumulation mechanism is established.
- Grasp still fails after190frames. Equal close-position targets yield different precontact finger motion:2.192/2.490mm farther open at frame166 versus0.0886mm EE error; first right-finger impulse arrives about88.9ms later. Next inspect actual drive/reset state. Analysis/docs only, raw sources retained, no new GPU/render by agent.

## 2026-10-30 — Bounded setup-force intervention prepared

- Add opt-in `--omit-setup-force-once` for matched fresh60 only. Pin the verified baseline trace; require the exact first/second call signature and normal first native return, then omit only the second submission. Forward all other calls. Default/frozen sources and control limits remain unchanged.
- Separate modified-force provenance and runtime intervention verification from comparison completion/grasp. No training eligibility or VLA claim. 287 portable CPU checks, final56 focused checks, SDK Python3.10 parsing and real scene/collector preflight pass. One user-run attempt without rendering is pending; no simulator efficacy claimed.

## 2026-10-30 — Matched fresh control measures a history/reset contrast

- Fresh60 starts without native abort, then stops safely at 169 recorded frames. All 15 core arrays match the prior fresh failure; guarded history60 instead matches original successful319-frame recording. Ball velocity differs at the first native physics step despite matching initial recorded state. No unique force/cache mechanism or general crash fix is established.
- Separate evaluable comparison evidence from worker completion and physical grasp success. Preserve original report and worker exit1; no simulator/controller/physics changes. 23 focused CPU regressions pass. No new simulation or rendering by agent.

## 2026-10-30 — Watch resources recover after editor reload

- Verify the old large watcher exits, summed watch references fall to 581, and the independent allocation probe passes 512/512. CPU-only USD load opens the task normally with zero watch errors.
- Matched fresh60 CPU preflight and exact collector/history checks pass. Prepare one user-run simulator retest without rendering; no controller, physics or SDK changes. Native-crash causation remains unproven.

## 2026-10-30 — Fresh control blocked; inotify exhaustion measured

- Same native Client mutex stack recurs on the first stage-load-barrier update; no attempt starts. The barrier does not reliably avoid the crash.
- CPU-only USD loader reports30 watch-creation ENOSPC errors. Host watch limit65536 is nearly full, mostly from the project VS Code watcher; direct probe allocates only26/512. Disk has357G available. Native-crash causation remains unproven.
- Add local watcher exclusions (activation pending window reload) and a prelaunch admission probe with unconditional resource cleanup.267 portable CPU checks pass. No GPU, controller, physics, kernel limit or SDK change by agent.

## 2026-10-30 — Original three-attempt history reproduces exactly

- Verify user run58/59/60: failed720frames, failed363frames, successful319frames;15 full action/observation arrays per episode and first7 native startup ball-velocity rows match originals exactly. Target60 passes independent grasp/initial-state/safety checks with2.483s contact and60.349mm lift. No learned policy ran.
- Startup barrier completes without timeline advancement; no native crash recurs once, which is not a general crash fix. History and startup scheduling remain confounded against the earlier fresh failure.
- Prepare explicit one-attempt `--fresh-control --stage-load-barrier`, pinning this successful history and identical collector/runtime dependencies. Keep no-render default.263 portable CPU checks and real scene/native-entry checks pass; matched-control GPU result pending.

## 2026-10-29 — Scene-load crash localized; bounded workaround prepared

- Capture the history launch abort before any attempt: Omniverse Client `OC Tick`, failed mutex-lock throw in its internal callback path. Specific cause remains unknown; no grasp or history result exists. A100-cycle CPU-only local client check passes but does not exercise Kit.
- Add opt-in `--stage-load-barrier`: finish Kit attachment with scripts deferred and timeline stopped before native robot USD edits. Preserve original default assembly, source/controller/physics and no-render default; record guards and require valid evidence. The startup sequence changes, and simulator efficacy is unverified.
- 260 portable CPU checks, actual scene/source preflight and exact native-entry assembly checks pass. No simulator run by agent.

## 2026-10-29 — Worker-history rendering becomes optional

- At the user’s request, default history runs now collect/analyze only and retain raw trajectories. Explicit `--render` opts into videos, including later `--finish --render`.
- Report skipped rendering separately and allow analysis-complete diagnostics without videos. Ten focused history checks and actual scene preflight pass; no simulation or rendering was launched for this change.

## 2026-10-29 — Bounded original worker-history comparison prepared

- Add one-worker attempts58/59/60 with original seed/profile/depth/reset and safety exits retained; no restart loop or success-target replacement. Pin all three original recordings and the prior fresh control.
- Rotate passive force/Cartesian recorders per attempt, restore prior wrappers, analyze each saved result and automatically render all saved successes/failures. Resume delivery with `--finish` without launching collection again.
- Eight new regressions bring portable CPU checks to251. Real scene/three-source preflight, Python3.10 parsing and actual native-entry assembly hash pass. GPU result and human video review remain pending.

## 2026-10-29 — Passive force measurement reproduces the saved failure

- User's GDB launch passes scene loading; the earlier native crash does not recur and remains unexplained. The native teacher again aborts at169 recorded/168completed intervals;15 compared action/observation/native arrays match the earlier uninstrumented failure exactly.
- Observe10 fan calls, including4 before recording at one unchanged clock reading;0 zero-configured-dt calls within coverage. Original force histories and accumulation behavior are unknown; no force/root-cause attribution or physics/controller fix is claimed.
- Verify all five30Hz videos with85 decoded frames, source indices and timestamps. Save hashed evidence and propose one original worker-history comparison; no additional GPU job, runtime edit or learned-policy evaluation.

## 2026-10-29 — Repeated native abort requires a backtrace

- Preserve the second identical scene-load C++ abort with no episode/force observations; avoid further unchanged retries or unsupported root-cause attribution.
- Add opt-in GDB launch of the same SDK child, with SDK library environment restored only to that child and fault stacks saved separately. No attachment to existing jobs, new SDK, force or controller change.
- 243 portable CPU checks, actual scene preflight and five CPU-only debugger integration cases pass, including actual SDK Python3.10 normal/controlled-abort startup without SimulationApp. Isaac debugger run and native-crash diagnosis remain pending.

## 2026-10-29 — Missing startup measurements labelled explicitly

- First passive diagnostic aborts natively at scene load with std::system_error/Invalid argument, before episode or force recording; preserve all raw files. No grasp or wind hypothesis is evaluated.
- Expose native fatal messages; classify missing episodes, missing force observations, unsupported signatures and incomplete videos separately. Print FAILED for incomplete diagnostics; saved physical failures can still be valid measurements.
- Five status regressions bring portable CPU checks to240 passes. One manual retry is proposed; native crash root cause remains unknown, no GPU retry/SDK/physics/control changes during this fix.

## 2026-10-29 — Startup physics evidence and passive force trace

- Locate cases1/7 ball velocity divergence at the first180Hz physics step, with matching initial native properties/first robot command and no nonzero recorded ball impulse over the first three steps. Recover original worker ordinals3/2 versus passing case18 ordinal1; reset history remains a hypothesis.
- Add an opt-in fan force-call forwarding trace, separate worker, CPU history analysis and one-attempt coordinator that renders saved success/failure. Physics, native force arguments and controller settings stay unchanged; observation cannot capture pre-installation force history.
- Eight new regressions bring portable CPU checks to235 passes. Actual scene preflight, Python3.10 simulator parsing and byte-identical preflight/runtime patch assembly pass; GPU execution remains pending.

## 2026-10-29 — Bounded contract batch produces one passing pair

- Verify all three native regenerations and the sole eligible 29D surrogate; traj_00018 teacher reproduces all15 core arrays exactly and surrogate passes269 intervals/807updates,1.767s bilateral contact and57.277mm retained lift.
- Two teachers safety-abort on velocity/acceleration/jerk feasibility after early frame3 source divergence; their surrogates are skipped, not failed. Preserve earlier retention failure; no VLA evaluation or broad reliability claim.
- All four new saved outcomes have five decoded review videos each; existing references add ten. Record source hashes and next investigation; no control changes or new GPU jobs during analysis.

## 2026-10-29 — Native capture callback frame error fixed

- First bounded batch rendered both existing references (ten RGB videos), but all three native teacher launches hit a recorder `KeyError: frame` before saving evaluable episodes. They are execution errors, not failed grasps; no new surrogate or grasp-rate evidence exists. Preserve the logs.
- Supply the explicit callback index on a non-mutating segment copy, reject conflicting frames, and add an actual `record_segment` callback regression. Preserve controller inputs. Stop the coordinator on unsaved execution errors instead of launching all remaining cases.
- Four regressions bring portable CPU checks to 227 passes; all 194 reference links across the existing 209-frame capture verify without caller mutation. Single-case compatibility preflight passes. CPU checks do not establish physical recovery; fresh bounded execution remains pending, without overwriting completed-error stages.

## 2026-10-29 — Bounded paired batch with all-outcome videos prepared

- Add a serial three-case teacher/29D-surrogate coordinator, automatic rendering of every saved success/failure and existing passing/failed references, unified five-view video index and protected human notes. Resume does not repeat completed failed physics; missing renderings can retry.
- Declare binary-jaw schema eligibility before outcomes: 9/20 new local sources supported, 11 intermediate-width sources unsupported, not failed/low-quality labels. Select attempts 60/144/345 (traj_00001/00007/00018); no replacement after teacher failure.
- Prepare directly from verified aligned native teacher labels without repeating historical component probes. Existing teacher fixture recreates the exact previous 209×29 action array; fixed runtime settings and action contract remain unchanged.
- Extend rendering to saved episodes without analysis reports or successful filenames, preserve source hashes, and verify every RGB MP4 plus source timing. 13 new regressions bring portable CPU checks to 223; actual three-case/two-render preflights and simulator parsing pass. GPU batch/results/videos remain pending.

## 2026-10-29 — Integrated interface verified; grasp retention fails

- User run completes 209 intervals / 627 physics and OSC updates with matching initial state, native safety and verified 29D command/live-posture boundary. Independent grasp retention fails: 0.433 s bilateral contact, no final retained lift, 55.786 mm relative-position jitter. Preserve the earlier passing baseline.
- Fix analysis-only exact-equality failures: three cross-environment quaternion-normalization differences and six native in-place orientation-axis normalizations, at double-precision roundoff. Allow only 1e-14 absolute tolerance on these two normalized fields; other inputs remain exact. Keep the raw strict audit and artifacts; no controller/runtime command changes or GPU rerun.
- Verify 21 recorded runtime dependencies and unchanged encoder/decoder functions. Before first contact, maximum EE difference from the passing baseline is 0.00217 mm; object states first differ at 136 and EE departure exceeds 0.01 mm at 139. Callback activation differs only from 173, so it cannot explain the earlier divergence onset. Individual float32/normalization/sign effects are not isolated.
- Five regressions bring portable CPU checks to 210 passes. Freeze settings; use a bounded 3–5 distinct development-episode diagnostic, retain all outcomes, and reassess open-loop contact sensitivity before formal training. No learned-policy/generalization failure claim.

## 2026-10-29 — Complete ball action-contract gate prepared

- Stop successive one-field deletions; freeze passing fixed gains/live-latched posture and prepare one integrated float32 command-interface check. Explicit 29D actions carry pose, independent velocity/acceleration references, null masks, controller modes and binary jaws; 30D robot-state schema exposes the robot posture anchor.
- Every other native input is fixed, robot-derived or rejected as unsupported. Always enable both non-target collision callbacks; no predicted safety-disable or hidden terminal/expert schedule. Preserve the native hard-limit/effort pipeline and all original data/reports.
- Actual 209-frame CPU preflight preserves jaw/null/mode labels, with maximum position roundtrip error 0.000237 mm. Real pinned upstream configuration accepts 29D actions/30D state; local official checkpoint projections retain width 32. This is shape compatibility, not a new model forward or learning result.
- Twelve regressions bring portable CPU checks to 205 passes; simulator Python 3.10 parsing passes. Physical result remains pending. Existing 40 processed 9D recordings need upper-command labels before this representation can train. Next gate is one integrated episode, then 3–5 distinct development episodes, not unlimited micro-ablations.

## 2026-10-29 — Live-initialized latched posture grasps successfully

- Both user runs complete 209 intervals / 627 physics and OSC updates with safe, independently verified grasping. Runtime anchor generation, raw robot-state correspondence and isolated input-change audits pass; 40 read-only checks pass per run.
- Both use the same seed/initial condition and reproduce all 15 core arrays exactly. Each retains 1.2 s bilateral contact, about 72.595 mm final lift and 0.160 mm relative-position jitter. This is exposed-case repeatability, not an independent success-rate estimate.
- Adopt the live-initialized posture configuration as a passing development baseline. Recorded target motion and other settings remain; no learned-policy success is claimed. Runtime unchanged in this result update; prior 193 CPU checks apply.

## 2026-10-29 — Robot-side latched-posture comparison prepared

- Add an opt-in posture anchor generated once from live initial joints and declared sampled robot variation, retained throughout the stream. Configuration excludes saved q/anchor and object state; native bounds and collision projection remain.
- First 173 requests exactly match the passing fixed-gain baseline; only the last 36 posture fields change. Save actual runtime requests and audit initialization, constant anchor, native resolved posture and raw robot state with a separate latched-posture gate.
- Seven new regressions bring the portable suite to 193 passes. Actual scene/previous-mode preflight, initializer fixture, saved baseline re-audit and Python 3.10 parsing pass. Physical result is pending; remaining recorded motion/settings dependencies are explicit. No GPU rollout launched.

## 2026-10-29 — Measured-posture ablation fails grasp retention

- User-run ablation completes 209 intervals / 627 physics and OSC updates safely. Request isolation and measured-posture audits pass, including exact correspondence to recorded robot q, but independent grasp retention fails.
- Bilateral contact spans frames 137–162 (0.433 s), final retained lift is approximately zero and relative-position jitter is 55.677 mm. The previous fixed-gain/recorded-posture run remains the passing baseline.
- Traced the original episode-latched posture to initial measured q plus sampled IK/homotopy variation; ordered native arithmetic reconstructs it exactly for this case without per-frame target truth. Test a robot-side posture configuration rather than assuming the restoring term can be deleted or must be learned. No runtime changes; prior 186 CPU passes apply.

## 2026-10-29 — Native measured-posture diagnostic prepared

- Add an opt-in posture probe based on the passing fixed-9/8 packet run: remove only the recorded seven-joint posture endpoint, using the existing native current-state redundancy branch. Exactly 173 input frames change; other requests and limits remain identical.
- Log and audit current measured q, native posture source and resolved posture, rejecting stale/nonfinite or saved-state resolution. Use a separate measured-posture physical gate; retain earlier modes and reports.
- Nine new regressions bring the portable suite to 186 passes. Scene and old fixed-gain command preflight, native posture branch CPU fixtures, saved baseline re-audit and Python 3.10 parsing pass. Physical result is pending; no GPU rollout launched.

## 2026-10-29 — Fixed-feedback-gain packet grasp passes

- User-run fixed-9/8 packet execution completes all 209 intervals / 627 physics and OSC updates safely. The independent grasp and fixed-gain gates pass.
- Read-only raw audit confirms that only the two feedback-gain fields change over frames 173–208. First 173 core frames exactly match the recorded-gain baseline; first changed action is 173 and measured state 174. All 37 provenance/execution checks pass.
- Bilateral contact lasts 1.2 s, final retained lift is at least 72.577 mm and relative-position jitter is 0.160 mm. These are slightly worse than the recorded-gain baseline but pass the same engineering criteria; no performance improvement or learned-policy result is claimed. Runtime unchanged, prior 177 CPU passes apply.

## 2026-10-29 — Isolated fixed-feedback-gain probe prepared

- Inventory of the passing explicit request stream identifies gains, posture/orientation, relative-speed and callback-mode schedules as remaining dependencies.
- Add opt-in `fixed-9-8` pose gains, gated on the successful recorded packet baseline. Exactly 36 frames change in this source; all other request fields and native limits remain unchanged. Runtime/analysis preserve profile provenance and report an independent fixed-gain gate.
- Seven new regressions bring the portable suite to 177 passes. Actual scene preflight, saved baseline packet re-audit, exact change isolation and simulator Python 3.10 parsing pass. Physical result is pending; no new GPU rollout.

## 2026-10-29 — Explicit Cartesian packet execution grasps successfully

- The corrected user-run packet controller completes 209 intervals / 627 physics and OSC updates safely, with no online task/skill decisions or expert joint-only terminal tail. Raw audit and contact recomputation match the saved report.
- First 194 references and all 15 core arrays match the exposed source exactly. The 15-frame Cartesian terminal hold retains the ball: 1.2 s bilateral contact, at least 74.861 mm retained lift and 0.131 mm relative-position jitter. No assistance or limit changes.
- Source numeric motion/settings remain explicit dependencies; this is one engineering gate, not VLA or generalization success. The unchanged collector FSM label remains false and is not relabeled. Runtime code is unchanged; the prior 170 CPU checks apply.

## 2026-10-29 — Restore measured robot motion in packet safety context

- Jaw-context correction allows the full safe 209-frame packet rollout, but retention fails. First arm-command divergence is frame 156; recorded robot/object states remain exact through that frame.
- Found another context omission: native terminal safety reads measured EE linear/angular velocities formerly refreshed by skill.step. Refresh them from robot-only pose history using native helpers, advance history once and audit the logged pose/twist sequence.
- Four new regressions bring the portable suite to 170 passes. Unchanged native CBF CPU fixtures confirm stopping-margin dependence, native motion helpers pass a 209-pose fixture, and scene preflight passes. Corrected physical outcome remains pending; limits and target-state restrictions are unchanged.

## 2026-10-29 — Synchronize explicit jaw commands with safety geometry

- Packet rollout reaches bilateral contact but aborts during lift: 174 completed intervals, first changed arm command at frame 151, with matching robot state/Jacobian/whole-arm halfspaces at that boundary.
- Identified omitted jaw context: explicit CLOSE starts at frame 118 while the geometry owner remains in fully-open mode. Synchronize that mode before servo evaluation and audit it per frame; closing is not a grasp certificate.
- Three new regressions bring the portable suite to 166 passes. Frozen native geometry-prefix checks and scene preflight pass. Limits, whole-arm collision code, source data and failed reports remain unchanged; corrected physical outcome is pending.

## 2026-10-29 — Fix optional safety-geometry protocol in packet adapter

- First physical packet attempt stops at frame 0: the adapter omitted the native optional `emission_surface` query, causing an existing fan mesh to be reported as missing. Reproduced offline with the frozen native geometry code.
- Forward non-target emitted geometry and use normal `AttributeError` semantics for unsupported optional attributes. Preserve target/subpath blocking and geometry completeness checks; missing paths now appear in errors.
- Three added regressions bring the portable suite to 163 passes. Native geometry matches direct queries with and without emitted surfaces on a CPU fixture; scene preflight passes. Physical rerun is pending; no task, limit or source-data changes.

## 2026-10-29 — Explicit request stream without online expert decisions prepared

- Added a packet boundary around native Cartesian feedback with explicit binary jaws and a 15-frame Cartesian hold replacing the expert joint tail. The online task/skill decision call is bypassed.
- Reuse native fixed jaw-drive initialization and geometry-only safety helpers; block target/subpath truth queries and reject missing non-target geometry. Recorded numeric settings/motion remain explicit dependencies, not a finalized VLA action contract.
- 160 portable CPU tests and real source/scene/native API preflight pass. Frozen cold geometry/jaw-drive initialization and actual patched decision execution were checked. The physical packet run is pending.

## 2026-10-29 — Upper-target live-feedback reexecution succeeds

- User-executed physical rollout completes 209 intervals / 627 physics updates / 582 OSC callbacks with safe, independently verified grasping. All 194 upper calls align; regenerated references and all 15 core arrays exactly match the source.
- Bilateral contact lasts 1.2 s, final retained lift is at least 74.861 mm, and relative-position jitter is 0.123 mm. Source files remain unchanged.
- Native expert scheduling, source settings, live safety callbacks, gripper and terminal joint fallback remain explicit dependencies. This is one exposed expert component result, not learned-policy or generalization success. Runtime code is unchanged in this evidence update.

## 2026-10-29 — Upper-target live-feedback reexecution prepared

- Added an opt-in diagnostic that regenerates native arm/OSC commands from captured upper goals and current robot state; recorded inner outputs are not used.
- Explicitly retain native expert timing, live safety callbacks, gripper and terminal joint fallback for this component gate. Callback descriptors are never executed; unexpected calls fail closed.
- Eight new regressions bring the portable CPU suite to 149 passes. Real source/scene/native API preflight and frozen Python 3.10 patch compilation pass; physical reexecution is pending.

## 2026-10-29 — Upper target capture verified; association audit corrected

- Original expert succeeds over 209 intervals, captures 194 upper Cartesian calls and exactly reproduces all 15 core arrays.
- Fixed an audit error: native reference consumption copies dictionary/array objects, so identity matching falsely rejected every call. Unique exact same-frame arm/full-reference content matching recovers all links offline; raw data and the old report are preserved.
- Four added regressions bring the portable suite to 141 passes. Inventory identifies changing gains/settings and callable safety dependencies; complete feedback execution remains pending. No simulation rerun for this correction.

## 2026-10-28 — Saved probe visualization helper

- Added a small wrapper around the existing frozen renderer for saved successful or failed HGwind probes, with a separate work copy, recorded camera scene and original-evidence hash checks.
- Real CPU preflight selects the requested zero-feedforward failure: one task / one trajectory at 30 Hz preview. No new simulation or GPU rendering was launched by the agent.

## 2026-10-28 — Original upper Cartesian target capture prepared

- Added an opt-in read-only observer of actual native pose-servo arguments and outputs, without changing original method arguments, return objects or exceptions.
- Link calls to executed OSC segments using returned arm references and the consumed reference object, retain callable descriptors and fallback frames, and require exact core-array reproduction before pairing with existing images.
- Eight new regressions bring the portable suite to 137 passes; actual scene and frozen patch preflight pass. One physical capture is pending; full Cartesian feedback execution remains unimplemented.

## 2026-10-28 — Task feedforward ablation outcome and dependency audit

- Zero-task-acceleration experiment completes all 209 intervals safely but makes no finger/object contact. Verified 110 altered commands, zero applied task acceleration and otherwise matching requested joint references/control fingerprints.
- Position divergence begins before contact; error reaches 50.122 mm at frame 15. CPU differencing of bounded task velocity does not reconstruct the separately supplied target acceleration.
- Preserve the passing reference and prioritize the complete Cartesian feedback contract. No controller changes, new simulation or policy-dimension increase in this evidence update; latest runtime suite remains 129 CPU passes.

## 2026-10-28 — Task feedforward ablation prepared

- Added an opt-in zero-task-acceleration comparison gated on the passing native-geometry baseline. Default logged references, gravity compensation and all safety limits are preserved.
- Record original/applied acceleration and use a separate ablation outcome; physical run is pending. This remains a partial expert-reference diagnostic, not a complete seven-dimensional policy contract.
- Six added regressions bring the portable suite to 129 passes. Matching baseline, scene and 209-frame CPU preflight pass.

## 2026-10-28 — Live native robot kinematics verified

- Native pinch geometry completes 209 intervals / 627 physics updates and passes independent grasp checks: 1.2 s bilateral contact and at least 74.861 mm retained lift.
- All 194 Jacobians recomputed from live robot state and all 15 core numeric arrays exactly match the exposed source episode. No cached original Jacobian executes.
- Other logged motor references and shadow bookkeeping remain; standalone policy execution and robustness are unverified. Documentation/evidence update only; latest runtime suite remains 123 CPU passes.

## 2026-10-28 — Post-contact divergence isolated for diagnosis

- Metric-corrected live-Jacobian run completes 174 intervals, with matching contact timing and 0.633 s bilateral contact, but fails lift and aborts during velocity-reserve recovery. Appreciable motion divergence begins after contact.
- Added an opt-in native pinch-geometry calculation that reuses frozen numerical operations on live robot state, excluding the servo/planner. Other references and limits remain fixed; physical outcome is pending.
- Three additional regressions bring the portable suite to 123 passes. Actual frozen geometry execution on CPU and the 209-frame scene preflight pass. No VLA result or robustness improvement is claimed.

## 2026-10-28 — Coupled task-metric adapter correction

- The first live-Jacobian diagnostic aborted after 150 completed intervals. Identified an adapter error: raw angular Jacobian rows were paired with task velocities already weighted by the original 0.03 m length scale.
- Restored the explicit native task metric; first-boundary matrix error decreases from 0.97 to 7.886e-10. Added all-source metric preflight and seven regressions; 120 portable CPU checks pass. Corrected physical execution is pending.
- Corrected the source-sequence narrative: its 15 joint-only fallback frames are the last frames (194–208), not startup. Preserved the failed run and original safety/physics/data.

## 2026-10-28 — Live kinematics component diagnostic prepared

- Added an opt-in current robot/pinch Jacobian branch around the passing full-control reference. Other logged signals, physics cadence and safety limits stay fixed; physical execution is pending.
- Added nine focused regressions; 113 portable CPU checks, simulator-source Python 3.10 parsing and a 209-frame scene preflight pass.
- Audited nonzero acceleration feedforward and geometry constraints before choosing action labels. A standalone VLA contract and ball-task training remain unverified.

## 2026-10-28 — Complete-control physical replay verified

- Logged q/qdot/OSC/limit replay completed 209 intervals / 627 physics updates and passed independent grasp checks. All 15 core datasets exactly match the original episode again.
- Preserved source-hashed evidence. Original cached Jacobians/constraints and shadow bookkeeping/safety remain; no live VLA interface or model success is claimed.
- No runtime control changes. Next verify matching action labels/execution using current robot kinematics before batch conversion or training.

## 2026-10-28 — Native expert episode reproduced exactly

- Original expert completed 209 intervals and independent grasp checks passed. All 15 core numeric datasets exactly match the source; this is one exposed expert case, not a VLA result.
- Recorded actual raw goals/paired velocities/OSC/policy limits; raw arm goals differ from exported actions by up to 0.046696 rad.
- Added a pinned complete-control-log replay diagnostic under the same original profile, and clarified untested/early-completion report fields. Six regressions added; 104 portable CPU checks pass. New physical replay pending.

## 2026-10-28 — Native-pose contact abort; original expert reference prepared

- Recorded a joint-5 overspeed/contact sequence and an empty acceleration/jerk/braking intersection. Constraints remain unchanged; full failure causality is not established.
- Corrected partial-run branch verification: 394 attempted OSC commands include the final rejected calculation; 393 callbacks / 131 intervals completed. Full success gates remain false.
- Added trace-only instrumentation around the frozen original expert profile and existing 2 mm depth helper for one fresh seeded reference attempt. Five regressions added; 98 portable CPU checks pass. New simulation pending.

## 2026-10-28 — Reject constructed task reference; reuse native pose servo

- The constructed OSC probe completed safely but had no bilateral grasp and worse pose tracking than joint tracking. Preserved both outcomes.
- Added a CPU endpoint/velocity semantics audit; naive cubic-Hermite interpolation also fails a sampled reference-limit check and is not installed as a replacement controller.
- Added native-pose delegation to the existing RoboPhys outer servo, preserving its matched endpoint/velocity/OSC tuple and source gripper timing. Seven regressions added; 93 portable CPU tests pass. Native-pose physical validation is pending.

## 2026-10-28 — Recorded-motion retention failure and task-space comparison

- Verified 208 completed control intervals, 624 callbacks and passing runtime safety, but contact was lost after a 0.65 s bilateral span; final lift/retention failed. Tracking error remains a candidate contributor, not a demonstrated sole cause.
- Added an isolated physics-rate tool-reference adapter to the existing RoboPhys OSC branch, retaining joint goals, gripper schedule, viability generator and final feasibility checks. Added live frame/Jacobian guards and branch-count verification.
- Eight regressions added; 86 portable CPU checks and a five-pose synthetic-inertia call into frozen native OSC passed. New physical rollout pending; no IK/VLA training result claimed.

## 2026-10-28 — Saved-target probe abort diagnosed

- Recorded a physical safety abort before jaw closure despite identical first states. Processed actuator references, missing original operational references and a different primary tracking branch prevent faithful replay; this is not an IK/VLA failure.
- Fixed aborted-run reporting and the collector output layout; distinguish 109 recorded states from 108 completed intervals. Added adapter-code fingerprint checks.
- Added a separate 208-transition recorded-motion diagnostic with explicit observed robot position/velocity references and unchanged gripper timing/constraints. Five regressions added; 78 portable CPU tests pass. New simulation remains pending.

## 2026-10-28 — Isolated effort-control comparison prepared

- Added a task-scoped saved-joint versus Cartesian/IK probe around the frozen RoboPhys effort executor, without editing upstream files or replacing effort control with position drives.
- Added source/version checks, discarded shadow motor references, per-frame assistance audits and an independent initial-state/contact/lift report. Cartesian execution requires a matching successful full joint baseline.
- Added 14 focused CPU regressions and included six existing grasp-evidence tests in the portable suite. Simulator execution remains pending; this is neither a VLA evaluation nor an exact reconstruction of unrecorded operational references.

## 2026-10-28 — Recorded tool-frame compatibility

- Added a metadata-defined Franka tool adapter and six synthetic regressions; all 40 ball episodes / 12,154 frames pass real-data FK checks, with maximum position discrepancy 0.001593 mm.
- Prepared one independent CPU action probe while preserving original data. Recorded effort-control execution differs from the old position-drive runner; no physics replay or training success is claimed.

## 2026-10-28 — Descriptive trajectory-quality screen

- Added independent CPU metrics for rotating-hand retention, recorded relative velocity, physical-substep versus control-bin contact and sampled target visibility; generated per-episode review sheets and video links for all 40 episodes.
- Recorded mixed source-group differences, 355/400 wrist samples touching the image edge, and the local traj_00019 contact-alternation/velocity case. No automatic exclusions, training-weight changes or causal depth-bias claim.
- Added 11 focused regressions and documented nine event-sheet spot checks separately from incomplete full-video review.

## 2026-10-28 — Full ball-cohort rendering and delivery verified

- All 40 camera-v2 episodes (21 main, 19 remote; 12,154 frames) passed the CPU audit for source identity, render completeness and exact 60 Hz frame/time alignment.
- All 19 remote episodes were uploaded and delivered to the main machine with pinned-revision, checksum-verified installation. A TLS interruption was recovered with sync-only resumption.
- No blank images were flagged in 600 sampled RGB frames. Human trajectory-quality review and VLA conversion remain pending; automatic completeness is not a learned-policy result.

## 2026-10-28 — Clean CI dependency correction

- The first GitHub run passed dependency installation and publication checks, then reported 35 passed / 1 failed because `huggingface_hub` was absent from the CPU dependency list. The earlier local test used an environment where it was already installed.
- Added the missing pinned dependency, selected Node-24-compatible Actions versions and fixed the runner OS to Ubuntu 24.04. Simulator and training environments are unchanged.
- Validation: all 36 focused tests passed in a newly created Python environment using only `requirements-cpu.txt`; `pip check` passed. GitHub run 40646715035 also passed all 36 tests.

## 2026-10-28 — Initial research repository preparation

- Documented the RoboPhys-to-VLA workflow, independent research scope and upstream attribution.
- Included existing project integration modules, diagnostic scripts and tests; kept datasets, simulator assets, checkpoints and machine configurations outside Git.
- Recorded the 40-success wind-driven-ball development cohort and the earlier DynamicVLA egg-task diagnostic with explicit limitations.
- Verified two revised-camera samples and the remote-render/Hub/local-delivery roundtrip; full-cohort rendering remains in progress at this snapshot.
- Added publication checks and a CPU-only regression workflow; 36 focused CPU tests passed locally and the staged file audit passed. Public repository creation and first remote publication are tracked separately from this local preparation.
