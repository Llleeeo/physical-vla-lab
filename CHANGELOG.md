# Changelog

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
