# Changelog

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
