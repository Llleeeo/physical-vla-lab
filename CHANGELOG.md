# Changelog

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
