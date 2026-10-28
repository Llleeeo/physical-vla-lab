# Changelog

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
