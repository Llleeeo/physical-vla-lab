# Results and limitations

Snapshot: **2026-10-29, Asia/Hong_Kong**. These are development observations, not a benchmark leaderboard or a paper's final test results.

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
