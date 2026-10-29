# Research roadmap

## 1. Complete and review the spherical-object baseline data

Finish the revised-camera cohort, validate every frame's alignment, and review visibility through approach, closure and lift. Evaluate quality using contact persistence, relative slip, lift retention and motion smoothness. Do not discard inconvenient failures from the experiment log.

Gate: immutable source manifests, complete visuals and explicit accepted/excluded/uncertain episode decisions.

## 2. Establish learned closed-loop grasping

**Current decision: stop successive parameter-removal probes.** The native expert, upper-target feedback and explicit packet controller have passed independent grasp checks on one exposed source episode. Fixed 9/8 gains and a live-initialized latched robot posture also pass; two same-seed repeats are identical. These establish engineering references, not learned or cross-trajectory success. Historical mismatches/results remain documented in [effort probes](HGWIND_EFFORT_PROBE.md) and [packet interface](CARTESIAN_PACKET_INTERFACE.md).

Next, follow the [complete ball action contract](BALL_ACTION_CONTRACT.md):

1. Run one integrated 29D float32 command surrogate through the native controller, with fixed gains/live posture and always-on collision callbacks. Require full timing, actual motor-boundary decode, raw robot-state correspondence and independent grasp retention. CPU/model-shape and actual command-boundary checks pass. The first 209-interval run completes safely but loses final grasp retention.
2. Freeze controller parameters and action representation for a bounded **3–5 distinct exposed development-episode diagnostic**. The first failure shows contact-stage sensitivity, not a decoded-field or safety abort; departure begins before changed callback activation. This batch diagnoses whether the failure is case-specific or widespread. Retain every outcome; it is not a training-readiness certificate or a test success-rate estimate. After that batch, explicitly decide whether to continue with the contract or reconsider it; no open-ended parameter-removal campaign.
The [bounded video batch](CONTRACT_VIDEO_BATCH.md) now completes after the recording fix. Native teacher/29D-surrogate pair traj_00018 passes (269 intervals, at least57.277 mm lift); teachers1/7 abort on motion feasibility before surrogate tests. All four new saved runs have five verified RGB review views. Preserve these failures and the earlier surrogate retention failure. Offline diagnosis now places ball velocity divergence at the first180Hz physics step, before arm divergence. The [passive force measurement](STARTUP_PHYSICS_DIAGNOSTIC.md) now reproduces teacher1 failure exactly in15 compared arrays and provides five review videos:10 observed force calls,4before recording,0zero-configured-dt calls. This does not establish a force cause; the earlier native crash also remains unexplained. Next proposed bounded experiment: regenerate original attempts58/59/60 in one worker and compare attempt60 startup to the existing fresh-worker control. It has not been implemented or run; preserve all attempts and render each saved outcome. Keep controller settings frozen until startup-history evidence supports a specific change. Binary-label schema exclusions remain explicit.

3. Establish the required upper-command labels. Existing 40 ball recordings contain processed 9D targets, not the complete 29D command envelope; preserve them and decide controlled label reexecution versus new collection after the small validation batch.
4. Freeze episode/physical-condition train/validation/test definitions, fit normalization on training episodes only, and perform a genuine 29D pretrained-model smoke before fine-tuning. Then evaluate camera-feedback Franka rollouts, with latency reported separately. No held-out physical-generalization claim follows from same-task development fitting.

A uniform episode-sampling baseline comes first. If quality review supports it, compare modest quality-aware weighting under the same training budget and fixed evaluation distribution. A possible 60:40 source-group mixture is a proposal, not an implemented or proven improvement.

Gate: repeatable closed-loop performance with contact/lift evidence and documented failure modes. A single successful rollout is insufficient.

## 3. Test physical generalization and disturbance recovery

Define new held-out seeds and physical conditions before tuning on them. Change one meaningful factor at a time, including controlled motion disturbances where appropriate. Measure success, capture timing, retention, recovery and failure categories; report sample counts and uncertainty. Keep visual, controller and latency confounds explicit.

The egg-object task can serve as an exposed development hard case. It cannot retrospectively become an untouched test set. DynamicVLA disturbance weakness is a hypothesis to evaluate, not an accepted conclusion.

Gate: comparable frozen baselines and a defensible attribution of any improvement.

## 4. Extend embodiment and deployment

UR10e simulation requires its own robot/gripper model, reachable-workspace checks, camera calibration and action execution interface. Real-world testing additionally requires latency characterization, safety review and an authorized human operator. Cross-embodiment adaptation and sim-to-real transfer are separate workstreams.
