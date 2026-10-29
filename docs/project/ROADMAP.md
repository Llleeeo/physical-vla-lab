# Research roadmap

## 1. Complete and review the spherical-object baseline data

Finish the revised-camera cohort, validate every frame's alignment, and review visibility through approach, closure and lift. Evaluate quality using contact persistence, relative slip, lift retention and motion smoothness. Do not discard inconvenient failures from the experiment log.

Gate: immutable source manifests, complete visuals and explicit accepted/excluded/uncertain episode decisions.

## 2. Establish learned closed-loop grasping

Validate the ball-task action conversion by replaying expert-derived commands before attributing failures to the policy. The isolated [effort-control comparison](HGWIND_EFFORT_PROBE.md) is prepared; first require a complete physical saved-joint baseline, then test Cartesian/IK with matching execution fingerprints. The first saved-target baseline aborted before closure. Observed-motion joint tracking completes safely but loses retention; the constructed task-space reference performed worse. CPU endpoint-semantics diagnostics now motivate delegation to the original outer pose servo, which aborted after contact. The original expert reference now reproduces all 15 core arrays exactly for one source episode. Complete-control replay now also exactly matches the source and passes grasp checks. The current robot/pinch Jacobian component run exposed a missing angular task scaling in the added adapter. The shifted-tool calculation diverges after contact; the native geometry implementation now recomputes the live Jacobians and exactly reproduces this one successful episode. Zeroing task-acceleration feedforward now completes safely but misses the object; naive velocity differentiation also does not reconstruct that input. Retain the passing native-geometry/logged-feedforward diagnostic and define the full upper Cartesian target feedback contract before changing the policy action dimension. The read-only [original Cartesian target capture](CARTESIAN_TARGET_CAPTURE.md) now supplies 194 verified upper calls with exact source-array reproduction. Upper-target reexecution through current robot feedback now passes physical grasp checks and exactly reproduces this source episode. Preserve that reference, then replace native expert timing, source settings schedule, safety-context dependencies, gripper and terminal fallback with a documented policy-facing contract. The prepared [explicit packet probe](CARTESIAN_PACKET_INTERFACE.md) removes online task/skill decisions and the jaw/joint-tail ownership while retaining recorded numeric settings and geometry-only safety context. Its corrected physical run now passes 209 intervals and independent grasp checks on one exposed seed. Preserve that passing packet baseline, then define/test the remaining fixed settings and motion-reference contract; recorded schedules are still dependencies. The first one-factor comparison fixes pose gains at 9/8 and now passes the same physical grasp criteria on this exposed case. Preserve both passing configurations; the remaining posture/orientation/callback settings and motion references still need an explicit policy-facing contract. A complete learned-policy executor remains pending. Remaining frozen command/constraint inputs still need removal before policy evaluation. Establish a live policy action contract with matching labels and current-state kinematics before batch conversion or training. Do not attribute this failure to IK or the policy. Freeze episode-level train/validation partitions and fit normalization on training data only. Train a pretrained DynamicVLA baseline and evaluate actual camera-feedback rollouts. Offline fitting and synchronous simulation latency are reported separately.

A uniform episode-sampling baseline comes first. If quality review supports it, compare modest quality-aware weighting under the same training budget and fixed evaluation distribution. A possible 60:40 source-group mixture is a proposal, not an implemented or proven improvement.

Gate: repeatable closed-loop performance with contact/lift evidence and documented failure modes. A single successful rollout is insufficient.

## 3. Test physical generalization and disturbance recovery

Define new held-out seeds and physical conditions before tuning on them. Change one meaningful factor at a time, including controlled motion disturbances where appropriate. Measure success, capture timing, retention, recovery and failure categories; report sample counts and uncertainty. Keep visual, controller and latency confounds explicit.

The egg-object task can serve as an exposed development hard case. It cannot retrospectively become an untouched test set. DynamicVLA disturbance weakness is a hypothesis to evaluate, not an accepted conclusion.

Gate: comparable frozen baselines and a defensible attribution of any improvement.

## 4. Extend embodiment and deployment

UR10e simulation requires its own robot/gripper model, reachable-workspace checks, camera calibration and action execution interface. Real-world testing additionally requires latency characterization, safety review and an authorized human operator. Cross-embodiment adaptation and sim-to-real transfer are separate workstreams.
