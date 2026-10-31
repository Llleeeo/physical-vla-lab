# Ball startup target and nominal controller audit

## Scope

This CPU-only diagnostic uses the exposed training episode `local/traj_00018`, its successful native teacher, the passing float32 surrogate labels and saved 200/2,000-step model predictions. It performs no new model inference, simulation, training or rendering. It leaves source labels, controller gains, motion limits and scene physics unchanged.

The [source-hashed summary](../../results/summaries/2026-10-31-ball-startup-target-audit.json) contains frame examples and phase metrics. The [analysis entry](../../scripts/analyze_hgwind_startup_targets.py) defaults to read-only; `--write` saves a fresh derived report. It requires the separately provisioned local artifacts described in [reproduction boundaries](REPRODUCIBILITY.md).

## What the target jumps mean

All 12 consecutive-frame upper position jumps over 100 mm already occur in the expert capture, before frame 42. These are desired controller targets, not observed single-frame robot displacements. Captured calls are consecutive and their float32 target positions exactly equal the training labels.

The ball initially falls and rebounds. Its recorded vertical velocity changes from below -0.15 m/s to above +0.15 m/s at frames 8, 18, 25, 31 and 35. This threshold is a descriptive event screen, not a task success criterion. The frozen expert computes an intercept target and a dynamic pregrasp clearance from relative motion and braking capability, then applies geometric and controller constraints.

At frame 18, ball height is about 30.17 mm and upward velocity is 0.493 m/s. The recorded safe pregrasp standoff is 362.79 mm; the live grasp target height is about 392.96 mm. The original local depth helper lowers the captured target by 2 mm, to 390.96 mm. Several upward target peaks numerically match this standoff construction. This explains a concrete contribution to the large targets; it does not reconstruct every branch or establish why the model learns them poorly.

Relevant frozen RoboPhys functions, resolved through the pinned worker bundle:

- `Scripts/expert_data/skills/catch.py`: `_minimum_time_contact_state`, `_center_pointing_track_step`, `_open_hand_pregrasp_goal`, `_live_pregrasp_standoff_goal` and `_approach_action`.
- `Scripts/expert_data/expert.py`: `operational_space_pinch_action`.
- `Scripts/expert_data/kinematic_servo.py`: `linear_task_velocity` and `_bounded_linear_command`.

The published repository does not vendor that frozen benchmark source. Its bundle manifest and the source hashes are checked by the diagnostic.

## Do speed bounds remove the learned error?

Not completely. The diagnostic decodes each predicted **first action** and calls the actual frozen pure-NumPy translation function at the **recorded robot state**. This includes position feedback, predicted linear feedforward, the absolute speed bound and the optional relative speed/reference bound.

| Phase, frame range | 2,000-step mean request difference | p95 | Accepted / total |
|---|---:|---:|---:|
| Startup, 0–41 | 0.569 m/s | 1.470 m/s | 42 / 42 |
| Approach, 42–143 | 0.116 m/s | 0.223 m/s | 102 / 102 |
| Closure to source contact, 144–161 | 0.096 m/s | 0.174 m/s | 18 / 18 |
| Source contact onward, 162–268 | 0.167 m/s | 0.357 m/s | 107 / 107 |

These are differences between model and expert **nominal velocity requests**, not measured robot velocity errors. Collision projection, orientation, Jacobian/IK, effort feasibility, contact dynamics and live feedback are excluded. Subsequent constraints can materially change these requests. The 42-frame startup partition is specific to this target audit; the earlier pose report used 30 frames. Do not compare those aggregates as if they had identical denominators.

There is also a semantic finding: 19 of 269 first actions from the 200-step model fail the native nominal-request function because a relative-speed cap is enabled without its required velocity reference. Structural decoder acceptance alone did not catch this. Three rejected actions occur in startup, so the 200-step startup mean of 0.925 m/s covers only 39 valid requests and is not an exact paired 42-frame comparison. The 2,000-step model has zero such rejects in these 269 first actions. This is not a validation of all 20 predicted chunk positions or physical feasibility.

## Decision and next implementation

Keep the 2,000-step checkpoint and current action/controller contract. This audit does not justify smoothing or deleting expert startup labels, relaxing safety limits or immediately increasing the training budget.

Prepare one bounded live integration on the already exposed case:

1. Use calibrated live `opst_cam` / `wrist_cam`, history offsets [-2, 0], measured robot state30 and the task instruction. Do not read recorded expert actions or object ground truth to generate policy commands.
2. Decode absolute29 predictions once, with the same normalizers and latched robot posture. Reject inconsistent native inputs explicitly, with no expert fallback. Preserve the native geometry/effort checks.
3. First execute eight control intervals to check live preprocessing, timing, request consumption and logging. A short check cannot assess the whole startup or grasp. If it passes, evaluate the full episode with independent contact/lift/retention evidence and retain every outcome.
4. Separate observation mismatch, invalid commands, controller tracking and learned grasp failure. Record the synchronous inference pause; it is not a real-time latency evaluation.

The new 29D live adapter was **not implemented or validated by this audit**. A subsequent [implementation milestone](BALL_LIVE_POLICY.md) now passes CPU checks; its physical run remains pending. The older 7D egg-task live adapter cannot be reused unchanged. No learned ball grasp, held-out generalization or disturbance robustness result is established.
