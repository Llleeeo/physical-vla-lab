# Recorded target-height diagnostic

## Evidence and scope

The refined checkpoint completes 269 control intervals/807 physics callbacks without a safety abort, but neither finger contacts the ball. Jaw commands change nine times (first112, last131), compared with one expert transition144. Ball position first differs by more than1mm from the successful source at166. Pinch-to-ball-center distance reaches8.17mm at138, which is not a measured contact clearance. From138 to156 the live finger midpoint rises15.47mm while the teacher midpoint descends15.03mm. [Source-hashed result](../../results/summaries/2026-10-31-ball-refined-live-result.json).

This is a descriptive near miss. It does not prove height alone causes failure, and less jaw chatter does not establish grasp improvement. Do not add training steps based on this observation alone.

## Fixed intervention

Use the same selected checkpoint, initial condition, legacy cameras, prefix4, 60Hz control, physical gripper and native effort/safety controller. At the per-frame action boundary, replace **only absolute world target-position z (channel2), frames138 through179 inclusive**, with the verified dataset's recorded expert target. Frame138 is included;180 is excluded. Outside that window, all29 channels come from the model. Velocity, acceleration, masks, orientation, horizontal targets and gripper remain model outputs. The intervention has no online object-state lookup, expert skill step, hidden controller changes, smoothing or endpoint blend.

It is a target-height test, not a replacement of the complete vertical control request. Remaining model velocity/feedforward references can still disagree with the substituted height. A failure therefore cannot rule out a broader vertical-control issue. A success supports an effect in this case but is **hybrid diagnostic success**, not unassisted VLA performance. Recorded time indexing is privileged diagnostic information, not a deployable correction.

The window is fixed before running. In future frames, all unmodified channels are generated from the intervention's actual camera/state feedback; they need not equal predictions from the old divergent rollout. Compare initial/prefix behavior and actual contact chronology, not an assumed identical future action sequence.

## Evidence and checks

`live_observations.hdf5/model_action` retains original predictions; `action` retains executed29D requests. IPC responses remain unmodified. The analyzer verifies raw response binding, exact identity of all non-z channels, exact identity outside the window, agreement of substituted heights with the hashed reference payload and actual native packet execution. Flags set `source_actions_used_for_control=true`, `learned_policy_evaluated=false`, `standalone_policy_evaluation=false`. The distinct `BALL_VERTICAL_DIAGNOSTIC_PASS` status means integration passed; inspect `independent_grasp.status` separately.

CPU unit tests cover window boundaries, no mutation of model arrays, invalid units/reference rejection, provenance, and actual mocked motor calls/archive retention. The real checkpoint strictly reloads and live preprocessing matches on CPU. Simulation has not been run by the agent.

## Run

```bash
cd /home/robophys/Capstone_Project_Yuan_Haoming

data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_closed_loop.py \
  --run-dir data/derived/hgwind_traj18_development_20261031/training_runs/20261031_133555_985430 \
  --frames 269 --prefix 4 --camera-protocol legacy \
  --diagnostic-vertical --run
```

Expect roughly5–6minutes based on the preceding run. This uses GPU inference/rendering/physics, with simulation paused during inference; no real-time result. Each invocation writes a fresh `closed_loop_runs` directory under the training run. Omit `--run` for CPU preflight, optionally add `--check-load`. Omit `--diagnostic-vertical` to restore ordinary model-only operation. Existing reports, model weights, videos and raw recordings are retained.

Assess first contact and bilateral contact, center/height alignment through138..180, jaw timing, sustained lift, and safety status. If the diagnostic succeeds, improve and validate the learned outputs; do not carry the recorded-height substitution into reported policy evaluation. If it fails, inspect which prerequisite remained unmet before selecting another intervention.
