# Original Cartesian target capture

The passing HGwind logs record the inner controller's q/qdot/OSC tuple, not the original upper Cartesian target. The failed zero-feedforward experiment and source audit show why reconstructing that target by relabeling processed drive references or differentiating bounded task velocity is not justified. The next step is a read-only observation of the actual original call, before implementing a complete target feedback executor.

## What is implemented

`CartesianTargetRecorder` wraps the original expert instance's `operational_space_pinch_action` only when `--trace-cartesian-targets` is requested. `inspect.signature` resolves parameter names/defaults for a snapshot, while the original method is called exactly once with the original arguments and objects. The returned action is passed through unchanged; original exceptions are recorded and rethrown. No pending reference is consumed by the recorder. The upstream files are not edited.

For each call, the local `cartesian_target_inputs.json` stores frame/call IDs; original target position/quaternion; optional target velocity/acceleration; gains, task metric, speed/limit and posture settings; returned joint action; and a snapshot of the pending OSC reference. Callables are described by module/name only. Unsupported and nonfinite values are explicit descriptors, not pickled objects or invalid JSON. These records are local research artifacts and remain outside public Git.

The original upper targets are world-frame pinch-center positions in meters and wxyz quaternions, per the native API; they must not be confused with recorded tool poses or Euler action labels. This does not select a VLA action dimensionality. Defaults and varying inputs still need classification after capture.

The frame hook starts immediately before the original controller step. Each emitted motor segment lists all observed call IDs, arm-output matches, and the call selected by identity of the pending operational-reference object consumed into the segment. This avoids choosing a last call by assumption when several calls have identical arm outputs. No Cartesian target is fabricated for joint-only fallback intervals. Callable state, collision projectors and controller memory are not thereby reconstructed; standalone replay remains unverified.

## Run one original expert episode

```bash
cd /home/robophys/Capstone_Project_Yuan_Haoming

data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_effort_probe.py \
  --tracking native-expert --trace-cartesian-targets --run
```

This restores the original smooth-lift/depth expert profile and its original acceleration feedforward. The 720-frame value is the maximum 12-second simulation budget. Normal expert success can end the run early; the previous matching source ended at 209 frames. This is one exposed development case, not a new cohort or a VLA rollout.

The `_target-capture` output directory contains the usual physics trace/report, `original_expert_control_references.json` with call linkage, and `cartesian_target_inputs.json`. `cartesian_target_capture_verified` additionally requires native grasp/safety success, frame coverage, unique selected-call linkage for every OSC segment, and exact numeric equality of source/captured core arrays. File-byte equality is not expected because run metadata differs. If equality fails, the new targets must not simply be paired with old rendered observations.

## Validation and rollback

Eight focused tests cover unchanged argument/return identity, snapshots, unconsumed control references, callable descriptors, exception propagation, multiple-call selection, fallback intervals, malformed values and patch boundaries. All 137 portable CPU tests pass. The real frozen profile/collector patches compile, Python 3.10 parsing passes for simulator modules, and the CPU scene preflight succeeds. The physical capture is pending; no successful capture or complete feedback execution is claimed yet.

Omit `--trace-cartesian-targets` to retain the previous native-expert path. No source demonstrations, upstream assets, robot limits or physics settings are changed. A future target-feedback adapter must explicitly handle callable geometry dependencies, controller state, velocity/acceleration semantics, timing and allowed policy observations; capturing arguments is preparation for that work, not proof of completion.
