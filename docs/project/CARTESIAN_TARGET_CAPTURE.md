# Original Cartesian target capture

The passing HGwind logs record the inner controller's q/qdot/OSC tuple, not the original upper Cartesian target. The failed zero-feedforward experiment and source audit show why reconstructing that target by relabeling processed drive references or differentiating bounded task velocity is not justified. The next step is a read-only observation of the actual original call, before implementing a complete target feedback executor.

## What is implemented

`CartesianTargetRecorder` wraps the original expert instance's `operational_space_pinch_action` only when `--trace-cartesian-targets` is requested. `inspect.signature` resolves parameter names/defaults for a snapshot, while the original method is called exactly once with the original arguments and objects. The returned action is passed through unchanged; original exceptions are recorded and rethrown. No pending reference is consumed by the recorder. The upstream files are not edited.

For each call, the local `cartesian_target_inputs.json` stores frame/call IDs; original target position/quaternion; optional target velocity/acceleration; gains, task metric, speed/limit and posture settings; returned joint action; and a snapshot of the pending OSC reference. Callables are described by module/name only. Unsupported and nonfinite values are explicit descriptors, not pickled objects or invalid JSON. These records are local research artifacts and remain outside public Git.

The original upper targets are world-frame pinch-center positions in meters and wxyz quaternions, per the native API; they must not be confused with recorded tool poses or Euler action labels. This does not select a VLA action dimensionality. Defaults and varying inputs still need classification after capture.

The frame hook starts immediately before the original controller step. Each emitted motor segment lists all observed call IDs, arm-output matches, and the unique call with identical full OSC reference contents in the same frame. Arm position/velocity use the original float32 merge convention; reference fields must match exactly and be finite. Ambiguous duplicate matches remain unresolved. The original identity-based check was incorrect because the native consume method copies dictionaries/arrays; the corrected audit can recover links from existing serialized evidence without changing it. No Cartesian target is fabricated for joint-only fallback intervals. Callable state, collision projectors and controller memory are not thereby reconstructed; standalone replay remains unverified.

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

Twelve focused tests cover unchanged argument/return identity, snapshots, unconsumed control references, callable descriptors, exception propagation, multiple-call selection, fallback intervals, malformed values, copied references, exact-content mismatch rejection, ambiguous references and patch boundaries. All 141 portable CPU tests pass. The real frozen profile/collector patches compile, Python 3.10 parsing passes for simulator modules, and the CPU scene preflight succeeds. The physical capture dated 2026-10-28 succeeds and its offline revalidation on 2026-10-29 passes, as detailed below. Complete feedback execution remains unverified.

Omit `--trace-cartesian-targets` to retain the previous native-expert path. No source demonstrations, upstream assets, robot limits or physics settings are changed. A future target-feedback adapter must explicitly handle callable geometry dependencies, controller state, velocity/acceleration semantics, timing and allowed policy observations; capturing arguments is preparation for that work, not proof of completion.


## Verified capture and offline association correction — 2026-10-29

The original expert completes 209 intervals with passing safety and independent grasp evidence. All 15 core numeric arrays exactly match the source. It captures 194 successful upper Cartesian calls for the 194 OSC segments; frames 194–208 remain joint-only fallback. No missing Cartesian targets are fabricated.

The first report marked target alignment false because the recorder used Python object identity. `Expert.consume_operational_space_reference()` explicitly returns a copied dictionary, including copied NumPy arrays, so every identity check failed despite identical data. The corrected audit uniquely matches all 194 calls by frame, returned arm tuple and exact full reference contents. It rejects ambiguous matches and even 1e-12 reference differences; numerical tolerances were not widened to pass this run. Existing input/control JSON files and physical traces are unchanged. The old report is preserved separately; `cartesian_target_capture_verified=true` follows offline reanalysis, not another simulation.

The local parameter inventory finds position/orientation gains 9/8 on 173 calls and 4/4 on the remaining 21 calls, with a transition at frame 173. It also records varying orientation/limit settings, a posture reference on 173 calls, and `cartesian_velocity_projector` / `joint_velocity_projector` callable descriptors. These are observed dependencies in one expert episode, not an approved runtime mode schedule or policy observations. The [result evidence](../../results/summaries/2026-10-29-cartesian-target-capture-result.json) keeps that scope explicit.

Next, use these actual upper goals and motion fields to generate q/qdot/OSC from current robot state through a documented feedback interface. Resolve the scene-safety callbacks, phase-varying controller settings, posture choice and terminal hold behavior explicitly. Do not replay expert phase timing or privileged object-state logic as undisclosed policy inputs. Target capture is complete for this case; standalone replay, VLA action dimensionality and ball-task training remain pending.


## Upper-target feedback reexecution prepared — 2026-10-29

The opt-in `--cartesian-target-run` diagnostic now passes recorded upper position,
wxyz orientation, velocity, acceleration and controller settings to the original
`operational_space_pinch_action`. That function measures the current articulation
and regenerates the arm q/qdot and pending OSC reference. Recorded returned joint
outputs, Jacobians and pending OSC fields are deliberately discarded by the
runtime target reader. An added CPU regression perturbs the measured robot state
and confirms that the same upper goal produces a different newly computed action.

This first diagnostic deliberately retains the native expert call schedule,
current native safety callback objects, gripper execution, controller memory and
15-frame joint-only terminal fallback. Recorded scalar/posture/orientation settings
also retain their source schedule. Callback descriptors are never executed or
deserialized: only the two known native callback names are accepted, and each must
match the callback actually supplied in that live call, including `None`.
Unexpected, missing, reordered or exhausted calls stop execution; no last target
is silently repeated. Thus this is an exposed expert component test, not a
standalone policy controller, a disturbance experiment or a learned-policy result.

```bash
cd /home/robophys/Capstone_Project_Yuan_Haoming

data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_effort_probe.py \
  --tracking native-expert \
  --cartesian-target-run data/derived/hgwind_effort_probes/20261028_235048_763413_joint_online-expert_native-expert_target-capture \
  --run
```

Without `--run`, the command checks source/capture alignment, task physics,
Isaac versions, frozen native API/safety file hashes and the real native parameter
signature; it never creates a simulator. Source input/report/control-log hashes
are recorded. Each executed call and its regenerated output/reference go into a
new `cartesian_target_reexecution.json`; original source files are not edited.

`cartesian_feedback_replay_passed` requires the complete 209-frame/194-call stream,
exact applied input correspondence, unique call-to-segment associations, 60/180 Hz
cadence including three OSC callbacks per Cartesian call, matching initial state,
runtime safety, no grasp assistance and independent contact/lift/retention checks.
All 15 source core-array comparisons are additionally reported; physical success
is not defined by exact state equality. The original unchanged-expert success gate
is left null for this new intervention. The portable CPU suite has 149 passes;
actual frozen worker/wrapper Python 3.10 compilation and real CPU preflight pass.
**Physical reexecution remains pending user execution.**

Omit `--cartesian-target-run` to use the previous native-expert path. The next gate,
after this component succeeds, is to remove the retained expert scheduling and
safety-context dependencies through an explicit runtime controller contract before
ball-task action labeling or DynamicVLA training.
