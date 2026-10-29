# Explicit Cartesian request boundary

## Purpose and verified scope

The exposed upper-target feedback run already succeeds. Its native expert still
selects when to call the pose API, commands jaws, refreshes safety context and
switches to a joint-only tail. The next isolated probe removes those online
decisions. `cartesian_packets.py` provides an explicit numeric request boundary
around the existing native robot servo; it does not select the final DynamicVLA
action dimension or certify a learned controller.

## Request and context contract

| Field or context | Convention | Current development source / remaining work |
| --- | --- | --- |
| `target_position` | Absolute world pinch-center position, meters | Captured upper goal; candidate policy output |
| `target_quaternion` | Absolute world pinch orientation, unit wxyz | Captured upper goal; map model rotation convention explicitly |
| Linear/angular velocity | World m/s and rad/s | Captured motion reference; policy-side generation remains unverified |
| Linear acceleration | World m/s², distinct from bounded task-velocity differences | Captured reference; no naive derivative substitution |
| Binary `gripper_closed` | Franka per-finger targets 0 m closed / 0.04 m open | Explicit source jaw command; not a contact-triggered expert decision |
| Pose gains, posture, orientation weights/axis and relative-speed settings | Original native API units | Recorded settings schedule retained explicitly; not yet a fixed controller profile |
| Callback mode | Two known native geometry helpers or explicit `None` | Rebound to an initialized geometry owner; no serialized callable executes |
| Timing | One request per 60 Hz tick; 180 Hz native effort/safety execution | Independent packet index; no task/skill `step()` |
| End of source upper stream | Hold last Cartesian pose, zero motion references | Explicit 15-frame hold replaces source joint-only fallback |
| Fixed actuator setup | Original native free-space jaw force/speed configuration | Reuse initialization helper once; no increase of motion/effort limits |

`execute_request(expert, packet, callbacks)` measures no task object: the native
`operational_space_pinch_action` reads the current robot, creates fresh arm/OSC
references, and a separate explicit `gripper_action` owns the jaws. Runtime
requests do not contain saved q/qdot/Jacobian/OSC outputs. The native controller
memory and existing 180 Hz feasibility/limit chain remain in use.

`PacketStreamExecutor.step` is called directly at the collector decision boundary.
The original `TaskRuntimeController.step` and skill decision loop are bypassed,
not run in shadow. It reuses an initialized skill object's geometry helpers and
fixed jaw-drive initialization only. A `TargetBlindSafetyWorld` adapter allows
non-target obstacle pose/velocity/mesh queries and blocks the target and every
child path. Blocked attempts are checked after geometry refresh even if the
native geometry helper catches the exception. Missing collision geometry stops
the probe before a motor request. Non-target geometry is declared safety context,
not an input to learned target prediction; this is not a real-robot perception
or safety certification.

This first probe **retains the recorded numeric controller-setting and motion
schedule** in its explicit packet stream. It does not prove that a 7D pose/gripper
policy suffices, or that the current settings generalize. Static configuration,
causal motion-reference generation and model label semantics still require a
validated contract before ball-task training. No old q-target label is silently
renamed as an upper Cartesian goal.

## Run the one exposed development case

```bash
cd /home/robophys/Capstone_Project_Yuan_Haoming

data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_effort_probe.py \
  --tracking native-expert \
  --packet-baseline-run data/derived/hgwind_effort_probes/20261029_011623_862187_joint_online-expert_native-expert_target-reexecution \
  --run
```

The historical `--tracking native-expert` option selects the frozen robot runtime;
`--packet-baseline-run` changes decision ownership to the explicit packet branch.
Preflight requires the passing retained-context reference, exact source/scene/SDK
provenance, and pinned native API, collision and Catch helper implementations.
Without `--run`, no simulator is created. The physical budget is 209 intervals
(about 3.48 simulation seconds), not the old 720-frame expert attempt budget.

The independent output directory ends in `_packets` and stores `packet_requests.json`,
`cartesian_packet_execution.json`, usual motor/physics traces and `report.json`.
`packet_controller_probe_passed` requires all requests, exact arm/OSC associations,
explicit jaw outputs, 60/180 Hz cadence, matching initial state, unchanged robot
limits, no grasp assistance, safe execution and independent contact/lift/retention.
The collector's expert-FSM success label is not this result: the FSM is not stepped.
A saved episode can therefore be called `failed_...` while only the independent
probe report answers the engineering question. Do not relabel it as official
benchmark or training success.

## Validation, rollback and pending result

160 portable CPU tests pass, including 11 new checks for current-state feedback,
explicit jaws, zero-motion Cartesian tail, target/subpath blocking (including
caught exceptions), missing collision geometry, input timing and actual patched
execution bypassing the original decision function. Real 209-frame packet
construction, trace setup, native cold geometry/jaw-drive initialization, frozen
worker/wrapper exports, Python 3.10 parsing and scene/native API preflight pass.
The existing retained-context target replay also still passes CPU preflight.
**The new physical packet probe is pending user execution.**

Omit `--packet-baseline-run` to restore the earlier paths. Original source data,
frozen RoboPhys, task physics and previous reports are unchanged. No GPU workload
was launched by the agent while preparing this interface.


## First-run protocol defect and correction — 2026-10-29

The first physical packet attempt stops at frame 0, before an arm request. Its
inventory contains every static background surface but lists the fan as missing.
This is an adapter defect, not evidence that the fan asset is absent: the native
geometry helper queries the optional `emission_surface` method after constructing
a rigid mesh. The wrapper rejected that attribute with `ValueError`; the native
helper catches the exception and classifies the whole body as missing.

A CPU reproduction using the actual frozen geometry method confirms a complete
rigid mesh under a direct world query and a missing mesh under the old wrapper.
The corrected wrapper forwards `emission_surface` for non-target paths, including
real emitted triangles, and raises `AttributeError` for unsupported attributes so
normal `getattr(world, optional_name, None)` capability checks work. Target and
child-path rejection, caught-access detection and geometry completeness checks
remain active. Error messages now include the missing paths.

Three added regression cases bring the suite to 163 passes. Frozen native geometry
produces exactly matching triangles and bounds with direct/corrected wrapped
queries, for both no-emission and emitted-surface CPU fixtures. Python 3.10 parsing
and real scene preflight pass. The original failure log is retained; the correction
has not yet been physically rerun. Use the same command above; it writes a new
output directory. See [the diagnosis](../../results/summaries/2026-10-29-packet-geometry-protocol-fix.json).
