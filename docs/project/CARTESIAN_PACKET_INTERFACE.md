# Explicit Cartesian request boundary

## Purpose and verified scope

The exposed upper-target feedback run already succeeds. Its native expert still
selects when to call the pose API, commands jaws, refreshes safety context and
switches to a joint-only tail. The explicit packet probe now removes those online
decisions and passes independent physical-grasp checks on one exposed case. `cartesian_packets.py` provides an explicit numeric request boundary
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


## Jaw geometry context defect and correction — 2026-10-29

After the optional-query fix, the packet run reaches first contact at frame 135
and bilateral contact at frame 137, but aborts during post-contact motion after
174 completed intervals (175 recorded states). It has 0.633 s bilateral contact
and about 19.14 mm retained lift before the abort. Its source upper inputs match
exactly. Recorded robot states remain exact through frame 151; the first arm
command difference is at frame 151 and measured-state divergence begins at 152.
The Jacobian and whole-arm constraint matrix/bounds still match at that first
changed command, directing investigation to the terminal projection context.

The source CLOSE command and source geometry-owner `_jaws_closed` flag both start
at frame 118. The new packet actuator issues that CLOSE correctly but omitted the
geometry-owner flag, which stays false in its recorded telemetry. The native
terminal projector uses this flag to decide whether the fixed fully-open finger
proxy is valid. Native comments explain that closing fingers move relative to
the pinch frame; their ongoing collision checks remain in the whole-arm model.
A frozen native geometry-prefix CPU fixture confirms the open/closed/reopen mode
switch without changing the whole-arm code or margins.

The packet adapter now copies the explicit commanded jaw mode into safety context
before geometry/servo evaluation and records `safety_gripper_closed` per call.
The audit requires correspondence to the packet. It does not set physical-grasp
or contact flags, run a skill decision or read target truth. Three new regressions
cover ordering, close/reopen propagation, stale-context rejection and preservation
of independent contact/skill state. All 166 CPU checks and scene preflight pass.
The failed run is retained. **Full grasp restoration remains pending a physical
rerun with the same command above.** See [the source-hashed diagnosis](../../results/summaries/2026-10-29-packet-jaw-context-diagnosis.json).


## Robot-motion safety context correction — 2026-10-29

The jaw-context-corrected packet rollout completes all 209 intervals safely and
passes explicit request/jaw/OSC alignment, but fails final grasp retention. First
arm-command divergence is at frame 156; recorded robot and object pose/velocity
arrays remain exact through that frame, and the first state difference is 157.
The first changed command still has the same robot Jacobian and whole-arm
halfspaces as the passing reference. The measured jaw-mode audit passes.

Native terminal collision safety also reads `_current_ee_linear_velocity` and
`_current_ee_angular_velocity` from its geometry owner to reserve stopping distance.
Those values were updated by the original skill tick and remained at zero after
that tick was bypassed. The failed packet stopping-margin telemetry is zero where
the original reference reports nonzero values. This is a verified context defect;
the complete causal recovery still requires a physical rerun.

The adapter now measures only `expert.pinch_pose()`, invokes the native causal
linear/angular difference helpers, and refreshes these fields before geometry and
servo evaluation. It preserves the native quaternion snapshot/normalization order
and advances position/orientation history once after the current servo call. It
does not call the skill finalizer, which would also record target-state data.
Each event logs `safety_robot_pose` and `safety_robot_twist_world`; the audit checks
linear differences and shortest-arc world angular differences from consecutive
measured poses. These are measured robot velocities, not derivatives of commanded
goals or the separate task-acceleration feedforward.

A small unchanged native CBF plane/palm fixture shows a zero stopping margin with
stale zero velocity and a 27.5 mm margin with its specified measured velocity.
The projected approach command correspondingly changes; these fixture numbers
are not a tuning recommendation. Native motion helpers and the causal audit also
pass a 209-sample recorded-robot-pose CPU fixture. Four new regressions bring the
portable suite to 170 passes, including update ordering, history advancement,
stale-context detection and invalid-pose rejection. Python 3.10 parsing and scene
preflight pass. Original reports and limits remain unchanged; the same physical
command above must still be rerun to determine final grasp performance.


## Physical packet gate passed — 2026-10-29

After restoring robot-motion safety context, the same packet command completes
209/209 intervals with 627 physics updates and 627 native OSC callbacks. Packet
alignment, commanded-jaw context, causal robot-motion context, initial-state
identity and runtime safety all pass. Read-only recomputation of the packet audit
and raw 180 Hz contact evidence matches the saved report; 30 source/execution
hash checks pass. Original reports and source files remain unchanged.

The first 194 control-reference rows and all 15 core HDF5 arrays for those frames
exactly match the passing source. The first changed action is frame 194 and first
changed measured robot/object position is 195, when the explicit Cartesian hold
replaces the source joint tail. Full-episode exact equality is not claimed.
Independent checks measure 72 consecutive bilateral-contact bins (1.2 s), at
least 74.861 mm final retained lift and 0.131 mm relative-position jitter. The
object moves at 0.2263 m/s immediately before first contact. No grasp assistance
or manufacturer-limit changes are present.

The original collector still saves `failed_00000` with FSM success false: this
probe bypasses the online decision loop that would update that FSM. Those files
and labels remain unchanged. `packet_controller_probe_passed=true` and the
independent contact/lift/retention report are this engineering experiment's
result; official benchmark certification is not asserted.

This removes online expert decisions, explicit jaw ownership and joint-tail
dependencies. Recorded upper motion, numeric settings/posture schedule, native
servo state and non-target geometry safety remain. It does not establish a
pose-only DynamicVLA action contract. Next, define and test a fixed policy-facing
controller profile and a compatible motion-reference construction against this
passing baseline before batch conversion/training. See the [source-hashed
result](../../results/summaries/2026-10-29-cartesian-packet-success.json).


## Fixed-feedback-gain diagnostic prepared — 2026-10-29

The successful stream uses 33 numeric API fields plus two callback-mode fields.
They are not all model actions. In the 194 source calls, pose gains switch from
9/8 to 4/4 at frame 173; posture, joint-limit margin and orientation-task mode
also change there. Roll weight switches at 135, orientation preservation changes
repeatedly, and relative-speed references vary. A static complete controller
profile is not established by the successful packet run.

The next probe isolates only pose feedback gains. `--packet-gain-profile fixed-9-8`
keeps the already-used approach gains throughout the stream. Exactly frames
173–208 (36 frames) change; every pose, motion/acceleration reference, other
setting, callback mode, jaw command and terminal-hold field remains identical.
These gains influence error correction; native effort, velocity, acceleration,
jerk and collision constraints are unchanged. The choice is a development
comparison, not an optimum or a new manufacturer limit.

Pass a physically successful recorded-gain packet run as `--packet-baseline-run`.
The runner verifies source/scene/controller fingerprints, follows its pinned
upper-feedback reference and rejects an ineffective or non-isolated change.
Runtime and analysis reconstruct the same transformed packets. The audit checks
actual applied gains and truthful configuration metadata. A separate
`packet_fixed_gains_probe_passed` requires isolated input changes, all completed
intervals, initial identity, safety, cadence and independent grasp evidence.
The original collector FSM is still bypassed; its old failed label is unchanged.

```bash
python scripts/run_hgwind_effort_probe.py \
  --tracking native-expert \
  --packet-baseline-run /path/to/passing-recorded-packet-run \
  --packet-gain-profile fixed-9-8 --run
```

The default `recorded` mode remains available. Seven new CPU regressions cover
exact preservation, invalid profiles, ineffective changes, altered motion/safety/
jaw requests and truthful runtime gain auditing. All 177 portable checks, actual
scene preflight and simulator Python 3.10 parsing pass. The prior physical packet
audit also still passes on the saved raw trace. No new GPU workload was launched;
physical fixed-gain performance is pending. Even a pass removes only this gain
schedule, not the remaining orientation/posture/motion dependencies. See
[preparation evidence](../../results/summaries/2026-10-29-fixed-gain-probe-preparation.json).


## Fixed-feedback-gain physical result — 2026-10-29

The user-run `fixed-9-8` probe passes both the explicit-packet and fixed-gain
physical gates: all 209 intervals / 627 physics and OSC updates, initial-state
identity, runtime safety, explicit jaw context and causal robot-motion audit.
Thirty-seven read-only provenance/execution checks and raw contact recomputation
match the saved report. The applied gains are 9/8 at every frame. Only the two
pose-gain fields differ from the recorded packet requests on frames 173–208;
all other input fields remain exact. All 15 core arrays match the recorded-gain
baseline for the first 173 frames. First changed action is 173 and measured
robot/object position is 174, after the gain schedules begin to differ.

| Engineering measurement | Recorded gains | Fixed 9/8 gains |
| --- | --- | --- |
| Complete control intervals | 209 | 209 |
| Continuous bilateral contact | 1.20 s | 1.20 s |
| Minimum final retained lift | 74.861 mm | 72.577 mm |
| Final relative-position jitter | 0.131 mm | 0.160 mm |
| Independent contact/lift/retention result | pass | pass |

This removes the need for the recorded pose-gain switch on this one exposed
case. It does not establish an improvement: retained lift is slightly lower and
position jitter slightly higher. Nor does it establish a frozen complete
controller or learned-policy interface. Posture/orientation settings, callback
modes and motion references remain recorded dependencies. Keep both passing
baselines. Original FSM failed labels, report files, limits and source data are
unchanged. No runtime code changed for this result update; the prior 177 CPU
checks apply. See [source-hashed evidence](../../results/summaries/2026-10-29-fixed-gain-probe-result.json).


## Measured-posture diagnostic prepared — 2026-10-29

The original upper requests supply a seven-joint `posture_target` for frames
0–172, then use `None`. The frozen `operational_space_pinch_action` already
handles `None` by using a copy of current measured joint positions for redundancy
resolution. Its minimum-norm and joint-limit terms remain. Native collision
projection can also select current measured posture. This avoids supplying an
extra expert joint endpoint to a future Cartesian policy interface.

The next one-factor probe keeps the passing fixed-9/8 gains and changes only
`posture_target` to `None` throughout the stream. Exactly 173 requests change;
all targets, velocity/acceleration references, other settings, callbacks, jaws,
terminal hold and native limits remain identical to the passing fixed-gain
baseline. The default recorded posture mode and earlier baselines remain usable.

`--packet-posture-profile measured` requires a physically passing fixed-gain
packet baseline, with source/scene/controller fingerprints and full input-change
isolation checked. Every call logs the current measured arm q, native resolved
posture and native posture source before any physics step. Execution and audit
require exact current-state correspondence; stale, nonfinite or saved posture
resolutions are rejected. The separate `packet_measured_posture_probe_passed`
combines that audit with full episode execution, safety and independent grasp
checks. The older one-factor fixed-gain result field is `null` for this distinct
posture experiment; the fixed gain profile is still checked by the packet audit.

```bash
python scripts/run_hgwind_effort_probe.py \
  --tracking native-expert \
  --packet-baseline-run /path/to/passing-fixed-gain-packet-run \
  --packet-gain-profile fixed-9-8 \
  --packet-posture-profile measured --run
```

Nine new regressions bring the portable suite to 186 passes. Actual scene
preflight, previous fixed-gain command preflight, saved baseline re-audit and
simulator Python 3.10 parsing pass. The exact frozen native posture-selection
branch also passes a 209-recorded-state CPU fixture. This checks native branch
semantics, not physical grasp performance. No new GPU run was launched; the
posture probe's physical result is pending. Other orientation/settings and
motion-reference dependencies remain. See [preparation evidence](../../results/summaries/2026-10-29-measured-posture-probe-preparation.json).
