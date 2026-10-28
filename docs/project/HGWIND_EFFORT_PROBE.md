# Saved-command effort-control comparison

## Purpose and current status

The wind-driven-ball recordings use `operational_space_inverse_dynamics_effort`. The earlier egg-task position-drive probe does not establish compatibility with them. This diagnostic first executes saved joint commands through the frozen RoboPhys effort pipeline, then permits a Cartesian/IK comparison using the same executor.

The first saved-target physical probe aborted during its 109th control interval. Recorded-motion joint tracking then completed 208 intervals safely, briefly held and lifted the ball, but lost it before the end. A task-space tracking comparison is CPU-checked and awaits simulation. No ball-task training or learned-policy result is claimed.

## Execution and recording boundary

- `scripts/run_hgwind_effort_probe.py` verifies source hashes, task physics, SDK versions, action schema and the recorded mode. It creates an independent run only with `--run`.
- `scripts/sim/run_hgwind_effort_probe_worker.py` loads the frozen collector with exact, in-memory insertion points. Original RoboPhys files and the source episode remain unchanged.
- `src/capstone_vla/hgwind_effort_probe.py` supplies the frozen collector's saved-action reference hook. Joint mode sends source joint goals. Cartesian mode computes the metadata-defined tool pose from each source goal and solves IK from measured robot joints. Object state is not an IK input.
- Both branches use the existing physics-rate reference generator and effort executor, with recorded manufacturer limits, 60 Hz command updates, 180 Hz physics and physical finger contact. The controller config, URDF, frozen worker and adapter hashes are pinned.
- The background task FSM still runs for bookkeeping. Its joint commands, operational references and policy velocity-limit suggestions are discarded. Its task outcome is not the independent diagnostic result.
- The recordings lack the complete native operational-space references. These branches therefore compare a shared joint-goal path through the original effort executor; they do not reconstruct every original controller input. Shared measured-state velocity fallback is explicit.

## First experiment

The default is `local/traj_00000`, 209 frames. Its recorded base seed is 202610280 and attempt index is 43. The new simulation requests that seed, attempt index, variation and accepted camera-v2 scene. Initialization must still be measured and compared: matching a seed alone does not prove identical state.

On the prepared workstation, CPU-only preflight:

```bash
data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_effort_probe.py --mode joint
```

Execute the full physical baseline:

```bash
data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_effort_probe.py --mode joint --run
```

A separate `--frames 8` run checks startup only and cannot establish grasp success. Both commands require the existing local assets/manifests and Isaac Sim 4.5 / Isaac Lab 2.2.1; the public source release does not bundle those dependencies.

Each execution writes a fresh directory under `data/derived/hgwind_effort_probes/`, containing the plan, simulation log, command diagnostics, newly simulated episode and independent report. It does not overwrite a source episode or invoke training. Headless execution supports a remote terminal.

## Acceptance gate

The independent report requires full source length, three physics callbacks per command, unchanged manufacturer limit profile, a passing runtime safety record and initial joint/object state agreement. The explicit initial-state tolerance is 1e-5 in each recorded quantity's native units; it is a comparison gate, not a claim about hardware precision.

Grasp evidence is recomputed from the new simulation: positive left/right impulses, persistent final contact, retention in the moving tool frame, at least 5 cm lift and object motion before contact. Hold duration, lift, retention distance and motion thresholds come from this source task's recorded success parameters. Physical contact is aggregated into 60 Hz bins: both fingers having impulses in a bin does not imply simultaneous contact at every 180 Hz substep. Per-frame checks reject robot-object attachment assistance.

These are independent engineering criteria, not the complete official benchmark verifier. Saved source success is never reused as the new result. If the joint baseline fails, inspect initialization, control tracking and physical contact before attributing failure to IK or the VLA.

A passing matching baseline enables the Cartesian branch:

```bash
data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_effort_probe.py --mode cartesian \
  --baseline-run data/derived/hgwind_effort_probes/BASELINE_RUN --run
```

The runner also prints the exact command after a passing baseline. It rejects a baseline from a different source, scene or execution fingerprint. Neither branch evaluates a learned policy, real-time inference, physical generalization or cross-embodiment transfer.

## First physical result and revised diagnostic

The saved-target joint probe recorded 109 states and completed 108 control intervals. It then stopped with `hardware constraints infeasible during velocity reserve recovery`. Source and probe first joint/object positions, velocities and orientations matched exactly. The source closes at frame 118; the probe issued no closed-jaw commands. It had already diverged from the source at frame 1, with 57.465 mm EE position difference at frame 90. The probe encountered finger/ball contact around 1.806 s; the source's first robot/ball contact was around 2.267 s. Contact preceded the abort, but this temporal association alone does not isolate the infeasibility's complete cause.

The collector overwrites saved arm `actions` with each interval's last drive-position reference. The source ran 582 operational-space controller callbacks; the probe ran zero after deliberately discarding the unrecorded expert reference, selecting joint inverse-dynamics fallback. A shared limit/effort pipeline therefore does not imply an identical primary tracking law. Reinserting the saved processed targets with a position-error-based velocity guess is not faithful original command replay. This result does not test IK or DynamicVLA. The source data's success label is unchanged.

`--reference recorded-motion` is a separate, offline desired-motion tracking diagnostic:

```bash
data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_effort_probe.py --mode joint \
  --reference recorded-motion --run
```

For each source row k it requests the observed arm joint position and velocity at k+1, keeping the original binary gripper request at k. There are 208 transitions from 209 observed states; the missing final state is not invented. Neither simulator state nor object motion is overwritten. The existing viability generator, physical gripper, effort feasibility checks and limits remain active. This uses recorded desired robot motion as diagnostic input, not a learned policy and not recovered native controller references. It changes both reference-position semantics and velocity supply; it cannot isolate their individual causal effects. A pass would support motion executability under this diagnostic, not full action-interface equivalence.

The mode currently supports joint tracking only and cannot unlock the saved-target Cartesian gate. Its result is reported separately as `motion_tracking_passed`; a future Cartesian reference/velocity contract still requires design and validation. Aborted runs with saved episodes now receive independent reports while preserving the nonzero worker exit code. The analyzer uses the collector's actual output layout and distinguishes recorded states from fully completed control intervals.

## Recorded-motion joint result and task-space comparison

The recorded-motion joint run completed 208 intervals / 624 physics callbacks with matching initial state and passing runtime safety. Both fingers had impulses in 39 consecutive control bins (frames 137–175, approximately 0.65 s), followed by lost contact. The ball rose transiently and fell; final retention and sustained lift checks failed. Whole-episode EE discrepancy from the source was 10.838 mm mean / 22.933 mm p95. Over frames 137–175, position/orientation discrepancy p95 was 22.717 mm / 7.599 degrees. These observations do not isolate friction or prove tracking error is the sole cause.

A new `--tracking task-space` branch keeps the same recorded-motion joint goals, velocity, gripper schedule, existing physics-rate viability generator and constraints. At each 180 Hz callback it maps the bounded reference into tool pose, twist and acceleration, then calls the existing RoboPhys `operational_space_inverse_dynamics_command` instead of joint inverse dynamics. It does not consume the shadow expert's Cartesian commands or object state.

The reference uses world-frame position error and shortest rotation error, `J(q_ref) qdot_ref` twist, and `J(q_ref) qddot_ref + Jdot(q_ref) qdot_ref` feed-forward. The native controller's optional measured `Jdot(q) qdot` argument is also supplied. Directional Jacobian derivatives use a centered 1e-5 s difference. Fresh physics-rate twist is not extrapolated a second time using elapsed control-interval time. The existing controller gain, inverse-dynamics solver, disturbance compensation and final hardware/geometry feasibility checks are retained. This is a constructed tracking reference, not the missing original expert reference.

Before its first command, the adapter compares its world-frame tool pose and Jacobian with live RoboPhys control kinematics (1e-4 m/rad and Jacobian absolute tolerances). A mismatch stops the diagnostic. The report verifies native OSC and generated-reference callback counts.

```bash
data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_effort_probe.py --mode joint \
  --reference recorded-motion --tracking task-space --run
```

Here `--mode joint` describes how the diagnostic goals are supplied; `--tracking task-space` selects their physical tracking controller. The mode does not perform IK or evaluate a learned policy. CPU checks cover transforms, rotation error, derivative terms, branch isolation and live-frame rejection. A call into the frozen native OSC implementation also passed at five source poses with an artificial identity-inertia model; this checks the software interface, not real robot dynamics. Physical success of the new branch remains unverified.
