# Saved-command effort-control comparison

## Purpose and current status

The wind-driven-ball recordings use `operational_space_inverse_dynamics_effort`. The earlier egg-task position-drive probe does not establish compatibility with them. This diagnostic first executes saved joint commands through the frozen RoboPhys effort pipeline, then permits a Cartesian/IK comparison using the same executor.

The first saved-target physical probe aborted during its 109th control interval. Recorded-motion joint tracking then completed 208 intervals safely, briefly held and lifted the ball, but lost it before the end. The constructed task-space comparison completed but performed worse. CPU endpoint-semantics checks identify an incompatible interpretation of its inputs; the native outer-pose-servo adapter subsequently aborted after contact. The original expert and complete-control replay now both reproduce the source episode exactly. The live policy action contract remains pending. No ball-task training or learned-policy result is claimed.

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

## Constructed task reference rejected; use the native command tuple

The task-space run verified 624 native OSC calls, completed safely, and had matching initial conditions, but formed no bilateral grip. Position/orientation discrepancy p95 increased from the joint run's 22.933 mm / 7.094 degrees to 42.259 mm / 22.408 degrees. This is a rejected diagnostic construction, not a policy or benchmark failure.

Inspection of `sim_common.motion_safety.DynamicDriveReferenceTracker.advance()` establishes that its position output is a control endpoint, while its velocity is independently filtered. The existing expert normally converts pose correction to a bounded twist and sets the inner OSC position error to zero. The earlier adapter instead interpreted endpoint position and filtered velocity/acceleration as derivatives of a differentiable trajectory.

`scripts/diagnose_hgwind_reference_clock.py` checks this boundary on CPU using the frozen generator and an explicitly assumed ideal measured motion. Over 624 updates, 412 had held endpoints with nonzero velocity. The endpoint finite-difference versus supplied-velocity residual had RMS 1.319 rad/s. This counterfactual is not a recorded physical-substep trace. Direct cubic-Hermite interpolation of measured q/qdot is not a ready replacement: among 625 sampled points, acceleration exceeded the reference limit at 282 and within-interval jerk at 80; acceleration can also jump at knots. These interpolation properties do not invalidate the source's measured physical motion or its original commanded-limit certificate.

The proposed `--tracking native-pose` branch reuses `Expert.operational_space_pinch_action()` to generate its matched q endpoint, qdot and OSC reference together. It maps the source robot waypoint into the world tool/pinch frame, including the offset velocity `omega × offset`, then invokes the existing pose servo. Its own pose correction remains in the outer servo, so the returned inner OSC pose error must be zero. Missing/nonfinite paired outputs stop the probe; stale shadow references are cleared before the call. Gripper commands at k remain unchanged.

The native API's default pose gains and speed settings are retained, with `preserve_orientation_during_translation=True`, `coupled_operational_pose_task=True`, `use_operational_space_inverse_dynamics=True`, source next-q posture and dt=1/60. The same physics-rate viability, inverse-dynamics and final effort/geometry checks remain in place. The report requires 208 native pose commands and 624 OSC callbacks for the full diagnostic. This changes the outer command construction as well as the inner-reference convention; it is not a pure controller-gain comparison or a recovered original expert.

```bash
data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_effort_probe.py --mode joint \
  --reference recorded-motion --tracking native-pose --run
```

The new branch passes CPU adapter tests and real-data preflight; its physical outcome remains pending. The earlier `task-space` branch is retained for provenance and is not the recommended continuation. Original source episodes and upstream code are unchanged.

Native-pose runs additionally save each returned OSC-reference dictionary alongside waypoint/twist inputs in `native_pose_references.json`, and the actual requested q/qdot in `command_diagnostics.npz`. This avoids repeating the earlier omission of the paired reference signals. These local diagnostic records are not published or declared training data.

## Native-pose contact abort and original-expert reference

The native-pose run completed 131 intervals and recorded the start of interval 132. It had five positive right-finger/ball contact reports from approximately 2.1667 to 2.1889 s and no bilateral grasp. Measured joint 5 velocity then reached 2.674532 rad/s against a 2.61 rad/s limit. Returning below the limit in one 1/180 s step requires acceleration at most -11.61579 rad/s², while the current acceleration/jerk constraints permit no value below -7.07415 rad/s². The intervals do not intersect. Even the strongest permitted braking yields about 2.63523 rad/s next step. This establishes why the final command was rejected, not a complete causal diagnosis of the approach/contact error. Constraints remain unchanged.

The original report incorrectly marked the native branch unverified because it compared 394 OSC calls with 393 completed callbacks. The native code increments the OSC counter before final effort feasibility. The last calculation was rejected before physics completion. The analyzer now permits exactly one such extra attempt only for an aborted run and reports completed-interval cadence separately; full episode/grasp gates still fail. The previous report is retained locally.

The next reference restores original online expert command generation instead of feeding any saved or converted command array. `--tracking native-expert` reuses the frozen collection worker and frozen `smooth_lift_v1` profile, plus the existing separately deployed depth helper at the source's 0.002 m setting. The profile wrapper, depth helper, worker and local trace adapter are hashed. Base seed 202610280, attempt index 43, variation, physics scene and physical grasp mode are retained. This asks for one fresh attempt, not a guaranteed reproduction of the source episode; initial states, the newly observed result and trajectory differences are checked afterward.

```bash
data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_effort_probe.py --tracking native-expert --run
```

The time budget is the original 12 simulated seconds (720 control steps maximum), with normal early completion after verified success. The old source episode's 208-transition replay budget does not truncate this online attempt. No new target, speed, friction, contact-recovery or limit parameter is introduced. Native target q/qdot, pending OSC reference and policy limits are copied at each actual command segment into `original_expert_control_references.json`; tracing does not modify them. Physical assistance is separately checked.

`scripts/sim/run_hgwind_native_expert_worker.py` assembles the original wrappers in memory. `src/capstone_vla/hgwind_expert_trace.py` only copies logs and adds metadata. The report separates `native_expert_reference_passed` from all converted-action and learned-policy gates. A successful original expert reference would support the original execution setup and provide the missing command log; it would not validate the conversion or DynamicVLA. A failed reference requires investigating original-expert/environment reproducibility before further adapter tuning. Original code and the 40 development episodes remain unchanged.

## Original expert reproduces the recorded episode exactly

The fresh native-expert reference succeeded after 209 control intervals / 627 physics callbacks, including 582 OSC calls. All 15 core datasets (actions, robot/object observations, timestamps and episode flags) are exactly equal elementwise to the source episode. This is numerical array equality; the HDF5 files contain different run metadata. Independent evidence passes: 72 bilateral control bins (1.2 s), final-window minimum lift 74.861 mm and tool-frame jitter 0.123 mm. This verifies one exposed seeded expert case, not VLA performance or generalization.

The 720-step setting was a maximum budget; success triggered normal early completion at 209. The analyzer now marks source replay length and irrelevant motion/Cartesian gates as null for online-expert runs, and reports `expert_early_success=true`. It must not imply those unrelated tests failed.

The captured raw arm goals differ from exported `actions` by up to 0.046696 rad. The existing export stores processed drive references, so identical field names do not make the two interfaces interchangeable. The successful run now supplies the previously missing original q goal, paired qdot, OSC reference and policy speed-limit tuple.

A separate `logged-command` diagnostic replays that tuple through the same frozen smooth-lift/depth configuration. It requires a passing native-expert report with matching source/scene/worker, exact logged measured q/qdot agreement, sequential frame/timing checks, finite matching task vectors, and pinned control-log/report hashes. Frames without an OSC reference retain the original joint-controller fallback. Policy speed limits are replayed together with the reference. Full-run acceptance additionally checks the expected OSC callback count.

```bash
data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_effort_probe.py --tracking logged-command \
  --reference-run data/derived/hgwind_effort_probes/20261028_191242_476137_joint_online-expert_native-expert --run
```

The subsequent 209-step physical replay passed, as detailed below. It intentionally uses cached original Jacobians/constraints and is limited to diagnostic reproduction; it is not a VLA observation/action contract. Its independent result is `control_reference_replay_passed`. A deployable policy adapter must generate its references from current robot state and the policy's documented outputs, after this integrity check.

## Complete-control reference replay passes

The logged-command physical replay completed 209 intervals with 627 physics updates, 582 OSC calls, matching initial state and passing runtime safety. Independent grasp evidence passes: 72 bilateral control bins (1.2 s), at least 74.861 mm retained lift in the final window and 0.123 mm hand-frame jitter. All 15 core datasets again equal the source exactly elementwise. Both the online original expert and complete-control replay reproduce this one source trajectory. This verifies execution/data integrity for one exposed case; no VLA ran.

Robot/object states were not overwritten. The replay nevertheless used original cached Jacobians/constraints and retained the shadow task FSM for bookkeeping/safety. It does not certify a standalone live Cartesian policy adapter. Earlier failed variants differed in multiple respects, so this result does not isolate one missing signal as the sole cause of all earlier failures.

The next gate is a documented policy action contract: define pose versus twist, frame, units, gripper semantics and timing; construct matching labels; derive runtime kinematics from current robot state; and validate an expert-derived Cartesian command stream through that live executor. Cached control matrices and source FSM timing must not become hidden policy inputs. Existing episodes/cameras are preserved; ball-task training and split assignment remain pending. Check a small set of additional episodes after the single-episode action gate before batch conversion.


## Live robot kinematics component check prepared

The next controlled change replaces only the cached task Jacobian at each 60 Hz control boundary. `live_kinematics.live_pinch_jacobian()` obtains the current robot tool pose and world-frame Jacobian from the existing RoboPhys robot API, then shifts its linear rows to the calibrated pinch center using `omega cross offset`. Angular rows are unchanged. It takes no trajectory or object-state input. The result is held for the original three 180 Hz physics ticks; this does not introduce a new servo cadence. The old Jacobian is retained only for comparison in the audit, never for command selection.

Other command components remain frozen. The successful source contains 194 six-dimensional OSC references followed by 15 joint-only fallback frames (194–208); 110 rows have nonzero task-acceleration feedforward, and 141 carry joint-velocity geometry constraints. Removing these together with the cached Jacobian would confound the diagnosis. Original q/qdot goals, twist, feedforward, constraints, velocity caps and gripper signals therefore remain in this component experiment, as does the shadow FSM for bookkeeping/safety. It is **not a standalone Cartesian policy controller**, even if the physical grasp succeeds.

```bash
data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_effort_probe.py --tracking live-jacobian \
  --reference-run data/derived/hgwind_effort_probes/20261028_191242_476137_joint_online-expert_native-expert \
  --run
```

The first physical component run aborted; the metric correction below is prepared and its rerun is pending. Acceptance requires all 209 intervals, matching initial state, original safety limits, 194 live kinematics updates / 582 OSC calls, and independent physical grasp evidence. The new result is `live_kinematics_component_passed`; no irrelevant motion/Cartesian gate is interpreted as a failed test. A numerical comparison with the successful logged-command run can identify divergence, while exact array equality is not assumed for a different Jacobian computation.

At preparation, 113 portable CPU checks passed, including nine new checks for point-offset sign, input validation, no mutation, independence from the stored Jacobian and propagation to the physics boundary with unchanged velocity caps. Python 3.10 parsing and the 209-frame scene/control-log preflight also pass. Existing data and upstream code are unchanged. Rollback: use `--tracking logged-command` for the existing complete-control reference; the new branch is opt-in.

### Policy-label decision remains open

DynamicVLA's local dataset adapter can disable relative-pose conversion, but this does not validate a twist-action policy. The existing egg pipeline subtracts current pose, wraps Euler differences and adds predicted deltas back to pose. Those operations must not be reused for velocity labels. A proposed six-dimensional twist plus one gripper command would need matching frame/units/control point, normalization, startup behavior and a robot-only execution rule, with pose-delta conversion disabled in every relevant layer. The original acceleration feedforward and geometry constraints also need an explicit runtime treatment. No ball-task training or bulk label conversion has started.


## Coupled-task metric error identified and corrected

The first live-Jacobian run saved 151 states and completed 150 intervals before an actual velocity/acceleration/jerk feasibility abort. Joint 5 reached about -2.6872 rad/s versus its absolute 2.61 rad/s limit. Bilateral contact lasted about 0.267 s, but sustained lift failed. The initial state matched; robot motion already diverged at the first control interval. These outcomes do not establish that live kinematics itself is unsuitable.

The added adapter omitted an existing task-metric transform. The frozen Catch call passes `operational_orientation_length_scale_m=max(r_A,1e-6)`; here the recorded configuration has `r_A=0.03 m`. The native coupled reference builder therefore uses `[J_linear; 0.03 * J_angular]` and `[v_linear; 0.03 * omega]`. The first live branch replaced this with the unweighted physical Jacobian while preserving the weighted task velocity. It changed the interface equation from the first interval; the omission was in the new adapter. The original tests checked robot geometry but missed this operational-space convention.

The fix keeps `live_pinch_jacobian()` as the physical world-frame Jacobian, then explicitly applies the original angular length scale through `operational_task_jacobian()`. The plan records the scale and its source. A source-specific CPU guard checks every angular column norm in all 194 coupled reference rows before simulator launch. Source matrices are used for preflight/audit only; runtime Jacobians still come from the current robot. The first boundary now agrees with the original matrix to a maximum absolute difference of 7.886e-10 (previously 0.97). This verifies the interface correction at matching state, not the future grasp result.

Seven further regression cases cover the coupled equation, invalid scales, source metric mismatch and propagation to the motor-reference boundary. All 120 portable CPU checks and the real 209-frame preflight pass. The command above is unchanged; it now also prints `TASK_METRIC_OK orientation_length_scale_m=0.03 coupled_rows=194`. Corrected physical execution remains pending; safety thresholds, gains, physics and demonstrations were not changed.

Correction to the earlier narrative: the 15 references without OSC are frames 194–208 at the end of the source trajectory, not startup frames. The source log and executable None handling were already correct; the sequence description was inaccurate. The original failed report remains unchanged, and a separate [diagnosis summary](../../results/summaries/2026-10-28-task-metric-correction.json) records the failure and correction.
