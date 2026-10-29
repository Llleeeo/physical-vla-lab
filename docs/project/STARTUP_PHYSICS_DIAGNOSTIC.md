# Startup physics mismatch: bounded passive diagnostic

## Startup force measurement: reproduced failure, observed setup calls

User run `20261029_225526_640181` reaches simulation under GDB, records169frames/168completed intervals, and renders five30Hz views of85frames each. The15 compared action/observation/native arrays match the prior uninstrumented fresh teacher1 failure exactly. This establishes observed noninterference for those arrays in this run, not a general claim about logging/debugger timing.

The joint-offset4 speed is2.6393rad/s versus the2.61 bound; required braking conflicts with acceleration/jerk bounds. Bilateral contact lasts0.4667s, but sustained lift fails. The diagnostic completing means that a measurement exists; grasp still fails. No learned policy ran.

Ten force API calls are observed: four before control-frame recording at the same0.033333334s clock reading, followed by six marked control frames. Setup force x-components are0.670589983,0.670589983,1.668164968,1.668164968N. All calls return normally and report configured dt1/60s. This configured value is not the native integration-step record. No zero-dt call was observed in coverage, so the earlier zero-dt hypothesis remains unsupported. Equal clock readings do not prove force accumulation or absence of internal stepping; original force-call histories are unavailable.

The two earlier C++ scene-load crashes do not recur in this launch. No native crash stack was saved and no crash fix is established. Proposed next step: compare original worker history58/59/60 with the already recorded fresh attempt60, preserving settings and every outcome. The [bounded history runner](WORKER_HISTORY_DIAGNOSTIC.md) is now implemented and CPU validated; GPU execution remains pending. Even a history effect would not isolate a unique fan/cache/solver cause. [Source hashes and measured values](../../results/summaries/2026-10-29-startup-force-measurement.json).

## Verified evidence

The completed three-case batch is documented in [results](RESULTS.md). The two aborted native teachers do not first diverge at grasp: the ball's native velocity differs immediately after the first 180 Hz physics step (about 5.56 ms). Their first motor actions are identical to the originals, and EE positions still match at the next 60 Hz observation. Initial native ball mass, inertia, COM and transform match; control-frame initial position, orientation and velocities also match.

| Source case | Ordinal within original worker | First physical-step linear velocity difference, fresh minus source (m/s) |
|---|---:|---|
| traj_00001 | 3 | (0.004831869, approximately 0, -0.000151388) |
| traj_00007 | 2 | (0.004739776, approximately 0, 0) |
| traj_00018 | 1 | (0, 0, 0) |

For case1 the downward-velocity difference grows to -0.024374 m/s by the third physics step. The initial three physical contact records report zero net ball impulse in both versions. That is an observation about recorded contact evidence, not proof that every internal contact/solver state matches. Case18 also matches every core array through the native rollout; its 29D surrogate passes independently.

`diagnose_hgwind_startup_history.py` resolves the ball by native rigid-body path, compares per-physics-step values and signed raw contact impulses, and identifies process boundaries from original collection logs. It writes `startup_physics_diagnosis.json` next to the batch. Sources are checksum verified, not edited.

## Remaining hypotheses

The originals for cases1/7 came from the third/second attempt in an existing worker; the new probes run fresh workers. Source case18 and its regeneration both begin with fresh workers. This correlation suggests reset/history deserves attention but is not a controlled causal experiment.

The frozen fan's `on_update` calls `apply_force_at_pos`; collector startup Play temporarily configures physics_dt=0 while pumping Kit events. Native forces may therefore depend on setup dispatch/cache history not represented by first-frame positions and velocities. The original data do not record those force API calls. We have not established that zero-dt force calls actually occurred, survived reset or caused the later safety abort. Gravity/damping/solver state is not fully ruled out by the initial native-property subset.

## Repeated native abort: capture a stack before further changes

Both runs `20261029_223230_505509` and `20261029_224044_493811` abort at the same scene-load point after `open_stage` returns, before completed script inspection. Both force traces are empty and neither saves an episode. The new wrapper/initialization path remains a candidate; no evidence yet identifies a specific root cause. Do not repeat the unchanged command indefinitely.

Use the existing system GDB for one instrumented launch:

```bash
cd /home/robophys/Capstone_Project_Yuan_Haoming

data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_startup_diagnostic.py --native-backtrace --run
```

This launches only its own SDK Python child under GDB. It does not attach to another process, update the SDK, replace the entry or change simulator/control parameters. The system debugger starts without SDK LD_PRELOAD/LD_LIBRARY_PATH/PYTHONPATH/PYTHONHOME; its child receives the exact SDK values via the inferior environment. All other established worker environment/arguments remain in place. The plan pins the debugger helper hash and generated wrapper/command hashes. No debugger download or installation is performed.

A native stop writes `<run>/native_backtrace.log` (40 current-thread frames and four frames per thread), also copied into `simulation.log`; the coordinator includes the path in `diagnostic.json`. If the child exits normally and has saved a trajectory, ordinary analysis/rendering proceeds. No automatic retry loop is added. GDB can change timing or stop on a different signal; examine the actual captured stop before connecting it to the original C++ abort. The user subsequently ran this under Isaac; it reached the same saved physical failure without a native stop, as documented above.

243 portable CPU checks pass. Separate CPU-only integration runs verify exact success/nonzero child statuses (0/7), preserved child library environment and saved stack on a controlled abort (255). Actual scene preflight passes, without GPU launch. Two further CPU-only tests through the actual SDK python.sh/Python3.10 chain preserve a normal exit and capture a deliberately triggered abort (python.sh reports1); SDK library/Python paths reach the child. No SimulationApp or GPU workload is started. These tests verify the diagnostic tool, not recovery of the simulator or physical task.

## First attempt did not produce a measurement

Run `20261029_223230_505509` aborted natively during scene loading (`std::system_error: Invalid argument`) before completed script inspection and recorder installation. There are no saved episodes, force-call samples or videos. Empty trace counts must not be interpreted as absent wind. The root cause is unknown. Preserve this failure, retry the same command at most once manually, and investigate native startup if it repeats; no SDK upgrade or random parameter adjustment is warranted by this log.

The coordinator now labels missing episodes `STARTUP_DIAGNOSTIC_FAILED status=no_evaluable_episode`. A recorded force measurement is distinct from physical grasp success, and saved physical failures are still rendered. Native fatal lines are printed in the launching terminal. This status fix passes240CPU checks; the retry has not been launched by the agent.

## Original measurement command (completed)

```bash
cd /home/robophys/Capstone_Project_Yuan_Haoming

data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_startup_diagnostic.py --run
```

Default case: traj_00001, one native expert attempt with the unchanged 12 s maximum simulated budget. No parameter sweep, training or repeat-until-success behavior. The user-run measurement is complete as documented above; default without `--run` remains CPU preflight. The command below is retained for provenance, not a request for another identical retry.

The optional `--trace-startup-forces` flag on `run_hgwind_effort_probe.py` selects a separate worker wrapper. It installs a fan-module-local proxy around the native PhysX getter. Every call is forwarded with unchanged arguments/return/exception; no force clearing, skipped application, additional native script update or robot command is introduced. Capture records body, force, application point, mode, control-frame marker, simulation clock and configured physics_dt.

The observer starts after initial script inspection and refreshes before reset/Play. It cannot see earliest loading updates or forces already in an accumulator. Recording retains setup calls and the first six control frames, bounded at 2048 calls. Logging can alter wall timing; physical noninterference is not established by CPU forwarding tests. Native initial-state/trajectory reports still decide what actually reproduced.

All force records stream immediately to `<run>/startup_force_calls.jsonl`. The coordinator retains a nonzero physics exit and renders any saved episode, successful or unsuccessful, through the existing saved-state renderer. Five 30 Hz review videos are checked; paths and trace counts appear in `data/derived/hgwind_startup_diagnostics/<timestamp>/diagnostic.json`. No episode can be rendered if execution stopped before saving states. These remain review videos, not formal 60 Hz training pixels.

**Interpretation gate:** observed nonzero force calls at physics_dt=0 would establish an unintended-startup-force candidate, not prove causality alone. Missing calls do not rule out effects before observer installation. Review the actual native state trace and startup call timing before changing physics; preserve the current passing case and original failures. Do not loosen velocity/jerk constraints to hide the aborts.

## Verification and rollback

243 portable CPU checks pass, including exact native object/argument identity and return forwarding even at zero dt, preserved native exception, limits that never suppress force calls, getter restoration, native body identity/contact impulse sign and original-worker ordinal parsing. Actual scene preflight and preflight/runtime patch assembly equality pass. Simulator modules parse as Python3.10. No GPU job was launched for this implementation.

Remove the new modules/entry points and the optional flag, or revert the milestone commit, to roll back. The frozen worker, asset scripts, existing probe wrapper, control parameters, source trajectories and prior reports are unchanged. Historical diagnostics and video files remain available.
