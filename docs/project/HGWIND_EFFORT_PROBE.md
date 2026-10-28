# Saved-command effort-control comparison

## Purpose and current status

The wind-driven-ball recordings use `operational_space_inverse_dynamics_effort`. The earlier egg-task position-drive probe does not establish compatibility with them. This diagnostic first executes saved joint commands through the frozen RoboPhys effort pipeline, then permits a Cartesian/IK comparison using the same executor.

The implementation and CPU preflight are checked. No physical rollout has yet been run with this adapter. No ball-task training or learned-policy result is claimed.

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
