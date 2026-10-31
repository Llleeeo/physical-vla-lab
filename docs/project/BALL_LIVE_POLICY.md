# Bounded live ball-policy integration

## Status

**Prepared and CPU verified; physical execution pending.** The current checkpoint is the exposed `local/traj_00018` model fine-tuned for 2,000 steps. Offline position error remains substantial, especially at startup. The purpose of the first run is to verify the complete live feedback path, not demonstrate grasp success or physical generalization.

Checks completed for this implementation:

- 389 portable CPU tests pass, including nine new live-interface regressions (75 tests in the directly affected interface group).
- Real saved data at frames 0, 1, 2, 4 and 268 produces exactly identical online/training image tensors, state tensors and instructions after upstream preprocessing. History is [-2, 0] with repeated initial observations.
- The actual fine-tuned checkpoint strictly loads on CPU with exact action/state normalizers. CUDA remains uninitialized.
- The actual frozen collector patch compiles, the scene physics signature and calibrated camera selection pass, and the generated configuration is JSON serializable. Six new runtime/entry files parse with Python 3.10 grammar.
- No GPU inference, simulation or real robot commands were launched to establish these preparation results.

## Data and control path

1. The unchanged collector initializes the declared scene/robot variation for the exposed development case. Recorded actions are not replayed.
2. At each control tick, the adapter reads the collector's measured joint positions/velocities and official end-effector transform, applies the same calibrated EE-to-pinch offset as training, and forms state30. The posture anchor is latched from initial robot state and declared robot configuration.
3. The existing `LiveRGB` helper captures the calibrated `opst_cam` and `wrist_cam` at 640×480. Rendering synchronizes live robot camera mounts and suspends BehaviorScript dispatch for zero-time render updates. Runtime checks require unchanged simulation time and robot q/qdot. The actual simulator behavior still requires the first GPU test.
4. A separate model process applies the same upstream 360×480 preprocessing and normalization and generates 20 absolute29 actions. It consumes only live RGB, measured robot state and the instruction; object ground truth and expert commands do not enter inference.
5. The executor consumes the first four actions, then requests a new chunk using updated observations. It does not add robot state to absolute actions. Output masks/scalars use the existing decoder. Inconsistent relative-speed/reference requests are rejected explicitly, without expert fallback or silent repair.
6. `PacketStreamExecutor` owns motor requests, binary jaws and target-blind non-target collision geometry. The native effort controller, limit checks and measured robot feedback remain in use. Online task/skill decision steps are bypassed. The frozen skill's initialization and geometry/drive helpers remain, as in the passing surrogate.

Simulation waits during inference. This is synchronous closed-loop simulation, **not** a real-time latency result. No friction, action gains, joint limits, scene physics or training labels have been changed. The old 7D task adapter and training files remain unchanged.

## Run the first check

From the configured project root:

```bash
data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_closed_loop.py --run --frames 8
```

Without `--run`, the entry performs CPU preflight only. Add `--check-load` to include a strict CPU model reload. The two-process simulator/model runtime uses the existing isolated environments and offline task context; it does not require Feishu or network model downloads.

Eight 60 Hz control intervals cover about 0.133 s of simulation and two four-action chunks. Most wall time will be model/simulator startup and camera initialization. This check cannot establish full startup behavior or grasp success. Inspect its report before scheduling a full 269-frame rollout.

## Saved evidence and interpretation

Each invocation creates a new `closed_loop_runs/<timestamp>` directory under the selected training run. It contains:

- `run_config.json`: checkpoint/runtime hashes, scene, reset seed, action schema settings and declared limitations.
- `live_observations.hdf5`: live pre-action RGB, state30, consumed absolute29 actions, camera transforms and timestamps.
- `ipc/request_*.npz` / `response_*.npz`: the exact model inputs and generated chunks.
- `policy_timing.json`, `packet_requests.json`, `cartesian_packet_execution.json` and `live_control_references.json`: queue origins, decoded motor inputs and native feedback references. Partial evidence is retained on Python-level failure where possible.
- `worker_output/`: the collector's raw trajectory, contact and physics records if the collector reaches its save boundary.
- `policy_server.log`, `simulation.log`, `worker_exit.json` and `report.json`.

`BALL_LIVE_INTEGRATION_PASS` requires all requested intervals, three native physics callbacks per interval, unchanged initial state/limits, aligned saved robot observations, exact IPC input and output bindings, native motor-boundary audit and no assistance. This is an engineering integration result. `BALL_LIVE_INCOMPLETE` retains failure details and must not be reported as a model success.

The native task FSM is bypassed, so its collector success field is retained separately and is not the learned-policy verdict. The independent contact/lift/retention evaluator runs on raw evidence. An eight-frame run reports only partial contact evidence; full-episode grasp assessment is reserved for a completed 269-frame run. None of these development runs supplies a held-out benchmark estimate.

## Files and rollback

New entries: `scripts/run_hgwind_closed_loop.py`, `scripts/hgwind_policy_server.py`, `scripts/sim/run_hgwind_policy_worker.py`, `scripts/analyze_hgwind_closed_loop.py`.

New adapters: `src/capstone_vla/ball_live_protocol.py`, `src/capstone_vla/ball_live_runtime.py`; regressions: `tests/test_ball_live_policy.py`.

Rollback consists of stopping use of these new entries; existing expert/surrogate/training interfaces and immutable artifacts are preserved. The coordinator stops only its own model/simulation process groups on exit.
