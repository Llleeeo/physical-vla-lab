# Bounded live ball-policy integration

## Status

**Full live execution verified; learned grasp failed.** The current checkpoint is the exposed `local/traj_00018` model fine-tuned for 2,000 steps. Offline position error remains substantial, especially at startup. The full feedback path now passes; approach/contact behavior is the next diagnostic target.

## Camera control outcome and bounded model refinement

The optional formal capture completes 8 intervals / 24 callbacks without a safety abort, with exactly matching initial robot state and camera poses. It is **not promoted to default**. Initial wrist RGB MAE against training increases from 13.92 to 24.35/255; external increases 8.36 to 10.34/255. External images at frames 4/7 improve, but wrist images do not. The measured pinch discrepancy at frame 7 is 13.49 mm legacy versus 13.78 mm formal, so there is no observed short-motion improvement. Eight frames do not evaluate grasp; later pixel comparisons also include differing robot poses. [Source-hashed result](../../results/summaries/2026-10-31-ball-camera-control-result.json).

Next, refine the existing 2,000-step model with a separate entry, at most 1,000 additional steps at learning rate 1e-5 (previously 1e-4). This loads the parent weights strictly and retains its normalizers, data and backbone freezing. The optimizer restarts because no optimizer state was saved. It does not restart model weights from the official checkpoint. Old scripts/checkpoints and runtime defaults are retained.

Every 250 steps, generate the complete 269-frame stream on recorded expert observations using the same four-action queue and 90000+frame noise. The archived batch4 parent reference has 66.69 mm mean approach target error (before expert closure at 144), six wrong gripper frames, five transitions, orientation p95 of 4.46 degrees and zero native request rejects. Select a candidate only when approach mean improves versus the current best, orientation p95/gripper error count/transitions do not worsen versus the parent, and native rejects remain zero. Selection recomputes the parent baseline with batch1, matching live inference, and verifies the available eight batch1 teacher-input references. Stored batch4 predictions differ slightly and are not used as an exact replay gate. This avoids selecting by a noisy flow loss alone. It remains same-episode development selection, **not a held-out validation metric or grasp success**.

The new run retains one selected full checkpoint under `checkpoint_final` and all candidate predictions/metrics. `selected_checkpoint_step` records when that checkpoint was selected; it can differ from the total number of optimizer steps. If no candidate passes, step 0 retains the parent and `selected_improved=false`. The original parent directory is never overwritten.

```bash
.venv_dynamicvla/bin/python scripts/refine_hgwind_development.py --run
```

CPU preflight and strict actual parent reload pass without CUDA; 13 focused selection/prediction checks pass. Training and physical improvement are pending. The user runs GPU jobs; this entry launches no Isaac Sim or robot commands.

## Input sensitivity result and optional render-protocol control

All 32 predictions completed, and the all-live 20-action chunks match saved execution responses exactly at eight anchors. Across their first four actions, image-only replacement changes predicted target position by 19.28 mm on average (max 43.31 mm); state-only replacement changes it by 6.30 mm (max 11.96 mm). These are input sensitivities, not additive causal failure contributions. At frame 0 states match exactly; replacing only images changes the first target by 18.50 mm. At later anchors the hybrid inputs can be inconsistent. Even teacher images/state retain first target errors 403.28 mm at 0,143.85 mm at 8 and95.27 mm at 40, so image consistency alone is not an established cure.

**Reporting correction:** one field ending `_mm` in the original input-probe report accidentally contained metres. The original report and prediction arrays are retained. `verified_input_metrics.json` recomputes the correct units directly from those arrays, and the diagnostic now multiplies by 1000. Existing first-target-error metrics were already correct. A 3-4-5 mm regression checks conversion. [Verified summary](../../results/summaries/2026-10-31-ball-input-sensitivity.json).

Code inspection finds an additional concrete mismatch: the formal recorder resets DLAA history around each four-subframe capture; the original live helper renders once before capture and does not reset that history. The opt-in `formal-temporal` boundary removes the extra prerender, brackets capture with the same history-reset key, and applies only the two renderer intensity clamps saved with this training scene. It does not treat probe ISO as exposure, load saved images, change model weights, use object state, alter scene physics or relax safety. It leaves the native timeline running while requiring zero simulation-time advancement. Camera images are still measured live. Exact appearance matching and improved grasp remain unverified; this combines the formal capture boundary components, not a reset-only causal ablation.

```bash
data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_closed_loop.py --run --frames 8 \
  --camera-protocol formal-temporal
```

The original `legacy` mode remains the default and rollback option. Eight frames test the new capture boundary and matched initial-frame input/output differences, not grasp. Tests: 23 focused CPU checks pass, including zero-time/reset cleanup, invalid calibration rejection and corrected units; the actual scene/model-input CPU preflight passes without CUDA. GPU runtime result pending.

## Observed full rollout and failure chronology

The user-run eight-frame integration and subsequent full 269-frame rollout both pass the live-chain audit. The full run completes 807 physics callbacks with matching initial conditions and unchanged native safety checks. **The learned grasp fails**: zero bilateral-contact frames, no retained lift. This is one exposed training case, synchronous simulation, not held-out or real-time evaluation.

Raw evidence locates the first right-finger contact at frame 104 while the fingers are still open (about 0.040 m per finger). The measured EE differs from the expert by 60.73 mm at that time. The ball remains on the expert path through that pre-action frame, then differs by more than 1 mm at frame 105. The model's 15 gripper-threshold transitions begin later, at frame 106, ending at 130. The expert closes at 144, first contacts at 162, and achieves bilateral contact at 163. Live right-finger contact occurs in only one control interval; left contact is absent throughout.

This supports prioritizing approach deviation and premature single-finger impact over a gripper-only fix. It does not identify why the learned approach differs. At the identical initial state, camera poses agree within 0.001 mm and 0.001 degrees; live/train RGB differs (mean absolute 8.36/255 external and 13.94/255 wrist). Brightness/rendering differences are a candidate input shift, not a demonstrated cause.

A new diagnostic keeps the checkpoint, noise seed, history and prefix fixed and crosses teacher/live images with teacher/live measured robot states at eight pre-contact/contact anchors. Frame 0 has identical states, isolating the image input contrast there. Later hybrid inputs can be inconsistent and are sensitivity probes, not realistic rollouts or counterfactual-success claims. Each all-live prediction must reproduce its saved response within explicit numerical tolerance before interpretation. No object ground truth enters the model; source object states are used only by the CPU evaluator.

```bash
.venv_dynamicvla/bin/python scripts/diagnose_hgwind_live_failure.py --run
```

This runs 32 model input combinations on GPU; it does not train or launch Isaac Sim. Without `--run`, it saves the CPU chronology and optionally a plot. No matplotlib installation is required in the model environment. The actual 32 input combinations, checkpoint contract, frame 0 state equality and saved request bindings have passed CPU preparation; GPU inference is pending. [Compact verified result](../../results/summaries/2026-10-31-ball-live-failure.json).

Checks completed for the original implementation:

- 389 portable CPU tests pass, including nine new live-interface regressions (75 tests in the directly affected interface group).
- Real saved data at frames 0, 1, 2, 4 and 268 produces exactly identical online/training image tensors, state tensors and instructions after upstream preprocessing. History is [-2, 0] with repeated initial observations.
- The actual fine-tuned checkpoint strictly loads on CPU with exact action/state normalizers. CUDA remains uninitialized.
- The actual frozen collector patch compiles, the scene physics signature and calibrated camera selection pass, and the generated configuration is JSON serializable. Six new runtime/entry files parse with Python 3.10 grammar.
- No GPU inference, simulation or real robot commands were launched to establish these preparation results.

## Data and control path

1. The unchanged collector initializes the declared scene/robot variation for the exposed development case. Recorded actions are not replayed.
2. At each control tick, the adapter reads the collector's measured joint positions/velocities and official end-effector transform, applies the same calibrated EE-to-pinch offset as training, and forms state30. The posture anchor is latched from initial robot state and declared robot configuration.
3. The existing `LiveRGB` helper captures the calibrated `opst_cam` and `wrist_cam` at 640×480. Rendering synchronizes live robot camera mounts and suspends BehaviorScript dispatch for zero-time render updates. Runtime checks require unchanged simulation time and robot q/qdot. Frozen-time checks passed in both user-run rollouts.
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


## Selected refinement checkpoint: matched full rollout

The additional-step1,000 checkpoint passed strict CPU reload and live-input preprocessing after an explicitly recorded report metadata completion. Recorded-input improvements do not establish grasp success; see [results](RESULTS.md#bounded-refinement-result--31-october-2026). Run the same initial condition and controller with the new weights:

```bash
data/derived/hf_pilot_render_repo/.venv/bin/python scripts/run_hgwind_closed_loop.py \
  --run-dir data/derived/hgwind_traj18_development_20261031/training_runs/20261031_133555_985430 \
  --frames 269 --prefix 4 --camera-protocol legacy --run
```

Outputs go to a new `closed_loop_runs` subdirectory of this training run. GPU simulation/inference uses synchronous pause; this does not measure real-time performance.
