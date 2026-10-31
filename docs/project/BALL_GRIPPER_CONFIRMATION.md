# Causal gripper confirmation experiment

Status: CPU-verified optional implementation; live efficacy pending. Development case only.

The retained 3,000-step model and latest prefix1 baseline remain fixed. At each 60 Hz action boundary, start open; require three consecutive raw jaw outputs >=0.8 to close; once closed, reopen only below0.2. Reset the consecutive counter whenever an open-state score is below0.8. These are regression scores, not probabilities; thresholds are development heuristics selected after inspecting this case.

The optional `--gripper-confirmation` switch is restricted to269 frames, prefix1 and legacy camera capture. It cannot be combined with the recorded-height diagnostic. No expert timing, recorded actions, ball ground truth, training or changed physical limits enter this gate. All28 other action channels are copied exactly at the postprocessing boundary. Changed gripper execution can change later observations and therefore later model arm predictions.

Raw responses remain in IPC; `live_observations.hdf5/model_action` saves consumed raw actions and `action` saves executed actions. The analyzer checks the full causal sequence, provenance, native requests and actual jaw execution. A successful run would be **VLA with gripper postprocessing**, not an unmodified-model result. Independent grasp evidence remains required; integration pass alone is insufficient.

## Reproduce

From the project root, run:

```bash
data/derived/hf_pilot_render_repo/.venv/bin/python   scripts/run_hgwind_closed_loop.py   --run-dir data/derived/hgwind_traj18_development_20261031/training_runs/20261031_133555_985430   --frames 269 --prefix 1 --camera-protocol legacy   --gripper-confirmation --run
```

Remove `--run` and add `--check-load` for the CPU-only strict checkpoint/preprocessing preflight. Remove `--gripper-confirmation` to return to unmodified jaw decoding. Existing outputs are preserved in separate run directories.

## Evidence and limits

The saved269-frame prefix1 stream has13 thresholded jaw transitions. This rule applied offline yields one transition at141. This is **not** a prediction of the altered live rollout: the old rollout first contacts the ball at126, before141, and changed actions alter observations. It neither demonstrates grasp success nor resolves premature descent with open fingers. See the [source-hashed summary](../../results/summaries/2026-10-31-ball-gripper-confirmation-preflight.json).

## Separate height-protection proposal

A proposed approach is to retain fingertip clearance during horizontal alignment, then descend toward a predicted interception position and close. Waiting until the ball is literally between elevated fingertips before allowing any descent can deadlock. Wrist tilt, fingertip swept geometry, ball height, motion during descent, estimation uncertainty and controller lag matter; a fixed wrist-height offset alone does not establish clearance.

The current learned-policy interface provides RGB and robot state but no verified visual ball-position/velocity estimator. A geometry-aware descent gate would require one. Simulator target truth may support a clearly labeled privileged diagnostic, but must not silently enter the evaluated vision policy. No height protection is implemented in this experiment, and the older recorded-height diagnostic remains separate. First assess whether jaw confirmation changes contact timing; if open fingers still hit early, prioritize descent/clearance rather than increasing jaw delays.
