# Ball command smoothing diagnostic

## 31 October 2026 — Position smoothing advances premature contact

The user-run pair `20261031_142217_528773` completed both269-interval/807-callback episodes with unchanged native safety checks. Both fail independent grasp verification and have zero bilateral contact. Raw commands first contact the ball with the right finger at frame120 (2.000s); XYZ-smoothed commands first contact at97 (1.617s), 23frames/0.383s earlier. Jaw commands are identical, with seven transitions; both first contacts precede a requested sustained closure. The alpha0.25 position filter is not promoted.

All six compared initial state arrays are identical between cases. Raw execution reproduces the original first120frames exactly in joint position/velocity and object position/quaternion/linear/angular velocity. This makes the fixed-stream comparison more interpretable than the earlier live height intervention. It remains one exposed command-reexecution case, not live-policy evaluation or a success-rate study. Position filtering changes timing/path while velocity and acceleration feedforward remain unchanged; these observations do not isolate which remaining prediction error causes collision or rule out every smoothing method.

Source-hashed evidence: `results/summaries/2026-10-31-ball-smoothing-result.json`. Both slow-motion MP4 files contain269 checked frames. Next inspect pre-impact position, orientation and feedforward against the successful demonstration before choosing another intervention; default runtime and training remain unchanged.


## 31 October 2026 — Early contact and fixed-command smoothing diagnostic

The recorded-height intervention completed 269 intervals / 807 physics callbacks without a safety abort, but failed grasp verification. First right-finger contact occurred at frame120, before the height intervention138..179. No bilateral contact occurred. Its initial state/camera poses matched the previous refined run, but initial RGB and first model output differed. Consequently this comparison cannot attribute the changed contact outcome to height substitution. See `results/summaries/2026-10-31-ball-height-diagnostic-result.json`.

Before contact (frames80..119), adjacent model XYZ target changes average30.7mm versus3.8mm for the teacher. Actual end-effector steps average2.5mm; target jumps are not measured instantaneous robot motion. Within-chunk jumps are also present. This supports testing command continuity, without proving it is the sole cause of impact.

New bounded experiment: `scripts/run_hgwind_smoothing.py --run` runs raw and causal XYZ-smoothed versions of one saved model stream sequentially. Alpha0.25 initializes at the first raw target; only channels0:3 change. Remaining26 channels, including orientation, feedforward, masks and jaw, are identical. No online inference or expert decision loop executes. Native controller limits and target-blind geometry checks remain active. This is command-reexecution diagnosis, not learned-policy success. RGB is captured and a two-view slow-motion MP4 is exported for each saved case, including failed grasps.

The selected source is the latest hybrid diagnostic, but the input is its raw `model_action`, bound exactly to saved model IPC responses. Recorded expert height is excluded. Later raw predictions still reflect the hybrid run's observations; this limitation is explicit. Both cases execute the identical frozen stream apart from the XYZ filter. The comparison checks matched initial states and whether raw execution reproduces the original first120 frames before attributing an original failure. Filter lag is material: over first120 commands, average displacement from raw targets is38.8mm and maximum218.3mm, including large startup transients. Retaining velocity/acceleration fields isolates position smoothing but may leave inconsistent feedforward. Physical benefit is pending; do not promote this filter by default.

CPU validation: all538 raw/filtered requests decode; seven new tests cover causal filtering, channel isolation, lag, invalid parameters, hash rejection and motor execution without IPC. A temporary fixture built from an actual recorded trace exercises the complete analyzer and rejects changed action data; it is not a new simulation result. Two GPU simulations remain user-run. Existing live/model code is unchanged; rollback is to stop using the separate diagnostic entry point.

Run from the project root:

```bash
data/derived/hf_pilot_render_repo/.venv/bin/python scripts/run_hgwind_smoothing.py --run
```

Outputs: `data/derived/hgwind_smoothing_checks/<timestamp>/comparison.json`, per-case `report.json`, `simulation.log`, `live_observations.hdf5` and `video_review/two_views_4x_slow.mp4`. Integration failures stop the pair and retain evidence. Success/failure of grasp is separate from integration. Inspect command-to-measured-pinch error, first contact, jaw transitions and initial/prefix correspondence before selecting a next intervention.

Release checks: 421 tests passed in a clean tracked export with pinned CPU dependencies and PYTHONPATH=src. Publication audit passed. The video exporter encoded and ffprobe-verified 269 saved frames in a temporary fixture. No new simulation was launched.
