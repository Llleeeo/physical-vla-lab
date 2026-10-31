# Visual approach-distance investigation

## Active behavior

The user withdrew the0.8 confirmation rule. The live launcher/runtime/worker/analyzer are restored byte-for-byte to their pre-experiment version at2478647. `--gripper-confirmation` and its runtime module are removed; the action codec retains its original0.5 binary threshold. Old videos, raw data and reports remain. Commit52c153d preserves the rejected experiment for historical reproduction.

## Implemented offline component

`ball_visual_geometry.py` detects orange components independently in two RGB views, matches candidate rays using camera calibration and recorded camera poses, and triangulates a unique match. It rejects negative depth, nearly parallel rays, inconsistent or ambiguous pairs. No object state, task clock, expert actions or source-trajectory lookup enters the estimator. Appearance thresholds are specific to the orange ball and this exposed development setting.

`audit_hgwind_visual_geometry.py` completes estimation before opening object positions for evaluation. It reports RGB-only localization errors and compares estimated ball positions to recorded robot finger reference points. The saved finger points are kinematic reference points, not complete collision meshes. This is CPU-only; it neither launches physics nor changes motor requests.

```bash
data/derived/hf_pilot_render_repo/.venv/bin/python scripts/audit_hgwind_visual_geometry.py --run-dir PATH_TO_COMPLETED_LIVE_RUN
```

The reported intrinsics use the existing camera calibration convention: USD camera forward is-Z/up+Y; RTX square pixels give fy=fx from horizontal aperture and640x480 capture. The evaluator rejects unsupported projection/offsets and binds authored camera optics to the recorded task hash. Known synthetic geometry checks the ray axes, metric scale, ambiguity and missing-view rejection.

## Observations

- Latest run:100/100 precontact frames localized; median/p95/max error1.99/6.56/11.76mm. Full stream147/269 frames localized.
- Previous prefix1 run:126/126 precontact frames localized; median/p95/max error2.04/7.11/12.32mm. Full stream204/269 frames localized.
- These are two exposed runs of the same case, not held-out detector validation. A5mm ray-consistency threshold does not bound3D error to5mm. Bbox centers are biased under perspective/occlusion.

## Optional live approach experiment — implemented, physics pending

`--approach-guard` enables a default-off filter between the retained VLA's raw 29D prediction and the native Cartesian controller. This is **VLA + RGB-guided approach control**, not unmodified VLA performance. It requires prefix 1 and the legacy camera protocol. It cannot be combined with recorded-height intervention. Remove the flag to revert. No training, jaw-threshold change or physics-limit change is involved.

Inputs are the two synchronized live RGB views, their calibration/measured robot-camera poses, current robot pinch pose and finger transforms. Ball position comes only from RGB triangulation. A declared static sphere radius (22.5167 mm in this scene) is read at preflight; no live simulator ball translation enters control. Full local finger bounding boxes are read before rollout and transformed using robot-only link poses after synchronized rendering.

At each control interval:

1. If the horizontal distance between the two finger reference points' midpoint and the visual ball center exceeds 10 mm, place a lower bound on the pinch target's height. The lower bound accounts for the lowest full-finger bounding corner at current and proposed pinch orientations, ball radius, 15 mm localization allowance, 10 mm surface clearance, and a descent reserve of measured downward finger speed times 0.12 s. Negative vertical target velocity, target acceleration and velocity-reference components are set to zero consistently; upward components remain. Horizontal model commands and the original jaw score remain.
2. If aligned, pass the model action unchanged. Alignment is a geometric permission, not a contact/grasp-success test.
3. Missing, ambiguous or inconsistent views, or an inter-frame visual jump over 40 mm, latch the current robot pose as a hold target. Translation feedforward is zeroed and the competing orientation-axis objective disabled. The hold does not follow subsequent drift. Two consistent valid visual estimates are required to resume. Stale ball coordinates are never extrapolated. The original jaw score still passes unchanged.

The 15 mm localization allowance exceeds the worst observed 12.32 mm error in the two exposed streams, but it is **not a statistical bound**. The 10 mm alignment/clearance and 0.12 s reserve are declared engineering hypotheses, not tuned or physically certified guarantees. The filter does not predict ball interception, guarantee swept-volume clearance under intermediate rotations, compensate arbitrary tracking error, or protect the entire arm. Persistent occlusion can cause a hold until the bounded episode ends, preventing a successful lift. It may also fail to align if the model's horizontal request is wrong. A pose-hold request is not an instantaneous physical stop.

Raw model responses, filtered actions, RGB, camera poses, robot finger points/corners and each decision are retained. The analyzer independently rebuilds localization/filtering from the archive, binds original model IPC responses, verifies unchanged jaw values and native jaw decisions, and checks independent grasp separately. Runtime hashes include the filter and vision estimator. The original 0.5 jaw codec remains in force.

## Verification and next run

The actual scene/optics/finger-bound preflight and strict CPU checkpoint load pass, alongside focused CPU checks for alignment, tilt, missing views/reacquisition, velocity references, raw prediction retention and original jaw transitions. No new GPU/physics rollout has been executed for this guard. The next bounded run must inspect first unilateral contact, midpoint offset, visual holds, bilateral contact and independent grasp; integration pass alone is insufficient.

```bash
data/derived/hf_pilot_render_repo/.venv/bin/python scripts/run_hgwind_closed_loop.py \
  --run-dir data/derived/hgwind_traj18_development_20261031/training_runs/20261031_133555_985430 \
  --frames 269 --prefix 1 --camera-protocol legacy --approach-guard --run
```

[Localization evidence](../../results/summaries/2026-10-31-ball-visual-approach-prototype.json) · [Guard CPU preparation](../../results/summaries/2026-10-31-ball-visual-approach-preflight.json).
