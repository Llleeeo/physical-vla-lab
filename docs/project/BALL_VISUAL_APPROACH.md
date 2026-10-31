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

## Observed first live result

The optional filter completes 269 intervals / 807 native callbacks without a safety abort. All raw model responses, unchanged jaw scores, filter decisions and native execution pass the corrected audit. Independent grasp fails: zero left, right or bilateral contact intervals. Unaligned height protection runs for 220 intervals, alignment passes for 9 (134–142, 0.15 s), and missing-view pose hold runs for the final 40 (229–268).

During the permitted descent window, the lowest recorded finger bounding corner remains approximately 30.31 mm above the ball top at frame 134 and 17.49 mm above it at 142. At 143 horizontal alignment is lost and height protection returns. This supports a missed descent/interception window; removing early contact alone has not made a grasp. These bounding-height measurements use object truth only after execution and do not certify full mesh clearance. The guard preserves jaw outputs; first raw close is 102, with continued switching.

The original `INCOMPLETE` report was a verifier error: exact equality of recomputed floating geometry rejected roundoff (maximum action difference 7.22e-16). The revised audit uses absolute tolerance 1e-12 for floating geometry/actions, while decisions, original model responses, jaw values and untouched channels remain exact. Only the explicitly identified legacy analyzer revision can be superseded; other runtime hashes still must match. The original report is retained. No simulation rerun was needed. A 269-frame, 4x-slow two-view video is exported from the actual saved RGB.

This is one exposed VLA-plus-visual-controller trial, not evidence of general grasp improvement. Keep the guard optional. Next inspect interception/vertical response time within the alignment window before changing clearance or adding more training. [Source-hashed evidence](../../results/summaries/2026-10-31-ball-visual-approach-live-result.json).

## Optional near-ball horizontal following — CPU ready, live result pending

The next experiment adds `--approach-follow` to `--approach-guard`. It aims to extend horizontal alignment while the existing height rule permits descent. It deliberately does **not** implement an earlier timed descent release. Inspecting the frozen rollout showed that linear extrapolation of recent *relative hand motion* can predict alignment that never occurs, because later VLA hand commands change. For example, at frame 126 a seven-frame linear fit and a hypothetical 0.05 s + clearance / 0.1 m/s descent horizon predict about 3.45 mm horizontal offset; the actual saved relative position at the corresponding frame 146 is about 17.6 mm. This is a diagnostic of that prediction on a frozen stream, not a prediction of a changed rollout.

The declared alternative is VLA plus local visual XY assistance:

- Collect seven consecutive accepted RGB ball positions (0.1 s span at 60 Hz). Fit velocity using centered least squares. Require maximum 3D fit residual <=4 mm and fitted speed <=0.5 m/s. Missing/jumping observations and the existing reacquisition hold clear the history. These are engineering rejection settings, not guaranteed perception bounds.
- Enable only when the measured horizontal finger-midpoint to visual-ball offset is <=80 mm. This explicit near region keeps distant approach with the learned policy. It is not a proven optimal capture range.
- Set the XY pinch target so the *desired finger midpoint*, accounting for proposed orientation and the current pinch-to-midpoint offset, follows the visually estimated current ball center. Set horizontal target velocity and optional velocity reference to the fitted ball velocity, and horizontal acceleration feedforward to zero. Preserve all masks, native speed constraints and all remaining action channels relative to the height-only guard.
- Apply the original height guard and 0.5 jaw decode unchanged. There is no forced closure, reopening delay, hidden grasp phase or future expert lookup. Poor velocity fit or being outside the near region falls back to the original height-only behavior; missing localization still holds the latched pose.

The feature explicitly replaces some learned horizontal commands. Any eventual success is a hybrid-controller result, not evidence that the original policy learned better interception. Fit rejection can switch the assistance off, and the model's jaw instability may still prevent grasp. Velocity estimation uses RGB only; live target state is not accessed.

`check_hgwind_approach_follow.py` replays saved observations, rebuilds perception, binds raw predictions, verifies the original guard decisions and checks that only eight horizontal channels can differ. On all 269 saved frames, native command decoding succeeds with no rejected actions. Assistance is active on 115 frames, first 93 and last 207; height, orientation, masks and jaw outputs remain exactly those of the old guard on these *same* observations. Actual trajectories, future images and alignment-window duration after intervention are unknown. This preview neither executes a model nor runs physics.

```bash
data/derived/hf_pilot_render_repo/.venv/bin/python scripts/run_hgwind_closed_loop.py \
  --run-dir data/derived/hgwind_traj18_development_20261031/training_runs/20261031_133555_985430 \
  --frames 269 --prefix 1 --camera-protocol legacy \
  --approach-guard --approach-follow --run
```

Remove `--approach-follow` to restore the height-only experiment; remove both switches to restore the unfiltered VLA route. Keep raw/filtered actions, vision history and native contact checks. Acceptance for the next run is longer physical alignment permitting descent without premature unilateral impact, followed by independent grasp evaluation. No new live improvement has been demonstrated. [CPU evidence](../../results/summaries/2026-10-31-ball-visual-follow-preflight.json).

## 31 October 2026 — Follow result and near-capture repair

The user-run follow experiment `20261031_180308_256823` saves168 frames and completes167 of269 requested intervals before a native acceleration/jerk/next-step velocity feasibility abort. It achieves first contact142, first bilateral158 and nine bilateral frames, with longest continuous bilateral duration0.1167s. There is no sustained lift. This is an incomplete episode, not successful grasp verification. The longest visually authorized alignment stretch113–134 lasts22 frames, compared with nine in the historical height-only run; changed closed-loop observations prevent treating this as a controlled success-rate comparison.

Wrist-ball truncation first causes missing localization135 and a world-pose hold. At162 and167 vision briefly returns, but the seven-frame visual-velocity history is not ready: v1 falls back to model goals about10cm away. Independently, joint5 already oscillates before these switches; terminal measured speed is-2.8223rad/s against the-2.61 bound. Therefore goal continuity is a concrete defect to repair, not proof of the entire abort cause. Native arm limits and torque/contact control are unchanged in this patch.

`--near-capture` (requires `--approach-guard --approach-follow`, prefix1/legacy) adds two explicit engineering changes:

- Prefer the original stereo estimator. Only when no stereo pair exists, fit the visible silhouette of a large orange wrist-image component cut by the image boundary to a sphere of the declared static radius. Ignore boundary pixels within8px of the image edge, require at least210 degrees of angular contour coverage, tangent-residual p90<=2mm, center range<=0.35m, and exactly one external-camera orange candidate whose ray is within5mm. Ambiguous, absent, nonconverged and inconsistent fits remain rejected. This assumes sphere shape, color, size and calibrated cameras; it is not a generic detector or certified depth estimate.
- After near-follow has engaged, inadequate vision/velocity fits hold the measured pose rather than release a raw distant target. Following resumes through a position step<=3mm and quaternion step<=1degree per control interval, with motion feedforward/axis goal suppressed during recovery. The existing height-protection floor takes precedence over these step bounds. This does not bound actual robot dynamics. A persistently missing target may still cause the episode to hold/fail.

The optional `--capture-depth-mm 5` changes the aligned, valid-follow target down by up to5mm, subject to a10mm clearance proxy above the visually estimated sphere bottom, using full finger bounds at the proposed orientation. The sphere bottom is not a measured support plane and this is not swept collision certification; native environment safety remains. The default is0 to isolate the visual repair before the depth ablation. Original jaw score/threshold0.5, model weights, scene and native limits remain unchanged. Rollback: omit `--near-capture` and the depth flag to use the original follow branch.

`scripts/check_hgwind_near_capture.py --source-run <run>` reconstructs the old issued stream, binds raw model responses, decodes both new candidate streams, and separately scores localization against recorded object truth. The runtime estimator/controller never receives that truth. On the saved168-frame stream,28 failed stereo frames gain fallback localization, with mean3D error1.737mm and maximum3.228mm. Both depth streams have zero native decode rejects and identical raw jaw scores. In the continuity-only ablation using old detections, goal displacement at162/167 becomes zero. New full preview has61 active-follow frames and13 recovery-hold frames; these are frozen-input calculations, not new observed motion. The saved geometry is from the old interrupted run and cannot predict new contact or stability.

First bounded user-run experiment uses extra depth0. Check episode completion, contact-stage oscillation, bilateral retention and lift; an integration pass alone is insufficient. If oscillation persists, inspect effort prediction/residual and contact response rather than increasing safety thresholds or training steps. [Source-hashed evidence](../../results/summaries/2026-10-31-ball-near-capture-preflight.json).
