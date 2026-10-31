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

## Remaining controller work

A runtime approach guard is **not implemented or physically validated** yet. It must consider horizontal midpoint alignment, fingertip/body clearance, ball motion during descent and localization uncertainty. Wrist tilt and motion feedforward must remain consistent with any guarded target; merely clipping wrist z can be insufficient. Missing localization must not silently mean permission to descend, and stale estimates must not be held indefinitely. Image-based protection would be an explicit VLA+visual-controller system, not an unmodified VLA result.

The latest failure happens before any closure request: an open finger contacts the ball while the gripper midpoint is horizontally offset. Prioritize this approach geometry; do not use more threshold tuning or retraining to claim it is fixed. The next implementation requires bounded missing-view handling and conservative distance projection through the existing native limits. No new collection is required for the current offline investigation.

[Source-hashed evidence](../../results/summaries/2026-10-31-ball-visual-approach-prototype.json).
