# Results and limitations

Snapshot: **2026-10-28, Asia/Hong_Kong**. These are development observations, not a benchmark leaderboard or a paper's final test results.

## Wind-driven spherical-object cohort

Task: `grasp_wind_driven_motion_1s1p_hgprgew`.

- 619 saved attempts: 40 accepted successes and 579 failures. Collection was intentionally stopped at 40 successes; its original target was 100.
- Main-machine group: 21 successes; remote group: 19. The groups used different approach-depth settings and must retain their provenance.
- 12,154 control frames, recorded at 60 Hz; 40 distinct numeric trajectory fingerprints.
- Raw/core HDF5 consistency, finite actions/states, timing, scene identity and contact indexing checks passed.
- Independent recording analysis found positive impulses on both fingers and no attachment assistance in these 40 episodes. These checks complement, rather than replace, task verification and visual review.
- Two revised-camera sample episodes (209 and 335 frames) passed all frame-index/timestamp checks. Five RGB views were sampled for blank frames; none were flagged. The remote sample completed upload, pinned-revision download and local verification.
- Remaining 38 revised-camera renders were started. This document does not assume their completion. No ball-cohort VLA training or held-out split is claimed yet.

A 40/619 collection ratio describes this adaptive development collection only. It is neither a policy success rate nor an unbiased estimate of a frozen expert's generalization.

## Earlier DynamicVLA integration: irregular egg-object task

Task: `grasp_sliding_motion_on_flat_surface_1s1p_g9wvkk1`. Earlier shorthand called the object a ball; the target is an egg. This task remains a difficult development case.

A 2,000-step run used 40 training episodes and eight same-task validation episodes. Normalization was fitted on training data only. The fixed validation flow-MSE probe decreased from **12.1492 to 0.4197**. This is a flow-prediction diagnostic, not a physical distance or grasp-success score.

The saved final offline first-action evaluation covered 1,848 validation frames: mean position error **9.16 mm**, p95 **17.36 mm**, mean orientation error **2.49 degrees**, and gripper-sign accuracy **88.53%**. These measurements condition on recorded observations. They do not establish performance on states visited by the live policy, and they are not directly comparable to evaluations with different sampling noise or execution prefixes.

Subsequent closed-loop debugging has not established reliable learned grasping. Camera consistency, action timing, velocity references, stochastic action continuity and visited-state distribution remain distinct sources of failure. Better offline fit is insufficient evidence of closed-loop success.

## What cannot yet be concluded

- No established VLA robustness gain under physical disturbances.
- No independent physical-generalization test score.
- No evidence that a 2 mm depth bias alone caused better demonstrations.
- No automatic Franka-to-UR10e transfer and no completed real-robot experiment.

The compact [evidence snapshot](../../results/summaries/2026-10-28.json) records source-report hashes and selected metrics. Full records remain outside Git; hashes establish provenance, not public reproducibility without access to those records.
