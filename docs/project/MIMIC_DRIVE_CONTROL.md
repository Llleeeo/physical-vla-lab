# Contact-stage drive control

Status: **31 October 2026 — CPU validated, physical result pending.**

## Verified observations

The initialized float32 surrogate and successful float64 control both develop alternating finger and wrist velocities after contact. In the same recorded 2.84–2.99 s window, joint5 maximum absolute velocities are 2.825 and 2.047 rad/s; maximum absolute sample-to-sample accelerations are about339.47 and376.48 rad/s² respectively. These are measured finite differences, not requested accelerations or continuous-time impact estimates. This window stops before the failed run's later post-abort records.

At the float32 safety abort, joint5 measured velocity is -2.82483 rad/s versus the preceding model prediction -0.95695 rad/s; the previous model acceleration was only -3.25063 rad/s². This shows a large response not captured by that one-step model. It does not isolate its cause. The successful precision run is not evidence of smooth or robust contact control.

The recorded native gains are10500/140 on BOTH fingers. Authored joint2 also retains PhysxMimicJointAPI:rotX, reference joint1, gearing-1 and zero offset. The restored warm-episode dual-drive configuration was deliberately a reproduction diagnostic; it has not been certified as the production gripper model.

[Source-hashed observations](../../results/summaries/2026-10-31-contact-drive-inspection.json).

## Hypothesis and bounded experiment

Independent follower drive combined with mimic/contact coupling may contribute to the sensitivity. Test one changed factor: set the follower's native stiffness/damping to0/0, retaining the leader at10500/140. Reuse the original live setter and verify both immediate and subsequent native readings. Require the original authored mimic relationship before the setter and on subsequent snapshots; retain the coupling and all USD properties.

This changes active gripper drive authority: unchanged70N caps do NOT imply unchanged total gripping force. A loss of grasp may therefore reflect reduced drive authority. Neither a success nor failure alone proves an overconstraint defect. No extra grip assistance, friction edit, relaxed arm limit, online expert decision, training or rendering is added.

The exact319×29 float32 actions, initial episode/seed, force-omission control, leader gains, jaw commands, scene, controller assembly, arm safety, existing callbacks, and remaining dependencies are checked against the saved failed float32 baseline. This is one exposed development case, not an evaluation split or a production fix.

## Run

```bash
cd /home/robophys/Capstone_Project_Yuan_Haoming

data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_mimic_contract_control.py --run
```

Without `--run`, the entry only performs CPU preflight and prints the plan. One attempt,319 control intervals; no retry or rendering. Output is a fresh `data/derived/hgwind_initialized_contract/*_single_drive/` directory.

Inspect `mimic_drive_comparison.json`, `initialization_report.json` and `gripper_gain_control.json`. Report intervention validity, initialization, full safe execution, actual decoded commands, physical grasp and contact-window motion separately. Missing/incomplete experiments remain unevaluated. Keep the collector's existing unused-FSM labels; independent grasp evidence is not official benchmark success.

Rollback: use the original `scripts/run_hgwind_initialized_contract.py` entry. Its configuration is unchanged. Remove the new opt-in adapter/entry/hooks to revert code; no frozen source, original trajectory, SDK or USD is edited.

If the case improves, verify repeatability and small action-error tolerance before adopting it or converting more data. If it does not, retain the failure and use it to reject this intervention; do not automatically launch a parameter sweep.

Validation: 369 portable CPU checks pass (69 focused), SDK Python3.10 parses the changed runtime files, and actual-source single-drive CPU preflight passes. Legacy float32/float64 entries retain dual-drive settings. Physical result pending.
