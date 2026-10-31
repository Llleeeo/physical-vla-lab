# One 29D surrogate with verified fresh60 initialization

## Purpose and current status

The native expert for exposed development episode `local/traj_00001` now reproduces original success exactly after one verified setup-force omission and one native finger-gain reapplication. This does not establish that the frozen29D action representation can retain the grasp. The user-run interface result under explicitly carried initialization is recorded below.

**Observed result (31 October): initialization passes, complete execution and grasp fail.** The initialized29D surrogate now verifies startup force/gain alignment but stops after179 complete intervals (180 recorded frames) of319. All180 issued command-boundary fields and jaw targets match the decoded action prefix; full-stream interface flags remain false because the run aborts. First contact166 and bilateral contact170 match the successful teacher. EE error exceeds0.01mm at172 and grows from0.0031mm at171 to4.3217mm at179. Native safety rejects the next command after joint5 reaches-2.8248rad/s against the2.61rad/s limit; required braking also conflicts with acceleration/jerk bounds. Independent grasp fails without sustained lift. This is an expert-labelled interface diagnostic, not learned-policy inference.

The original `initialization_report.json` is preserved. Its `interface=false` requires full319-event coverage; an additional read-only check verifies all180 issued fields and jaw actions match the decoded prefix. Source/callback/settings comparisons find only small numeric representation differences and equivalent quaternion signs in these180 upper calls. Root cause remains unisolated. [Result and hashes](../../results/summaries/2026-10-31-initialized-contract-result.json).

The338 portable CPU checks and actual scene/entry preflight were preparation evidence. The command below describes the already completed run; repeating it without a new controlled change is not the next experiment.

## Matched precision control prepared — 31 October

A single original-double-precision label control is now implemented and CPU checked. The physical result is pending. It keeps the existing decoder, initialized fresh60 process, scene, fixed gains, posture, callbacks and safety limits. It reconstructs the original numeric labels before float32 rounding, with the same quaternion hemisphere choices as the failed float32 run. Casting all319×29 values back to float32 must reproduce that run's action array exactly; widening previously rounded float32 values is rejected.

```bash
cd /home/robophys/Capstone_Project_Yuan_Haoming

data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_initialized_contract.py --precision-control --run
```

One attempt, maximum319 intervals, no rendering or automatic retries. Allow approximately1–2minutes for startup and rollout. The default command without `--precision-control` retains the float32 path.

New outputs end in `_float64`. Both `contract.json` and the analysis identify the precision diagnostic; this does not change the model's declared float32 action contract. Runtime admission verifies the pinned failed baseline and reconstructs original labels again before starting physics. Actual entry assembly and decoder loading are checked on CPU; the assembled collector hash equals the failed baseline.353 portable CPU tests and SDK Python3.10 parsing pass, including unchanged default-entry checks.

The final `PRECISION_CONTROL` line reports initialization, completed intervals, issued-prefix input matching, independent grasp and the combined result. `precision_comparison.json` retains these distinctions when execution is truncated. `initialization_report.json` and the existing full-stream interface gates are preserved. A complete pass supports precision sensitivity in this exposed case, not reliability under model prediction errors or a general remedy. A failure requires comparing the first divergent native states and controls before changing parameters. This experiment does not train or evaluate a VLA.

Rollback: omit `--precision-control`. The new module, tests and optional runner/admission/reporting branches can be reverted without editing the frozen RoboPhys sources, saved runs or SDK.

## Command

```bash
cd /home/robophys/Capstone_Project_Yuan_Haoming

data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_initialized_contract.py --run
```

One fresh process, attempt60 only, maximum319control intervals; no warmup episodes, retries or rendering. This run reached the safety abort after about38seconds of rollout plus startup. Without `--run`, preparation checks inputs and prints the command only.

The source is pinned to the verified teacher under `20261030_022121_219398_fresh60_omit1_gains/attempt_00060`; source and dependency hashes must match the published result. Outputs use a new `data/derived/hgwind_initialized_contract/<timestamp>/` directory. Parent recordings, failed runs, task assets and frozen sources remain untouched.

## What executes

| Component | Source and role |
|---|---|
| Upper actions | Frozen29D float32 labels derived from the verified expert capture; no model predictions |
| Gripper commands | Explicit binary action channel, preserving teacher close/open events |
| Pose control | Existing native effort controller, fixed9/8 pose gains, live-initialized latched robot posture |
| Safety geometry | Existing always-on non-target callbacks and live robot motion; target truth blocked by existing adapter |
| Stage loading | Same explicit stopped-timeline barrier used by verified fresh60 |
| Wind startup | Same exactly pinned single pre-recording omission; every other native call forwarded |
| Finger gains | After original frame0 setup, one native reapplication of10500/140 on both fingers, with caps/targets/USD/arm state invariance and later readbacks |
| Reset observation | Existing history-named boundary reused for attempt60 only; no prior episode is executed |

The online task/skill decision call is replaced by the packet executor. Reused skill helpers configure the fixed gripper and refresh robot/non-target geometry; they do not generate target-tracking or grasp-stage actions. No recorded joint/OSC commands are executed. Privileged original target state belongs to offline labels and the verifier, not a learned observation interface.

`initialized_contract.py` installs force recording, the guarded force proxy, reset observation, passive finger readbacks and the one gain setter explicitly. Merely inheriting those fields in a plan is insufficient. The standard old packet command now rejects inherited force/gain hooks unless the initialized entry is selected; its ordinary uninitialized plans retain their old path.

The float32 action representation, fixed pose gains, latched posture and always-on callbacks remain the previous integrated contract. This is not an exact-source replay or a clean ablation of each individual contract choice. A failure with matched initialization still needs diagnosis within that integrated interface; it does not identify a unique action-channel cause.

## Evidence and outcome interpretation

Inspect the last line:

```text
INITIALIZED_CONTRACT initialization_verified=... interface=... grasp=... combined_passed=...
```

- `initialization_verified`: force/gain audits, stage barrier, reset/packet order, initial robot/object state and first two native ball pose/velocity rows match the verified reference. Original caps and robot limits remain checked by the existing controller/analyzer.
- `interface`: actual decoded motor-boundary fields, always-on callbacks, live posture/raw robot correspondence and packet/no-online-decision audits pass. This field is computed separately from grasp success; the legacy combined contract gate is also preserved.
- `grasp`: independent physical contact/lift/retention result. Missing episodes remain `not_evaluated`; they are not counted as failed grasps.
- `combined_passed`: initialization, full safe319interval execution, interface and independent grasp all pass. A parameter-write message alone cannot satisfy this gate.

New `initialization_report.json` accompanies existing `report.json`, raw HDF5, `actions.npy`, `contract.json`, `model_config.json`, packet execution, force and gain audits, passive drive trace and runtime order audit. Interrupted or failed outputs are retained. Reanalyze initialization without rerunning physics:

```bash
data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_initialized_contract.py \
  --finish data/derived/hgwind_initialized_contract/TIMESTAMP
```

Use the actual printed timestamp. This only analyzes already saved evidence; it does not repair missing simulation output.

## Scope, rollback and next decision

This is one exposed expert-labelled surrogate, not a DynamicVLA inference result, held-out reliability estimate or official benchmark success certificate. Both active measured finger gains still coexist with mimic coupling; this reproduces the reference and does not certify a production gripper model. The explicit force/gain controls retain diagnostic provenance and training exclusion.

On success, preserve this reference and assess a small distinct exposed set before deciding labels/cohort conversion and model training. On failure, inspect whether initialization, motor-field decoding, tracking/contact or retention failed; do not restart random collection or change thresholds based solely on the final label.

Rollback: omit the new initialized runner. Reverting its new module/entry/tests and the optional entry-selection guard restores the earlier integration. No upstream/frozen source edit or SDK upgrade is required.
