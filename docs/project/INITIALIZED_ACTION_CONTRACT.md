# One 29D surrogate with verified fresh60 initialization

## Purpose and current status

The native expert for exposed development episode `local/traj_00001` now reproduces original success exactly after one verified setup-force omission and one native finger-gain reapplication. This does not establish that the frozen29D action representation can retain the grasp. The next run tests that interface under explicitly carried initialization.

**Prepared, physical outcome pending:**338 portable CPU checks, SDK Python3.10 parsing, actual teacher/scene preflight and actual native-entry assembly checks pass. The319×29 float32 action stream is generated from304 captured upper calls plus explicit terminal hold. No new GPU simulation, rendering or training has been launched by the agent.

## Command

```bash
cd /home/robophys/Capstone_Project_Yuan_Haoming

data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_initialized_contract.py --run
```

One fresh process, attempt60 only, maximum319control intervals; no warmup episodes, retries or rendering. Recent launches suggest about1–2minutes including startup, but this new path has not yet been timed. Without `--run`, preparation checks inputs and prints the command only.

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
