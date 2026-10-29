# Ball action contract v1: one integrated development gate

## Decision and purpose

**Prepared, physical result pending.** Freeze the passing fixed-9/8 gain and live-initialized latched-posture baseline. Stop successive parameter-removal probes. Test a complete model-sized command interface once, then validate 3–5 distinct development episodes before cohort conversion/training. If it fails, report the integrated blocker and reconsider the interface rather than starting an open-ended deletion campaign.

The present command is an **expert-labelled action surrogate**, not DynamicVLA inference. It answers whether commands can pass through the proposed float32 data representation and decoded controller interface while retaining a physical grasp. It does not establish learned success, generalization or real-time performance.

## Implementation and verified upstream capacity

- [`ball_action_contract.py`](../../src/capstone_vla/ball_action_contract.py): complete field accounting, encoder/decoder, robot-state schema, configuration adapter and actual motor-boundary audit.
- [`run_hgwind_action_contract.py`](../../scripts/run_hgwind_action_contract.py): parent provenance/physical-scene preflight and one isolated simulation command. Default is CPU dry-run.
- Existing trace/analyzer have opt-in branches; old probe commands keep their original behavior. Frozen RoboPhys, raw trajectories and passed/failed historical reports remain unchanged.

Pinned DynamicVLA revision `434dc7105cf6d559018d15af5fecbcdad44f036e` supports maximum action/state widths of 32. The local official checkpoint has `action_in_proj.weight=(720,32)`, `action_out_proj.weight=(32,720)`, and `state_proj.weight=(960,32)`. The real upstream configuration class accepts action width 29 and state width 30. This was checked without instantiating a model or starting CUDA. It verifies **shape compatibility only**, not a new checkpoint load, forward/backward pass or trainability outcome for this representation.

This is a custom RoboPhys action representation using the pretrained architecture. It changes the pretrained action-channel semantics; no zero-shot compatibility is implied. The earlier egg-task 7D adapter/training remains separate.

## Explicit action schema: 29 float32 channels

All pose, velocity, acceleration and orientation-axis quantities use the **simulation world frame**. Position is absolute metres; quaternion order is **wxyz**; velocity is m/s and acceleration is m/s². Actions and images remain at 60 Hz; native physics/effort updates remain at 180 Hz. Desired velocity and acceleration are independent expert command labels; do not derive them by differentiating pose or bounded velocity.

| Channels, zero based | Meaning |
|---|---|
| 0–2 | Absolute pinch target position |
| 3–6 | Absolute target unit quaternion |
| 7–9 | Target linear velocity |
| 10–12 | Target linear acceleration |
| 13 | Acceleration present flag |
| 14–16 | Relative-motion linear velocity reference |
| 17 | Velocity reference present flag |
| 18 | Maximum relative linear speed reference |
| 19 | Relative-speed reference present flag |
| 20–22 | Orientation task axis in world frame |
| 23 | Orientation axis present flag |
| 24 | Orientation roll weight |
| 25 | Soft joint-limit avoidance margin, 0–0.08 rad |
| 26 | Preserve-orientation solver mode |
| 27 | Operational-orientation mode |
| 28 | Binary close-gripper request |

Nullable flags preserve the difference between an absent reference and a present zero vector. Binary values decode at 0.5. Quaternions use causal sign continuity in labels and normalize on decode; active axes normalize too. Degenerate/nonfinite vectors reject. Roll weight, the soft avoidance margin and relative speed are bounded by the documented contract; these channels cannot change manufacturer limits. There is no model-controlled callback-disable channel, hidden contact/FSM phase, terminal timer schedule, saved joint target or cached Jacobian/OSC output. The final source hold is represented by explicit pose/motion commands, not a separate expert phase.

All remaining native fields must match an explicit constant controller profile (9/8 gains, zero angular feedforward, the existing speed/acceleration/jerk configuration, task metric and solver settings). Unmapped fields or differing constants cause preflight rejection; they are not silently placed in a side file. This profile is currently specific to the exposed ball development setup, not all RoboPhys tasks.

## Robot state: 30 float32 channels

`arm q[7], arm qdot[7], measured world pinch position[3], measured world pinch quaternion[4], latched robot posture[7], measured finger positions[2]`.

The latched posture is generated once from live initial robot joints and declared robot-side variation, then exposed in proprioception. Otherwise demonstrations with different nullspace anchors would have an undisclosed control dependency. It is not a target-object observation. The codec declares this state interface; live model-state assembly and dataset export are later work, not implemented training capabilities.

## Controller and safety boundary

The existing native feedback/effort pipeline recomputes inner commands from current robot state. It keeps physical jaw contact and existing hardware constraints. Non-target geometry helpers use causal measured robot motion and explicit jaw mode; target-object queries remain blocked.

**Both collision callbacks are now enabled at every tick.** In the passing recorded packet baseline they were disabled in the final 36 frames. The integrated experiment therefore tests this protection change together with float32 representation and removal of hidden phase scheduling. It is not an exact-source replay or a one-factor experiment; a failure cannot by itself separate these changes. It is not a hardware safety certification.

## Model and data integration requirements

1. Set both model `use_delta_action=False` and upstream dataset `delta_action=False`. The upstream delta implementation subtracts/adds state across all action dimensions except the final one; applying it to this schema would corrupt velocity, acceleration and mode fields.
2. Fit new 29D action / 30D state statistics on the designated **training episodes only**. Frozen controller constants are not learned channels. No held-out episode/physical-condition statistics may enter normalization.
3. Preserve sequence padding (`actions_id_pad` in the pinned upstream forward), and report meaningful losses on the first 29 action dimensions. Upstream pads to 32 and its aggregate loss still includes padded dimensions; do not present that aggregate alone as command accuracy.
4. Keep camera history, causal timestamps, explicit gripper transitions and instruction matching. Expert object state can generate supervised command labels; it must not enter the policy observation or runtime request decoder.
5. Existing 40 ball HDF5 episodes contain processed 9D actuator targets, **not all new upper-command labels**. They cannot be renamed or directly converted to this contract. After the integrated gate, capture/validate the required labels on 3–5 exposed development episodes before deciding controlled reexecution versus new collection. Preserve the original 40 datasets and camera assets.
6. Reuse pretrained backbone/action-expert weights through the official loader and new normalizers. A 29D model smoke and fine-tune are still pending; the current step does not train or run DynamicVLA.

## Run and acceptance checks

From the project root:

```bash
data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_action_contract.py --run
```

Expected duration is similar to previous one-episode effort probes; startup determines much of the wall time. Output is a new timestamped `data/derived/hgwind_action_contract_v1/` directory containing `actions.npy`, `contract.json`, `model_config.json`, `plan.json`, traces and `report.json`. No source files are overwritten.

The integrated gate requires complete 209 intervals, initial-state matching, 627 physics and OSC updates, unchanged safety/limits, exact action-to-motor-boundary decoding, raw robot-state/posture consistency, always-enabled collision callbacks, no auxiliary grasp, and independent contact/lift/retention passing. Inspect **`BALL_ACTION_CONTRACT_PASSED True`** and **`INDEPENDENT_GRASP pass`** together. Collector/FSM success labels remain unmodified and are not this engineering result.

On pass: freeze this contract and test 3–5 distinct development episodes; then prepare labelled cohort conversion, episode splits and a real 29D model smoke. On failure: inspect this integrated failure against the preserved passing baseline and state the blocker before further controller changes. Do not move directly to training or claim the benchmark/policy has failed.

## CPU verification

Actual 209-frame codec preflight preserves every gripper/null/controller mode. Maximum world-position roundtrip error is 0.000237 mm; sign-invariant quaternion error is approximately 3.68e-8. This is a numerical-format check, not physical validation. CPU regressions also reject unaccounted native fields, changes to fixed settings, degenerate outputs and action-file tampering; the runtime branch is checked without access to saved HDF5 action values.

Rollback: omit the new entry point and use the earlier passing packet command; the new optional trace/analyzer branches are inactive for existing plans. Reverting this milestone removes the adapter and its optional branches without changing raw/frozen artifacts.
