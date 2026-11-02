# Starting inventory: HT and dynamic manipulation

Checked on 2 November 2026 against the existing research checkout. This is an artifact audit, not a fresh reproduction of the historical experiments. [Machine-readable evidence and hashes](../../results/summaries/2026-11-02-ht-readiness-audit.json).

## What can be reused

The DynamicVLA checkout is pinned to `434dc7105cf6d559018d15af5fecbcdad44f036e`. The inspected model environment reports Python 3.12.7, PyTorch 2.7.1+cu126, Transformers 5.2.0 and LeRobot 0.3.3. Simulator files report Isaac Sim 4.5 and Isaac Lab 2.2.1. These versions are recorded separately because model dependencies and the simulator run in different environments; metadata inspection does not establish current CUDA or simulator health.

The earlier task has a LeRobot export with 40 training episodes (11,128 frames) and eight validation episodes (1,848 frames). The corresponding parquet files and two videos per episode are present. Its export report records successful upstream-reader checks, but also declares attached-object assistance in the source demonstrations. It is useful for data-interface regression checks, not a certified unassisted dynamic baseline.

The wind-driven ball collection contains 40 manifest entries and 12,154 frames. All 40 raw replay files and all 40 rendered HDF5 files referenced by the inventory exist. The saved audit checks timestamps and samples three RGB frames per camera; it is not a full manual review. Raw actions have nine channels. The subsequent development model uses 29 action channels and 30 robot-state channels. The saved readiness report explicitly says that multi-episode training is not ready. Full upper-command labels have been prepared for an exposed training case, `local/traj_00018`; the remaining raw episodes cannot simply be passed to that adapter.

Existing training checkpoints are present for the earlier multi-episode experiment and ball refinement. They preserve the old baselines; they are not HT checkpoints and were not reloaded in this audit.

## Code map and current data flow

- `src/capstone_vla/hgwind_cohort.py` and the collection/render scripts preserve episode identity and delivery provenance.
- `src/capstone_vla/hgwind_development_data.py` builds the exposed ball training case from aligned RGB and captured upper commands. Its `aligned_indices` and `normalization` functions enforce the current data conventions.
- `src/capstone_vla/ball_action_contract.py` defines the 29D action codec and 30D state. It includes motion references and controller-related fields beyond a conventional pose-and-gripper action.
- `scripts/finetune_hgwind_development.py` adapts the pretrained model to this contract. `scripts/predict_hgwind_development.py` evaluates generated actions offline.
- `scripts/run_hgwind_closed_loop.py`, the policy server and simulation worker connect live images to model commands. Optional RGB approach/capture rules make some runs hybrid controllers; they must be labelled separately.
- `src/capstone_vla/grasp_evidence.py` and `ball_repeat.py` distinguish integration completion from physical grasp outcomes.

The implemented development path is: recorded expert motion and rendered RGB → aligned command labels for the selected case → DynamicVLA fine-tuning → live observation server → action codec → simulated controller → independent physical checks. The arrow from all raw episodes to complete 29D labels is not established.

The launcher explicitly reports `synchronous_pause_during_inference`. It supports engineering debugging, but does not measure the benefit of acting faster while the target continues moving. The new study needs an evaluator that represents inference delay in physical simulation time.

## Historical results now reconciled

The frozen five-repeat batch is complete. Its saved categories are three completed grasp failures, one runtime safety abort and one incomplete/invalid result, with zero recorded grasp passes. The incomplete result must remain visible rather than becoming a fourth completed failure. All runs use one exposed initial condition and vary model sampling noise; they do not measure generalization across scenes. See [results](RESULTS.md).

## Candidate data and unresolved checks

An existing screen identifies `grasp_sliding_motion_on_inclined_plane_1s1p_z5jnepn` as a candidate with 256 catalogued trajectories. It is a regular box on an incline, not a ball. Three sampled episodes are source-labelled successes with matching scene hashes and no recorded attachment. They use position-drive arm execution, unlike the later ball effort-controller experiments. The sampled camera view also has visibility concerns. This is a shortlist entry, not an approved task or a verification of all 256 episodes.

No exact natural translating-ball match was established by that screen in its accessible subset. The wind task remains useful historical evidence but is deferred as the principal latency experiment. Task selection and any rendering need a separate decision after the HT reproduction setup is understood.

The prior HT source review identifies a state-based RoboMimic Lift experiment (`configs/experiment/robomimic/state/lift/ph/ht_transformer_nu200.yaml`) as a candidate starting point. It is not a frozen runnable configuration yet. There is no HT checkout under the inspected project `third_party` directory, and neither inspected Python environment reports an installed RoboMimic or MuJoCo distribution. The next work is to verify the official revision, dependencies, dataset variant and checkpoint availability before creating an isolated reproduction environment.

The GPU query failed to communicate with the NVIDIA driver in this audit session. This does not identify the cause or prove a hardware fault. GPU visibility must be checked in the intended execution terminal before training. No remote-machine readiness is implied.

## Decisions

Reuse the data checks, evidence schema and historical baselines. Keep the simulator environment intact. Start the HT reproduction in a separate environment; choose a dynamic task only after checking its controller and dataset compatibility. Do not adopt the 29D ball contract as a requirement for every future task. New HT/Flow comparisons need fixed task splits and common execution settings, with actual-delay and matched-delay evaluation.
