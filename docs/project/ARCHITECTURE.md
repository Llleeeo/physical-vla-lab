# Architecture

## Ownership and dependencies

RoboPhys remains a separate dependency. Its task USDs, scene assets, Franka configuration, expert controller, HDF5 recorder, physical evidence and exact-state rendering machinery are reused. This repository contains integration and diagnostic code; it does not redistribute the upstream simulator or asset bundle.

| Layer | Files in this repository | Responsibility |
|---|---|---|
| Action contract | `src/capstone_vla/franka.py`, `task2_bridge.py` | Explicit end-effector frame, FK/IK, gripper convention and simulation control bridge |
| Policy execution | `closed_loop.py`, `live_policy_protocol.py`, `chunk_execution.py` | Observation exchange, action queue boundaries and execution timing |
| Learning/data | `pilot_dataset.py`, `cohort_training.py`; `scripts/convert_task2_cohort_lerobot.py`, `finetune_task2_cohort.py` | LeRobot conversion, training-only normalization and DynamicVLA fine-tuning |
| Diagnostics | `prediction_metrics.py`, `grasp_evidence.py`; `scripts/diagnose_task2_cohort_sampling.py` | Pose/gripper errors, contact evidence and stochastic prediction diagnostics |
| Expert experiments | `hgwind_control.py`, `hgwind_mimic_actuation.py`, `hgwind_capture_gate.py` | Explicit, recorded experimental control variations; not all proposed variants improved collection |
| Cohort verification | `hgwind_cohort.py`; `scripts/prepare_hgwind_cohort.py` | Source identity, raw/core consistency, finite states, clocks, fingerprints and staged copies |
| Rendering | `scripts/render_hgwind_cohort.py`, `run_hgwind_dual_render.py`, `hgwind_remote_render_job.py` | Reuse frozen RoboPhys renderer; pin camera and scene contracts; split work across machines |
| Delivery | `hf_archive.py`, `hgwind_render_delivery.py` | Package provenance, pinned cloud revisions, checksums and verified installation on the main machine |

Module paths in the table are under `src/capstone_vla/` unless a full path is given.

## Data and timing contract

Raw recordings contain timestamps, 9-dimensional Franka joint targets, robot/object states and contact evidence. They are not directly a DynamicVLA action dataset. Conversion must document the robot-base frame, orientation convention, units, absolute/delta semantics, gripper interpretation and observation-to-action causality.

Earlier datasets used 30 Hz images with 60 Hz control. Downsampling actions can remove short gripper events. The current ball rendering pipeline produces 60 Hz visual records with exact source frame indices and timestamps. Matching rates alone does not establish correct causal alignment.

The policy adapter uses an explicit Cartesian pose plus gripper action representation. Offline prediction, converted expert replay and actual image-feedback policy execution are separate experiments. The synchronous development loop pauses simulation during inference and does not measure real-time feasibility.

## Provenance boundaries

Original recordings remain immutable. Rendered copies may enrich camera metadata but must preserve physics records. Both workers use the same scene/camera contract. Raw data, rendered imagery and human review have separate validation states.

The current two collection groups also differ in approach-depth configuration (2 mm versus 0 mm). Machine identity is not a quality label. Any comparison must retain this confound rather than attribute differences to hardware or claim a causal improvement.
