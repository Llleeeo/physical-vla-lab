# Physical VLA Lab

**Evidence-based research on dynamic manipulation and physical generalization in vision-language-action policies, built on RoboPhys.**

This project connects RoboPhys simulation, expert demonstrations, synchronized visual data, and pretrained VLA policies. The initial testbed is dynamic grasping with a simulated Franka arm. The research objective is to understand when learned behavior transfers across physical conditions and why it fails.

This is an independent research integration, not the upstream RoboPhys benchmark or an official DynamicVLA release.

## Current status — 29 October 2026

| Workstream | Verified progress | Remaining work |
|---|---|---|
| Franka action interface | Recorded-tool FK verified across all 40 ball episodes; explicit Cartesian request execution grasps successfully on one exposed seed without online task/skill decisions | Live policy action contract and learned dynamic grasping |
| DynamicVLA baseline | Pretrained-model loading, fine-tuning and offline evaluation on an earlier egg-object task | Offline prediction improvements have not established robust closed-loop success |
| Wind-driven ball demonstrations | 40 collector-accepted physical grasps; recording/contact checks and descriptive quality screen completed | Full visual review and downstream training conversion |
| Visual data pipeline | All 40 revised-camera episodes verified at 60 Hz (12,154 frames); all 19 remote episodes delivered through the private Hub to the main machine | Trajectory quality/visibility review and downstream action conversion |
| Physical generalization | Evaluation questions and failure-analysis workflow defined | Independent held-out physical conditions and systematic experiments |
| UR10e | Later-stage research direction | Embodiment adaptation, simulation validation and supervised hardware experiments |

The 40 ball episodes are **expert development data, not 40 VLA successes**. No disturbance-robustness improvement or sim-to-real transfer is claimed. The [effort-control probes](docs/project/HGWIND_EFFORT_PROBE.md) exposed an action-reference mismatch: recorded-motion joint tracking runs safely but loses the ball during lift. A constructed task-space reference performed worse; endpoint semantics were audited and the native pose-servo adapter aborted after contact. The original expert and complete-control physical replay both reproduce one source episode exactly across all 15 core numeric arrays. The native geometry implementation now recomputes all 194 task Jacobians from live robot state and exactly reproduces the successful episode. Other logged command references remain. Zeroing task-acceleration feedforward completed safely but missed the object; retain the passing reference while defining the complete Cartesian feedback contract. The [original Cartesian target capture](docs/project/CARTESIAN_TARGET_CAPTURE.md) now verifies 194 upper calls with exact source-array reproduction. Upper-target feedback reexecution now passes physical grasp checks over 209 intervals: all 194 regenerated control references and all 15 core arrays exactly match the exposed source episode. This component explicitly retains native expert scheduling, source settings, live safety callbacks, gripper and terminal fallback. An [explicit packet interface](docs/project/CARTESIAN_PACKET_INTERFACE.md) now bypasses online expert decisions and owns jaws/terminal Cartesian hold; after correcting geometry-query, jaw-mode and measured-robot-motion context, its physical run passes all 209 intervals / 627 OSC updates and independent grasp checks. Continuous bilateral contact lasts 1.2 s with at least 74.861 mm retained lift. All first 194 core frames match the source exactly; the final 15 intervals use Cartesian hold. The recorded motion/settings schedule remains, so this is not a learned-policy result. 223 CPU checks pass. The fixed-9/8 gain diagnostic also passes all 209 intervals and independent grasp checks, retaining at least 72.577 mm lift without the later gain switch. The measured-posture ablation executes safely but loses final grasp retention. In this exposed case the passing posture anchor reconstructs exactly from initial robot joints plus sampled robot trajectory variation; preserve the passing fixed-gain baseline. Live-initialized latched posture now passes two same-seed repeats, with identical core arrays, 1.2 s bilateral contact and about 72.595 mm retained lift. This verifies one exposed case; it is not a cross-trajectory success-rate result. Other recorded settings/motion dependencies remain. A live policy action contract remains unverified. See [results and limitations](docs/project/RESULTS.md).

The first [integrated ball action-contract check](docs/project/BALL_ACTION_CONTRACT.md) now completes 209 intervals safely and verifies the 29D action boundary, live posture and always-on geometry callbacks. **Final grasp retention fails:** bilateral contact lasts 0.433 s rather than the passing reference's 1.2 s. Two strict-equality report issues were corrected for normalized-vector roundoff only; fixing the report does not change the physical outcome. Contact-stage divergence precedes the changed callback activation. Freeze the interface and use a bounded 3–5 distinct development-episode diagnostic before further controller changes or formal training. No new VLA result is claimed; existing 9D ball recordings lack complete new upper-command labels. See the [source-backed result](results/summaries/2026-10-29-ball-action-contract-result.json).

The [bounded paired diagnostic with automatic all-outcome videos](docs/project/CONTRACT_VIDEO_BATCH.md) is now prepared: three predeclared schema-supported local cases, at most six new physics runs, plus videos of both existing references. Every saved success/failure receives five 30 Hz review views and a unified human-review index. CPU checks pass; GPU execution and new videos are pending.

## Workflow

```mermaid
flowchart LR
    A[RoboPhys scenes and task semantics] --> B[Simulator expert]
    B --> C[Raw actions, states, contacts and provenance]
    C --> D[Recording and physical-evidence checks]
    D --> E[Calibrated 60 Hz multi-view rendering]
    E --> F[Data alignment and explicit action conversion]
    F --> G[Pretrained DynamicVLA fine-tuning]
    G --> H[Closed-loop Franka evaluation]
    H --> I[Failure analysis and held-out physical evaluation]
```

Simulator state is available to experts and evaluators. A learned policy must use its documented camera, proprioception and language inputs; object ground truth must not become an undisclosed policy input.

## What this repository contains

- `src/capstone_vla/`: action interfaces, execution adapters, data checks, metrics and verified archive delivery. The historical Python package name is retained for compatibility.
- `scripts/`: research and diagnostic entry points, including collection and two-machine rendering orchestration.
- `tests/`: CPU regressions plus environment-dependent integration checks.
- `configs/examples/`: sanitized configuration templates; operational configurations stay local.
- `docs/project/`: architecture, reproduction boundaries, results and roadmap.
- `results/summaries/`: small, source-hashed result summaries rather than raw episodes or model weights.

## CPU-only checks

Use a separate Python 3.12 environment. This command does not launch simulation, download weights or contact a robot:

```bash
python3.12 -m venv .venv_cpu
.venv_cpu/bin/python -m pip install -r requirements-cpu.txt
PYTHONPATH=src .venv_cpu/bin/python -m pytest -q   tests/test_hgwind_cohort.py tests/test_hgwind_render_delivery.py   tests/test_hf_archive.py tests/test_prediction_metrics.py tests/test_chunk_execution.py tests/test_publication.py tests/test_trajectory_quality.py tests/test_recorded_franka.py tests/test_hgwind_effort_probe.py tests/test_grasp_evidence.py tests/test_live_kinematics.py tests/test_cartesian_target_trace.py tests/test_cartesian_target_replay.py tests/test_cartesian_packets.py
```

These tests use synthetic fixtures. They verify software contracts, not grasp success. Other tests may require local data, USD assets or Isaac installations.

## Simulation and training setup

The existing simulator uses **Isaac Sim 4.5 / Isaac Lab 2.2.1**. Model/data dependencies are isolated from it. Do not upgrade the simulator to match a baseline's upstream requirements.

Simulation entry points currently retain experiment-specific paths and require separately provisioned RoboPhys assets, saved manifests and local configuration. This repository is a research source release, **not a turnkey distribution of those assets**. Read [reproduction boundaries](docs/project/REPRODUCIBILITY.md) before running any simulation or remote orchestration script.

## Research direction

The planned contribution is an evidence-backed account of physical generalization: separate observation, action execution and contact failures; control changes in physical conditions; and evaluate recovery under explicit disturbances after establishing a reliable baseline. These are research goals, not established novelty claims. [Roadmap](docs/project/ROADMAP.md).

## Attribution and updates

RoboPhys supplies the benchmark scenes, expert and evidence framework. [DynamicVLA](https://github.com/hzxie/DynamicVLA) supplies the pretrained policy baseline and upstream learning tools. See [third-party notice](THIRD_PARTY.md) for revision and licensing boundaries.

Code changes, verified results, limitations and direction changes are maintained together through [the update policy](CONTRIBUTING.md) and [changelog](CHANGELOG.md). Large data and checkpoints are managed separately; this repository contains no access credentials.
