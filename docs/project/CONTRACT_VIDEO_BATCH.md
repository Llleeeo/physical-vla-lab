# Bounded paired diagnostics with videos for every saved outcome

**Fresh batch completed:** `traj_00018` native teacher and 29D surrogate pass; the other two native teachers abort on motion feasibility and their surrogates are skipped. Four new saved results have 20 verified review videos; both existing references add ten. See [results](RESULTS.md) and the [source-hashed summary](../../results/summaries/2026-10-29-bounded-contract-batch-result.json). Earlier missing-frame recording errors and the first surrogate retention failure remain preserved. No learned policy was evaluated; human review and early teacher-divergence diagnosis remain pending.

## Cases selected before new outcomes

The existing development source `local/traj_00000` has already been exposed. Among the other 20 local episodes, 9 have symmetric binary 0/0.04 m actuator jaw labels and 11 include intermediate jaw targets unsupported by the frozen binary action contract. Those 11 are recorded as **schema incompatible**, not failed grasps or poor-quality demonstrations. Never infer a binary CLOSE from an intermediate width.

Select first/middle/last schema-supported local episodes ordered by original attempt index:

| Source | Original attempt | Recorded frames |
|---|---:|---:|
| `traj_00001` | 60 | 319 |
| `traj_00007` | 144 | 293 |
| `traj_00018` | 345 | 269 |

The scope screen, selected identities, seeds and hashes are saved before simulations. These are exposed development cases, not reserved evaluation episodes. Results do not estimate success over all 40 demonstrations. The complete scope screen is in the [preparation summary](../../results/summaries/2026-10-29-bounded-contract-video-batch-preparation.json).

## Fixed workflow

1. Render both existing references: passing fixed-gain/latched-posture baseline and failed 29D surrogate.
2. For each selected case, execute the unchanged native expert once with upper-command capture, then render its saved trajectory **regardless of success**.
3. If that native teacher passes physical/initial-state checks, its captured commands align, and its posture/controller fields fit the frozen contract, prepare and execute the same 29D surrogate once. No historical sequence of component ablations is repeated. Render that saved outcome regardless of success.
4. If teacher execution fails or labels are unsupported, retain its video and reason, skip the surrogate and do not replace the case with another successful outcome.
5. Finish with an index of every run, independent grasp result, render state and five RGB MP4 paths. Missing reports do not prevent rendering an otherwise valid saved trajectory. A crash with no saved states is explicitly unrenderable; it is not silently omitted. The coordinator stops immediately on that execution error instead of spending GPU time on the remaining cases.

Maximum: **6 new physics runs**, plus rendering **2 existing runs**. Maximum **8 result sets / 40 RGB videos**. The earlier success/failure collector filenames stay unchanged; independent engineering grasp status is displayed separately. All work is serial on the main machine. No remote worker, training, automatic successful-case replacement or controller tuning is performed.

The native teacher conditions on simulator state to generate supervised labels. The surrogate consumes only its serialized upper actions, fixed controller settings, current robot state and non-target safety geometry. It does not run the online task/skill decision loop. A teacher-conditioned replay is a diagnostic reference, not a learned VLA result.

## Run

```bash
cd /home/robophys/Capstone_Project_Yuan_Haoming

data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_contract_batch.py --run
```

The command first performs CPU scene/source/SDK preflights. It prints `BATCH_OUTPUT` and `VIDEO_REVIEW` paths, then renders the existing passing/failed runs before starting new paired physics cases. This batch can take tens of minutes, with rendering dominating; no exact runtime has been measured for this new coordinator.

To prepare only, omit `--run`: no GPU is launched. Renderer preflight may stage small isolated work copies; raw episodes and task USD are checksum protected.

To resume an interrupted batch:

```bash
data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_contract_batch.py \
  --resume data/derived/hgwind_contract_batch/TIMESTAMP --run
```

Use the exact previously printed timestamp. Completed physics stages, including failures, are not repeated. Interrupted recorded stages are retained rather than automatically rerun; missing/failed videos are retried. A coordinator lock prevents two drivers from owning one batch. Source manifest, contract and task hashes must remain unchanged.

## Human review files

Output root: `data/derived/hgwind_contract_batch/<timestamp>/`.

- `batch.json`: source/schema scope, stage results, paths, errors and per-camera frame verification.
- `VIDEO_REVIEW.md`: regenerated index linking front, left, right, overhead and wrist RGB videos for every actual result.
- `MANUAL_REVIEW.md`: human notes table; it is created once and **never overwritten on resume**.
- Stage-specific physics/render logs: failures remain inspectable even if automatic analysis fails.

Videos live under each actual run's separate `visualization/Trajectory/1s1p/<task>/<saved_episode>/video/` directory, named `<camera>_rgb_vis.mp4`. The coordinator prints the unified review index so users do not have to locate scattered experiment paths.

The existing renderer replays **recorded robot and object states**; it does not execute the expert or policy again. Review videos are **30 Hz**, 640×480, sampling the 60 Hz states at indices 0,2,4,..., with all five configured cameras. Every RGB MP4 is decoded with `ffprobe` and checked against the HDF5 camera frame count, dimensions and FPS; source indices/timestamps are verified too. These are inspection videos, not formal 60 Hz training pixels. Automated checks do not establish ball visibility or plausible motion; human inspection remains necessary.

Check interception, closure depth, contact/relative slip, lift/retention, arm jitter and whether the ball remains visible. Record reasonable/uncertain/unreasonable motion separately from automatic success/failure. Do not relabel an unsuccessful physics trial as successful merely because the video looks plausible.

## Verification and rollback

227 portable CPU checks pass, including failed-episode resolution without a report, ambiguous-path rejection, all five camera streams, truncated MP4/misaligned timestamp rejection, schema-aware deterministic selection, incompatible posture rejection and resume without repeated failed physics. All three actual case preflights and both existing render preflights pass. Direct preparation from the original verified teacher recreates the previous surrogate's 209×29 float32 action array exactly; it does not change the action contract.

The first user batch completed both reference renders, then three native captures stopped on a missing-frame recording callback before saving trajectories. This is not three grasp failures. The frame fix changes recording only; a new batch is required because completed-error physics stages are intentionally not repeated on resume. No GPU rerun was started during the fix. Existing single-case baseline/contract/render commands remain available; reverting this orchestration milestone leaves raw data, prior reports, fixed controller parameters and original RoboPhys unchanged.

## Observed completed batch

Batch `20261029_215343_401088` is finished with all saved runs rendered. The execution after the recording fix produced evaluable states, including both teacher safety aborts. Three selected teachers yielded only one eligible surrogate pair. Do not replace the skipped cases or count them as failed surrogate runs; no aggregate reliability claim follows. Before another GPU batch, inspect the original-versus-fresh teacher divergence at frame 3 in cases 1 and 7 and complete video review.
