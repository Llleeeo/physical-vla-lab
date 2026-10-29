# Bounded original worker-history diagnostic

The original attempts58,59,60 occurred in one worker. Their saved lengths are720,363,319 control frames; the first two failed grasping without a safety abort and the third succeeded. A fresh-worker attempt60 instead differs in ball velocity after the first physical step and eventually aborts on motion feasibility. This experiment tests whether reproducing the original within-worker history changes that mismatch. It does not identify a unique reset/cache/force mechanism or evaluate a learned policy.

## Current environment blocker

The matched fresh60 launch aborts before any attempt with the same Client stack. The stage-load barrier does not reliably avoid startup failure. Inotify resource exhaustion has now been independently measured; see [startup resources](NATIVE_STARTUP_RESOURCES.md) before another simulation. Local VS Code exclusions await window reload and the coordinator now checks watcher headroom before SDK startup.

## Observed reproduction and one matched-startup control (30 October)

User run `20261029_234931_549595` completes all three requested attempts in one process. A separate CPU reread verifies original file hashes and repeats the full comparison:

| Attempt | Original recording | New result | Control frames | Exact full-length action/observation arrays |
|---|---|---|---|---|
|58|failed_00057|fail|720|15/15|
|59|failed_00058|fail|363|15/15|
|60|traj_00001|pass|319|15/15|

The first seven native ball-velocity rows also match each original exactly. Attempt60 maintains bilateral contact for2.4833s and at least60.349mm final lift, passing independent engineering grasp checks with no assistance. All three initial-state and runtime-safety checks pass. These selected outcomes are a reproduction check, not a collection success-rate estimate or learned-policy evaluation. [Hashed verified results](../../results/summaries/2026-10-30-worker-history-reproduction.json).

The stage barrier completes two Kit updates in24.545s with zero timeline advancement and no script instances. No native crash recurs in this launch; general crash avoidance remains unproven. Recorded force calls before control-frame recording are4/3/3 for attempts58/59/60, versus4 in the earlier fresh60 control; equal clocks/call counts do not establish force accumulation or its causal effect.

Because both worker history and stage-attach scheduling differ from the earlier fresh control, run **one** fresh60 using exactly the same guarded startup and native collector assembly as this successful history launch:

```bash
cd /home/robophys/Capstone_Project_Yuan_Haoming

data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_history_diagnostic.py \
  --fresh-control --stage-load-barrier --native-backtrace --run
```

`--fresh-control` is explicit, requires the barrier and the verified history reference, and selects only attempt60 with one attempt allowed. It neither substitutes a new seed nor runs attempts58/59. Preflight checks exact native collector assembly and pinned scene/runtime dependencies against the history launch. Reports compare the new result to its original, the old fresh60 and history60. Rendering remains skipped. This control is CPU validated (263 portable checks,20 focused history checks, real scene and actual native-entry assembly); simulator execution is pending.

If it matches the earlier fresh failure while the history prefix matches the originals, that strengthens evidence for a within-worker state/history effect. If it instead reproduces the original success, guarded startup may explain some previous discrepancy; do not attribute the result solely to history. Partial or different results require inspecting their raw startup arrays before a mechanism claim. Omit `--fresh-control` for the existing three-attempt diagnostic.

## Current blocker and optional startup workaround

The user run `20261029_232920_958901` aborts after `open_stage()` returns, before attempt58 starts. GDB locates the exception in `libomniclient.so`, on its `OC Tick` thread. Disassembly of the matching installed library places the throw after a failed `pthread_mutex_lock`, on an internal callback path. The mutex/request ownership and triggering file remain unknown; this does not establish a disk, network, physics or controller cause. No history or grasp outcome was evaluated.

A CPU-only check of the same installed client2.55.0 succeeds for100 local stat/subscription/callback/stop cycles, with Hub disabled for that isolation check. It starts no SimulationApp and cannot validate Kit attachment. [Hashed observations and limitations](../../results/summaries/2026-10-29-startup-client-abort.json).

The frozen collector immediately prepares robot USD after `open_stage()` returns. Other code in that same collector acknowledges that Kit attachment may still be pending. An **opt-in, unproven workaround** waits for attach completion before these robot edits. It holds both the existing script-load barrier and pending script-application calls, requires a stopped and unchanged timeline and no script instances, and restores the native method on success/error/timeout. It never calls Play or a physics-step method. The wait is bounded by1200 Kit updates and a60s wall budget checked between updates; one blocked native update can still exceed that budget. Intermediate evidence is written atomically to `stage_load_barrier.json`.

```bash
cd /home/robophys/Capstone_Project_Yuan_Haoming

data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_history_diagnostic.py \
  --stage-load-barrier --native-backtrace --run
```

This changes startup scheduling and must be identified as a modified-startup comparison. Reaching `STAGE_LOAD_BARRIER ready` only confirms its guards; it does not prove the intermittent crash is fixed. A complete history still requires all three original attempts, source/initial-state comparisons and independent physical evidence. Missing/failed barrier evidence blocks diagnostic completion. Existing source files, physics values and control profile are preserved. Rendering remains off by default. Omit the option to recover the original collector assembly, whose SHA256 is verified identical to the previous run.

260 portable CPU checks, including17 focused history checks, actual scene/source preflight, Python3.10 parsing and actual native-entry assembly checks pass. No new simulator execution was launched by the agent. Proposed GPU execution is pending.

## Run once

```bash
cd /home/robophys/Capstone_Project_Yuan_Haoming

data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_history_diagnostic.py --native-backtrace --run
```

Without `--run`, only CPU preflight executes. Exactly one native worker receives `--attempt_offset58`, `--max_attempts3` and no success-count target. Original seeds, variation, smooth-lift profile,2mm depth, physical jaws, effort execution,12s budget, reset code and safety exits remain. A safety abort or planned native restart ends that worker; the coordinator never replaces the missing attempts with fresh processes. GDB is the existing optional child debugger used in the measured fresh control; its previous success does not establish a native-crash fix.

The independent entry restores and rotates the passive force and Cartesian observers at each original attempt boundary. Control/force calls remain native. Per-attempt labels, command traces and startup-force records cannot carry into the next recording. Initial loading updates remain outside complete force-observer coverage. The same patched source is checked at preflight and actual native entry before execution.

Original failed/successful recordings are checksum pinned by attempt index. Each saved result is independently analyzed against its own original, with decoded action/state and raw180Hz ball velocity comparison. Attempt60 also compares against the already saved fresh-worker failure. Original force-call history is not available. Matching episode seeds and sampled variation are checked before interpreting the result.

## Output and interpretation

A fresh directory under `data/derived/hgwind_history_diagnostics/<timestamp>/` contains the plan, single-worker log, observed attempt order, individual attempt folders and final `report.json`. Physics outcomes and delivery completion are distinct. A partial prefix is explicitly incomplete; no episode is counted as tested when it did not run.

Rendering is skipped by default at the user’s request. Raw trajectories and analysis are retained; missing videos are marked skipped and do not block an otherwise complete diagnostic. To generate five30Hz saved-state review videos later, explicitly add `--render` to the delivery command below. These are not formal training renders; human visual review remains pending. Render preparation uses copies, preserves raw collector files, and reports analysis/render errors independently.

To render a completed diagnostic later without recollecting:

```bash
data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_history_diagnostic.py --finish <HISTORY_OUTPUT-directory> --render
```

Interpret the three raw comparisons before drawing conclusions. If earlier attempts already differ from their originals, the original history was not fully reproduced. If the prefix matches and attempt60 recovers, that supports a process-history effect in this exposed sequence; it does not distinguish fan caches, force accumulators or other solver/reset state. A different grasp outcome alone is insufficient. The frozen29D policy contract and previous failures remain unchanged.

## Validation and rollback

251 portable CPU checks pass, including eight new checks for strict sequence, observer restoration, independent per-attempt traces, partial-prefix retention, body-path velocity matching and all-saved-outcome render delivery even when analysis fails. Actual scene/three-source preflight passes, original HDF5 readers work, and the actual native-entry patch assembly matches its expected hash. Simulator-side code parses as Python3.10. These are preparation checks; the three-attempt GPU run has not been launched by the agent.

Remove the new history module, worker, runner and tests, or revert the preparation commit. Existing single-attempt tools, frozen collector/control and original trajectories are preserved.
