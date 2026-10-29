# Passive gripper-drive comparison

## Why this measurement

The [wind-call intervention](SETUP_FORCE_CONTROL.md) restores ball pose/velocity through native rows0..500. Its actual fingers still close more slowly despite matching recorded position targets: at frame166 they are2.192/2.490mm farther open, versus only0.0886mm EE displacement. First positive finger/ball contact is about89ms later and grasp fails. Saved targets alone cannot explain the live drive response.

The frozen `Expert.ensure_gripper_force` writes runtime gains/caps and some USD drive attributes. This makes reset/configuration history worth measuring; no difference in actual gain values has yet been established.

## Bounded protocol

One command runs two serial workers, with at most four attempts:

1. Original history58→59→60, with passive observations. The first two original outcomes are failed grasps; preserving their full duration retains the verified process history. Before continuing, require the original motion arrays and valid target60 drive observations.
2. Fresh60 with the already verified single setup-force omission and the same passive observer. Compare its motion against the earlier uninstrumented omission run.

Both keep scene, controller configuration, native commands, limits and SDK versions. The second condition retains its explicit modified-force provenance. No render or training runs. No automatic retries are allowed. If observation changes the known motion, the result is flagged as a confound rather than a certified parameter explanation.

```bash
cd /home/robophys/Capstone_Project_Yuan_Haoming

data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_gripper_diagnostic.py --run
```

No `--run` means CPU preflight only. Typical prior wall times suggest roughly6–10minutes for both workers, with startup variability; this is an estimate, not a measured new-protocol duration. Individual results stay under history diagnostics; the pair manifest/logs/report are under `data/derived/hgwind_gripper_diagnostics/<timestamp>`. Reanalyze finished jobs with `--finish <pair_directory>` without `--run`.

## What is read

- USD finger drive and mimic properties before each reset.
- Valid live-physics gain, force-limit and velocity-limit readbacks at frame0, selected preclosure/closure frames, and the first close-action boundary.
- Actual joint positions/velocities and native applied position/velocity targets. Explicit actuation-force entries are not the resulting implicit position-drive force.
- Properties before/after existing `ensure_gripper_force` calls; the original calls, arguments, return values and exceptions are preserved.
- The sampled expert configuration and drive bookkeeping. Native gain units must be interpreted with the authored drive type; USD attributes remain raw authored values.

Missing/nonfinite live readbacks are recorded as errors, never zero gains. An invalid native handle does not fall back to USD as if it were live data. Observation is bounded; hitting its event cap invalidates coverage without suppressing native calls. There are no added simulation updates or native setters in the observer. New Python wrappers are local to the opt-in worker and are restored when episodes rotate.

## Interpretation and rollback

Require `GRIPPER_DIAGNOSTIC comparison_ready=true` before attributing measured property differences to the previously observed pair. Compare actual native gains/limits/targets with the authored USD values and their changes across reset/closure. If those values match, hidden mimic/solver state and other unmeasured conditions remain possible. A parameter difference alone does not establish a successful correction.

The observer is off without `--trace-gripper-drives`. Original RoboPhys/frozen source and data remain unchanged. Reverting the new observer/entry/coordinator restores the prior integration.

Preparation verification:300 portable CPU checks; actual paired scene preflight; matching native collector assembly across enabled/disabled paths; SDK Python3.10 parsing. No simulator measurement has been run by the agent. [Source hashes](../../results/summaries/2026-10-30-gripper-drive-diagnostic-prepared.json).
