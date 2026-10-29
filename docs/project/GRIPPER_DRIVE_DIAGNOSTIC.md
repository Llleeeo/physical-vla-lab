# Passive gripper-drive comparison

## Why this measurement

The [wind-call intervention](SETUP_FORCE_CONTROL.md) restores ball pose/velocity through native rows0..500. Its actual fingers still close more slowly despite matching recorded position targets: at frame166 they are2.192/2.490mm farther open, versus only0.0886mm EE displacement. First positive finger/ball contact is about89ms later and grasp fails. Saved targets alone cannot explain the live drive response.

The frozen `Expert.ensure_gripper_force` writes runtime gains/caps and some USD drive attributes. This made reset/configuration history worth measuring. The completed result below now establishes an actual native gain difference.

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

## Observed result 30 October

The pair `20261030_015645_885994` is valid. All15 core arrays in history58/59/60 match their originals; fresh60+omit1 matches the previous uninstrumented omission failure exactly. Source hashes and full-array comparisons were independently rechecked after the user run.

| Native property at first close (frame154) | Successful history60 | Fresh60+omit1 |
|---|---|---|
| Finger1/finger2 stiffness | 10500 /10500 | 10000 /0 |
| Finger1/finger2 damping | 140 /140 | 1000 /0 |
| Finger1/finger2 force ceiling | 70 /70 N | 70 /70 N |
| Finger1/finger2 speed ceiling | .05 /.05 m/s | .05 /.05 m/s |

At frame0 **before configuration**, both native gain arrays are still10000/0 and1000/0. The history force ceiling is70/70N while fresh is7.2/0N; these caps become equal after configuration. Do not attribute closure failure to a7.2N closure cap.

All attempts call `ensure_gripper_force(70, speed=.05, reconfigure=True)` at frame0. First episodes (history58 and fresh60) return true and record10500/140 in bookkeeping, but live gains remain old. Later history59/60 genuinely read back10500/140 on both. At closure the compared native applied targets and USD/mimic snapshots match across target60 conditions. This supports initialization-history dependence, while gain correction efficacy is still untested.

Code evidence: frozen `Expert.ensure_gripper_force` sets controller gains and caps, then authors USD force ceilings, applying a missing second-finger DriveAPI in the first episode. Native overwrite due to that late USD operation is a hypothesis consistent with the order and observations; no internal setter trace establishes that exact mechanism. Importantly, warm gains actively drive both DOFs while the original mimic constraint remains. Reproducing this observed configuration is not confirmation that two active drives are the right production model.

[Compact measurement and provenance](../../results/summaries/2026-10-30-gripper-drive-result.json).

## One explicit gain-reapplication control

The smallest next experiment runs fresh60 once with its already-verified force omission. After the original frame0 force setup, target only finger gains through the native view with `save_to_usd=False`, using10500/140 for each finger. Preserve original force/speed limits, position/velocity targets, mimic constraint, arm gains, measured state and authored USD properties. No added physics step or source-stage save occurs.

The command is deliberately restricted to this observed case. It requires the pinned measurement hashes, exact native ensure arguments/result and expected live values; unexpected or missing readbacks stop the diagnostic. After the one setter, frame1 and selected later frames check persistence. The setter audit is separate from grasp success and does not require the old closure time: changed closure timing can be an experimental effect, not invalid initialization evidence. Known unsupported extra initialization calls stop rather than silently add setters.

```bash
cd /home/robophys/Capstone_Project_Yuan_Haoming

data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_history_diagnostic.py \
  --fresh-control --stage-load-barrier \
  --omit-setup-force-once --trace-gripper-drives \
  --restore-gripper-gains --native-backtrace --run
```

One attempt, no rendering/retries; typical recent runs suggest about1–2minutes including startup. CPU-only preflight omits `--run`. Results go to a new `hgwind_history_diagnostics/<timestamp>_fresh60_omit1_gains` directory, including raw trajectory, `report.json`, passive drive trace, force audit and `gripper_gain_control.json`.

Inspect `GRIPPER_GAIN_CONTROL verified=... grasp=...`. Verification means the gain intervention was actually observed; it does not imply successful grasp. An early failure can still establish intervention validity if native readbacks/persistence are available. If it succeeds, deterministic initialization and mimic-consistent modeling still require a separate decision before training eligibility or new collection.

Preparation:314 portable CPU checks, SDK Python3.10 parsing, actual scene/hash preflight, exact unchanged collector assembly, default-entry isolation and per-case dependency pins. Physical efficacy pending. Original RoboPhys, frozen bundles and raw data were preserved. Roll back by omitting `--restore-gripper-gains`; the new module/worker and opt-in runner branch can be reverted independently of earlier diagnostics.

## Gain-reapplication result 30 October

The user-run fresh60 control now passes: `GRIPPER_GAIN_CONTROL verified=true grasp=pass`; both force/gain audits verify. It completes319intervals and957physics callbacks with no safety abort. Independent grasp evidence gives2.483s bilateral contact and at least60.349mm final lift, without assistance.

The saved report is independently checked against original raw data: all15 complete core arrays, plus all958 native ball transform/velocity samples match exactly. Recomputed contact/lift checks equal the reported checks. The omission-only190frame failure remains preserved. Conditional on the existing omission, the measured gain reapplication restores original expert motion for this exposed case.

This is not a learned VLA result, a held-out reliability estimate, or certification of both active finger drives alongside mimic coupling. The native internal overwrite cause remains inferred. Before any29D surrogate run, explicitly port the verified initialization into its entry point: copied configuration fields alone do not execute hooks. No further GPU work or rendering was launched for this analysis.

[Result and provenance](../../results/summaries/2026-10-30-gripper-gain-reapplication-result.json).
