# Fixed-observation ball sampling diagnostic

## Question and scope

The precision experiment retained the 3,000-step parent. Two existing noise draws differed by about24.25mm in predicted XYZ on the approach window. This does not establish whether sampling variability or persistent prediction bias dominates. The new diagnostic holds the checkpoint, expert RGB/state history, preprocessing, batch size and observation anchors fixed, and compares10 versus30 flow integration steps with paired initial noise.

This is one exposed training episode (`local/traj_00018`). No training, simulator rollout, full-action averaging, sampler promotion or held-out evaluation occurs. Increasing integration steps is a hypothesis; it may increase runtime without improving predictions.

## Experiment

- Parent run: `20261031_133555_985430`, strict weights and exact normalization.
- Anchors:0,40,80,88,96,104,112,120,128,140,144,160; each contributes its next four actions, without padding.
- Eight noise draws per anchor, seed `90000 + 100000*draw + frame`, batch1; same noise for both integration counts.
- Upstream sampling modifies noise in-place. Each call receives a clone. The unchanged original noise is checked after each paired prediction.
- Count actual denoiser calls; require exactly10 or30 and restore both model/policy configs even on exceptions. Reset policy queues per prediction.
- Repeat the first10-step draw at each anchor and compare within `rtol=1e-4, atol=1e-5`. Compare the same prefix against archived parent predictions at the same tolerance.
- Save full20x29 chunks, selected frame indices and synchronized GPU timings after each completed anchor. Report only first-four-action statistics. The first timing includes warmup; no real-time claim.

## Reading the output

For each target offset, average XYZ over the eight draws. Report the empirical mean's RMS distance from the expert target, RMS spread around that mean, and total RMS target error. Using population averaging, `total_error² = empirical_mean_bias² + sampling_spread²`. Compute overall selected-anchor statistics, each anchor separately and the selected frames80..119 subset. This sparse subset is not the prior dense40-frame near-contact mean; RMS also differs from mean Euclidean distance. Do not compare these numbers as interchangeable metrics.

Eight draws estimate variability, not the exact model population mean. The XYZ mean is a diagnostic statistic, not an executable29D command. Quaternion, gripper/mode accuracy and native decoder rejection counts are reported separately. Transition counts are omitted because concatenated disjoint anchors/draws do not form a trajectory. Native decoder acceptance does not establish physical feasibility or grasp success.

Use the paired results to decide whether inference discretization merits further investigation or the mean predicted trajectory itself needs attention. Do not cherry-pick a successful noise draw or promote30 steps automatically.

## Run

```bash
cd /home/robophys/Capstone_Project_Yuan_Haoming
.venv_dynamicvla/bin/python scripts/diagnose_hgwind_sampling.py --run
```

Default invocation performs CPU preflight; `--check-load` additionally strictly loads actual weights and checks normalization on CPU. The GPU command performs192 paired-condition predictions plus12 repeated controls, with no Isaac Sim. Outputs are under the parent run's `sampling_diagnostics/` directory. Source/checkpoint/data hashes are recorded.

Portable tests cover empirical decomposition, invalid data, in-place noise mutation, shared/separate configs, reset behavior, call-count enforcement and exception restoration. Actual CPU model loading and10/30-step sampling are checked locally. GPU results remain pending. Rollback: stop using the new entry point; existing training, live inference, weights, source data and control limits are unchanged.
