# Executed-prefix loss ablation

## Hypothesis and control

Shared-noise replanning did not reliably improve the complete expert-input stream. Return to supervision rather than extending sampling sweeps. The model predicts20 future actions but the controller consumes four before replanning. Test whether emphasizing these executed offsets improves generated targets. This mismatch is not a code bug or established cause of physical failure. The experiment does not rebalance episode-start sampling, change labels or add demonstrations.

Two sequential branches start from the exact selected3,000-step parent `20261031_133555_985430`:

| Setting | Uniform control | Prefix4 treatment |
|---|---:|---:|
| Additional optimizer steps | 500 | 500 |
| First four relative action weights | 1 | 4 |
| Later16 weights | 1 | 1 |
| Batch / learning rate | 2 /1e-5 | 2 /1e-5 |

Loss is upstream squared flow residual over29 valid action channels, averaged using temporal weights and excluding padded targets. Denominator is29 times the valid weight sum. In a full chunk, the prefix therefore receives half the total temporal weight, versus one fifth in the control. This is a bounded engineering choice, not a tuned optimum. No channel-specific XYZ/jaw weights, endpoint penalties or temporal smoothing losses are added. Weight1 numerically matches the existing masked loss. Normalizers and target representation remain exact.

Both branches restart AdamW with the same settings, identical precomputed observation indices and step seed `310000+step`. Gaussian noise and Beta flow times are explicitly supplied; their hashes must match at every paired step. The frozen vision/language backbone stays in eval mode. Fresh parent reloads and normalization checks separate the branches. Per-step RNG seeding also matches stochastic forward operations; GPU numerical determinism is not universally guaranteed.

## Evaluation and records

Before training, regenerate all269 actions for the eight predeclared noise seeds90000 through790000 (increments100000) and require full-chunk agreement with the prior independent-noise audit. Each final branch uses the same eight seeds,10 denoising steps, batch1, four-action prefixes and expert RGB/state observations. No held-out set is accessed and no favorable seed is selected.

Save both final candidate checkpoints, even if neither improves. No intermediate checkpoint selection, default model update or automatic simulator launch. Each candidate has reload-compatible source/normalizer metadata, training logs, checkpoint hash and actual projection delta relative to official weights. Complete metrics include startup, approach and near-contact position error, full position p95, orientation, boundary/within-chunk increment error, control-mode accuracy, jaw errors/transitions and native decoder rejects. Parent/default remains unchanged pending review.

The primary comparison is prefix4 versus the matched uniform continuation, with both also compared to the parent. Improvement over parent alone cannot establish that temporal weighting helped. Check all eight pairs and pose/jaw/mode regressions together; smoothness alone is insufficient. Repeated noise seeds from one exposed training episode do not establish data generalization, training-seed robustness or physical grasp success.

## Run

```bash
cd /home/robophys/Capstone_Project_Yuan_Haoming
.venv_dynamicvla/bin/python scripts/train_hgwind_prefix_ablation.py --run
```

Default invocation checks provenance, loss masks and the existing parent. `--check-load` additionally checks actual CPU forward/backward, exact weight1 equivalence and finite action-head gradients. `--run` trains the two branches sequentially on one GPU and evaluates three models; estimated4–6 minutes from prior measurements, depending on contention. No Isaac Sim.

Outputs: dataset's `prefix_ablation_runs/<timestamp>/`, with root `protocol.json`/`report.json`, parent predictions and `uniform/` plus `prefix4/` branches. Final branches each have `checkpoint_final/`, a complete training log and generated predictions. GPU results remain pending. Rollback: continue using the parent and remove the isolated new files if desired; old training and live scripts are unchanged.

## Observed result — 31 October

Both500-step branches complete; each samples262 distinct observation anchors. Scheduled indices and explicit training noise/time hashes match at every paired step. Both saved checkpoint inspectors pass, source/protocol/checkpoint hashes verify, and all24 generated stream metrics plus paired aggregates recompute exactly.

Across eight evaluation seeds, parent/uniform/prefix4 near-contact position means are20.28/21.96/21.42mm. Prefix4 improves versus uniform in all eight pairs by0.54mm on average (2.47%), but remains worse than the retained parent. Approach means are58.62/59.20/58.94mm. Increment residual means are21.10/20.29/20.50mm: prefix4 is worse than uniform in all eight pairs. Jaw errors average4.50/4.50/4.25frames and jaw transitions2.0/2.0/1.75; all native decoder rejects remain zero. Mean per-run position p95 is161.36/161.50/160.76mm, with prefix4 p95 worse than uniform in five pairs despite the slightly lower mean. This is a modest tradeoff, not a robust improvement over the existing model.

Startup0..39 position errors remain142.17mm (uniform) and142.21mm (prefix4), with first-four errors225.58/223.77mm. Do not promote either branch or launch physics solely because one statistic improves. Both final candidates remain available; the default3,000-step parent stays unchanged. One training schedule/eight inference seeds do not establish generalization.

A read-only label inspection finds target height0.47594→0.06686m from frame0 to1, while measured pinch height changes0.47796→0.47729m. The first target Z is about8.38 normalization standard deviations; frame0 was actually sampled five times in each training branch. This is an upper-goal discontinuity, not a40.9cm physical arm step or proof of corrupted labels. Next audit source upper-goal/feedback generation around startup and target switches, including time binding and control mode/avoidance context, before another loss-coefficient or step-count sweep. Do not delete or smooth these labels without establishing their meaning. [Source-hashed result](../../results/summaries/2026-10-31-ball-prefix-ablation-result.json).
