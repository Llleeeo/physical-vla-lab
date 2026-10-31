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
