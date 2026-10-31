# Full-episode noise replanning diagnostic

## Purpose

The matched10/30-step probe found more solver steps slower and less accurate on the selected expert observations. Keep10 steps. Test whether drawing new Gaussian noise at every four-frame replan contributes to discontinuous targets. This is an offline diagnostic on one exposed269-frame training episode, not a grasp or generalization test.

## Frozen comparison

Strictly load the selected3,000-step parent `20261031_133555_985430`, its exact normalizers and existing RGB/state preprocessing. Update expert observations at all68 replans. Use batch1,20-action chunks, first-four execution prefixes and10 denoising steps for both conditions. Consume only one action at the final anchor268, excluding padding.

Eight base seeds are fixed in advance:90000,190000,290000,390000,490000,590000,690000,790000. For each seed:

- **Independent:** initial Gaussian noise seeded with `base + frame`, matching the previous pipeline.
- **Shared:** Gaussian noise seeded with `base` at every replan. The same relative chunk slots are reused; the noise is not shifted with absolute time, and previous predicted actions are not reused.

All eight pairs are reported. No best-seed selection or automatic deployment. Shared noise retains a sampled bias and can produce relative-slot periodic effects; boundary/within-chunk metrics make these visible. Marginal Gaussian sampling is unchanged, while temporal correlation changes. This does not prove that correlated samples improve feedback control.

The sampler clones noise because upstream updates it in-place, checks actual10-step calls, resets policy queues per prediction and restores runtime integration settings. Both modes must match at frame0. Independent-mode full chunks must reproduce every saved anchor/draw in the previous sampling diagnostic; base90000's full consumed stream must reproduce the archived parent stream within `rtol=1e-4, atol=1e-5`.

## Metrics

Report all per-seed full-stream metrics and paired changes: approach mean, dense frames80..119 mean, full position p95, orientation p95, jaw errors/transitions, control-mode accuracy and native decoder rejects. Report startup0..39 and first-four position errors separately.

For both the complete episode and near-contact interval, distinguish increments ending at a replan boundary (`frame % 4 == 0`) from within-chunk increments. Record predicted step magnitude, expert step magnitude, and the norm of their vector difference. An exact rapid expert motion therefore has zero residual; simply reducing movement is not a pass. Near-contact increments end at frames80..119 and include79→80. Record the first predicted closure plus all falsely closed/open frames.

Check smoothness and accuracy together. Reduced jumps with worse trajectory tracking, orientation, mode flags or jaw timing do not support promotion. These are command errors, not measured robot motion. Feedback deviations, physical contact and real-time throughput are not tested. No smoothing, averaging, expert action substitution, retraining or control-limit edits occur.

## Run

```bash
cd /home/robophys/Capstone_Project_Yuan_Haoming
.venv_dynamicvla/bin/python scripts/diagnose_hgwind_noise_replanning.py --run
```

Default invocation performs CPU data/provenance/padding checks. `--check-load` strictly loads actual weights on CPU and checks exact normalizers. `--run` makes1,088 GPU predictions, estimated about2–3 minutes from the previous measured89.9ms per call plus loading and data overhead; GPU contention can increase duration. It does not launch Isaac Sim.

Outputs: parent run's `noise_replanning_diagnostics/<timestamp>/`, including source-hashed configuration, full chunks/latencies saved after each complete seed pair, partial metrics and final report. GPU results are pending. Portable tests cover seed pairing, padding removal, invalid inputs, boundary accounting, demonstrated increments and early jaw closure. Rollback: stop using this isolated entry point; existing live and training commands remain unchanged.

## Observed result — 31 October

All eight seed pairs complete (16 full269-frame expert-input streams). Source/config/array hashes verify and every stream metric recomputes exactly in the original model environment. Prior generated references match with maximum difference0.

Across seeds, independent→shared near-contact position mean is20.28→21.57mm (3 improve,5 worsen); expert-relative increment mean is21.10→20.90mm, only0.95% lower on average. Boundary increment residual is23.89→23.34mm; within-chunk residual is20.17→20.09mm. Mean per-run full position p95 rises161.36→167.42mm, jaw errors4.50→4.625frames and jaw transitions2.0→2.5. Orientation p95 improves3.73→3.42degrees on average but worsens in3 pairs. Native decoder rejection is zero throughout. Startup0..39 position mean is141.62→144.50mm. These are predicted-command statistics, not robot motion.

Do not adopt shared noise or select a favorable seed for a success claim. This particular correlation scheme is not a reliable fix. The evidence does not rule out other sampling methods or prove the cause of physical failure; stop extending this sampling sweep and retain the original10-step independent-noise behavior.

A read-only supervision audit confirms uniform observation-anchor sampling with20 future targets, while live execution consumes four before replanning. Over one pass through all269 anchors, the first four relative offsets contribute1,070 of5,190 valid label occurrences (20.62%). Absolute action0 appears once, action19 twenty times. These are structural label counts, not exact optimizer-gradient weights, and long-horizon supervision is intentional rather than a code defect. Next proposed bounded training ablation emphasizes the executed prefix while preserving longer-horizon learning, and audits startup/near-contact, jaw and pose performance together. It is a hypothesis, not an established remedy; no new training or physics is launched here. [Source-hashed evidence](../../results/summaries/2026-10-31-ball-noise-replanning-result.json).
