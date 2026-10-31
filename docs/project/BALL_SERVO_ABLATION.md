# Conditional translation-objective ablation

## Question and scope

Can a bounded translation-command loss improve the exposed ball demonstration's startup predictions without worsening near-contact pose or jaw behavior? The previous audit found exact label reconstruction and native startup ingress transitions, but substantial nominal translation mismatch. This experiment changes training only. It is neither a new runtime controller nor a demonstrated grasp remedy.

## Matched comparison

Run `scripts/train_hgwind_servo_ablation.py --run` in the existing model environment. Two branches start independently from the retained 3,000-step checkpoint, with fresh identical AdamW settings, 500 steps each, batch size 2 and learning rate 1e-5. Both use the same precomputed observation indices and explicit Gaussian noise/Beta flow times, with hashes compared at each paired step. The visual/text backbone remains frozen. No data augmentation or sampling-protocol change is introduced.

- `uniform`: original valid, unpadded 29-channel flow MSE over the 20-action chunk.
- `servo`: the same flow term plus 0.1 times a Huber loss on the first four valid endpoint translation requests, normalized by 0.1 m/s.

The coefficient and scale are engineering hypotheses, not physically certified tolerances. This is one paired training schedule; eight inference noise seeds do not provide eight independent training experiments.

## Auxiliary computation

For upstream flow `x_t = t * noise + (1-t) * action`, estimate the endpoint as `x_hat = action - t * (predicted_velocity - (noise-action))`, then denormalize once. This endpoint estimate is not an action obtained by full inference-time integration.

Translation starts from `9 * (target_position - measured_pinch_position) + linear_feedforward`. Apply the native absolute 1.7 m/s norm cap, then the native relative-speed ball about the velocity reference. The auxiliary term uses **teacher reference, cap and enable masks** for both predicted and teacher requests. Only predicted XYZ and linear feedforward receive its gradients. This prevents reducing the auxiliary error by inventing a smaller cap, but means it is conditional supervision, not the complete predicted runtime contract. Other action channels, including all masks and jaw, retain ordinary flow supervision.

Each future action is paired with the corresponding future expert measured pinch state, used **only as an offline training target context**. Model observations remain the existing RGB, historical measured robot state and instruction. Padding and offsets 4–19 receive zero auxiliary gradient; ordinary flow still supervises their valid actions. Correct expert discontinuities have zero auxiliary error. No object truth, expert skill step, extra policy input or deployed timing rule is added.

This approximation stops before geometry projection, IK and effort dynamics. Different upper actions may generate the same saturated translation, and radial gradients can vanish under saturation; the original flow objective remains necessary. No change to simulator safety limits is involved.

## Evaluation and retention

Generate full 269-frame first-four-action streams for the parent and both candidates using the same eight noise seeds and 10 integration steps. Check parent predictions against the archived baseline. Report original position/orientation, increment, jaw and native-decoder metrics, plus pre-collision translation errors for startup 0–39, near-contact 80–119 and the full episode.

Unlike the conditional training term, evaluation decodes **all predicted** reference/cap/mask fields using the runtime codec. A rejected command is listed and makes whole-stream translation aggregates unavailable; it is never silently excluded to improve a mean. Compare candidates against both the parent and the equal-budget uniform branch. Save both checkpoints and prediction arrays; select neither automatically. Review regressions before deciding on any physics rollout.

## Verification and rollback

Default invocation performs CPU preflight; `--check-load` also strictly loads the real model and checks forward/backward without CUDA. Verification includes 807 comparisons against the frozen native NumPy servo (original and perturbed commands), finite-difference gradient checks in interior/saturated regions, exact endpoint and padding/tail-gradient checks, and future-state/action alignment.

The standalone script/module and tests can be removed to roll back. Existing source files used by prior experiments, datasets, checkpoints and runtime controller are unchanged. GPU results remain pending.

## Observed result — 31 October

Run `20261031_155803_063005` completes both 500-step branches and all 24 full prediction streams. Both checkpoint inspectors, all source/checkpoint hashes, matched training indices/noise/time, and recomputed per-stream/aggregate metrics pass. The older partial `155643` directory is a separate incomplete run; it is not evidence of failure in this completed run.

Across eight inference seeds, parent/uniform/servo startup nominal translation errors are 0.56375/0.56791/0.56191 m/s; near-contact errors are 0.06591/0.06589/0.06253 m/s; full-episode errors are 0.16496/0.16406/0.15956 m/s. Relative to parent, servo improves near-contact requests by 5.13% and full requests by 3.27%, in all eight pairs. Startup improves only 0.33% on average, with four better and four worse. These are pre-collision requested-velocity errors on expert states, not observed robot tracking.

Near-contact position target error is 20.28/21.96/20.97 mm: servo improves against equal-budget uniform training but is 3.41% worse than parent in every seed. Mean jaw errors remain 4.5 frames and transitions remain 2; native decoder and command rejection counts are zero. Thus the auxiliary has a consistent but small effect on its related command metric, without resolving the startup or jaw issues. No grasp improvement or generalization is established.

Keep both candidate checkpoints and the current default. Next run one bounded 269-interval exploratory servo-candidate rollout with the existing prefix4/legacy camera protocol and unchanged controller limits, without any expert action substitution. Review early contact, closure timing and independent grasp evidence. Earlier live runs had initial-RGB variation; a single new rollout cannot establish a causal improvement over a historical run or a success rate. Do not launch further training merely because the auxiliary metric decreased. [Verified evidence](../../results/summaries/2026-10-31-ball-servo-ablation-result.json).
