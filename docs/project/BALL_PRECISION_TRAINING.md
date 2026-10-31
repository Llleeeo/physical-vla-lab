# Ball prediction precision training

## 31 October 2026 — Precision training completed; parent retained

Run `20261031_144208_499175` completes1,000 additional optimizer steps and samples all269 training frames. All ten saved generated streams reproduce their reported metrics exactly; all four checkpoint selections correctly reject. The selected weights are byte-identical to the3,000-step parent. No new rollout has been run or is justified by an unchanged checkpoint.

For seeds90000/190000, the last candidate reduces approach means61.26→59.23mm /55.97→50.88mm (3.31%/9.10%), near-contact means20.33→18.42mm /21.23→18.26mm, and increment residuals25.36→22.68mm /21.18→18.76mm (10.55%/11.42%). Required approach/increment reductions are10%/20% in both probes. Seed90000 orientation p95 rises3.95→4.07degrees. Jaw errors/transitions are unchanged at step1,000. The conservative engineering screens reject modest improvement; they are not physical laws defining grasp success.

With identical expert observations, the two noise draws differ by24.25mm (parent) and21.58mm (last candidate) on average over frames80..119. This motivates separating inference variability from persistent prediction bias, not claiming randomness alone causes failures. Only two draws were observed. The last candidate's generated arrays remain, but its weights were not saved because it failed selection. Next use the retained parent for a bounded fixed-observation sampling diagnosis before choosing another training or inference change. Source-hashed evidence: `results/summaries/2026-10-31-ball-precision-training-result.json`.


## 31 October 2026 — Precision-oriented training prepared

Saved generated predictions with expert training observations still show20.33mm mean XYZ error and24.95mm mean target step over frames80..119; teacher mean target step is3.84mm. Closed-loop drift is therefore not the only unresolved issue. After XYZ-only postprocessing advanced contact, return to training supervision rather than promoting that filter.

`train_hgwind_precision.py` strictly starts from the selected3,000-step lineage checkpoint and caps additional training at1,000 steps, batch2, learning rate1e-5, frozen backbone, unchanged269-frame episode, controller and29D contract. The separate objective increases normalized flow weights on XYZ and jaw to4 (others1), and adds0.1-weight physical XYZ endpoint and0.1-weight demonstrated-increment Huber residuals, both scaled by10mm. These coefficients are an unvalidated engineering hypothesis, not proven optimal settings. The upstream equation is `x_t=t*noise+(1-t)*action`; its endpoint estimate is `x_t-t*v_t`. A hook observes the existing output projection without editing upstream code and checks its residual against the upstream loss. Endpoint reconstruction during training is not equivalent to a fully sampled inference rollout. Temporal penalties compare predicted versus expert increments, preserving correct rapid motions, and exclude padded endpoints. They act within chunks; full generated-stream probes also measure changes across replans.

Every250 steps, generate the complete stream with prefix4 and two fixed seeds90000/190000. Both probes must improve approach and frames80..119 position means by at least10%, and expert-relative increment error by20%. Full position p95, orientation p95, jaw error count/transition count and all recorded mode accuracies must not regress from the parent; native decoding rejects must be zero. Select the lowest combined near-contact error among passing candidates. If none passes, `checkpoint_final` retains the parent, `selected_checkpoint_step=0`, and no physics rollout is recommended. Fixed seeds are repeated training-case diagnostics, not independent validation data; results do not establish generalization. Selection thresholds are screening rules, not task-success thresholds.

Actual CPU strict load, normalizer identity, output-hook/upstream residual equivalence and backward gradient checks pass without CUDA. Synthetic gradient checks exclude padding and padded dimensions and give zero increment penalty for a correct expert discontinuity. Portable metric tests guard both seeds and prevent pose/jaw regressions from hiding behind position improvements. GPU training remains user-run and pending. Existing checkpoints/source training files are preserved; rollback is to use the parent run and stop the new entry point. Full report metadata includes projection delta so the existing checkpoint inspector can reload the selected result.

Run:

```bash
.venv_dynamicvla/bin/python scripts/train_hgwind_precision.py --run
```

The command trains and performs generated-action probes only; it does not start Isaac Sim. It saves predictions at every probe, logs components separately from the original29D flow MSE, and records selected versus total optimizer steps. Completion is `BALL_PRECISION_TRAINING_COMPLETED`. Review `selected_step` and both probe tables before any live rollout.

Release verification:433 portable tests pass in a clean tracked export with PYTHONPATH=src. The model environment separately passed an actual CPU forward/backward with nonzero finite projection gradients and exact normalizer reload. No CUDA or simulation was started by these checks.
