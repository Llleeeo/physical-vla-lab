# Ball development training bridge

Status: 31 October 2026 — matched 2,000-step training and offline predictions verified; startup target prediction and closure timing remain unresolved. Learned grasp is unverified.

## Observed 2,000-step result

The matched 2,000-step fit reduces fixed-probe flow MSE from the 200-step value 2.0003 to 0.3066. Only the step budget changes; the first 200 sampled batches, losses and gradient norms match exactly. Strict CPU checkpoint reload passes. Offline first-action mean position error improves 127.05→46.42 mm, p95 290.69→187.17 mm, mean orientation error 6.19→2.02 degrees, and gripper accuracy 79.18→97.03%. Thresholded first-action jaw transitions drop 78→1, but closure is frame136 versus expert144, 133.3ms early. Prefix4 still has five transitions; this is not a guarantee for queued execution.

Errors are concentrated early: first30-frame mean163.02mm, frames30–143 mean45.06mm, closure-to-contact mean25.46mm, and contact-onward mean18.70mm. All25 worst frames occur within frames0–35. Captured expert upper targets themselves contain12 consecutive-frame position jumps over100mm, all before frame42; the measured robot pinch path is smoother. These are high-level targets, not actual robot displacement. The jumps are present in the teacher capture and do not alone prove a label defect, cause of model error, or poor source grasp.

All source/archive hashes and raw prefix metrics were rechecked; the adapted pretrained baseline reproduces prior metrics exactly. Motion-reference mean errors improve over200 steps, but tail errors do not uniformly improve. No labels were smoothed, removed or replaced. This is one exposed training episode with one fixed noise realization per observation, not held-out success or learned closed-loop control.

**Next:** audit startup target generation and downstream control response before further training or action-label changes, then prepare bounded live-feedback integration. Do not immediately increase steps again or discard the startup frames. Preserve current controller and safety settings. [Source-hashed comparison](../../results/summaries/2026-10-31-ball-2000step-prediction-result.json).

## Verified offline result and bounded next experiment

Historical 200-step result and experiment plan; the completed 2,000-step result is above.

Raw prediction archive, input/checkpoint/source hashes and all prefix metrics were independently rechecked. The 200-step model's first-action mean position error is 127.05 mm (p95 290.69 mm), down from 495.93 mm under the adapted pretrained baseline; mean orientation error is 6.19 degrees. These are expert-target prediction errors, not measured robot tracking errors. Before closure, between closure and first teacher contact, and from contact onward, mean position errors are 129.63, 134.58 and 122.30 mm respectively. The error is not confined to contact.

First-action gripper accuracy is 79.18%. Thresholded predictions across recorded observations change state 78 times; the target changes once at frame 144. During frames 144–161, gripper accuracy is only 38.89%. These are offline independently sampled outputs, not 78 executed physical jaw movements. Velocity and acceleration mean prediction errors also worsen relative to the adapted pretrained baseline. Zero decoder rejections means structurally decodable outputs, not successful or feasible control. [Verified result](../../results/summaries/2026-10-31-ball-offline-prediction-result.json).

**Next: test insufficient optimization with one bounded 2,000-step run.** Keep episode, seed, learning rate, batch size, model, normalization and action interface fixed. The existing entry starts from the same official pretrained weights; it does not resume or overwrite the 200-step checkpoint. CPU preflight passes. No improvement is claimed before this run and its matched action prediction finish.

```bash
cd /home/robophys/Capstone_Project_Yuan_Haoming
.venv_dynamicvla/bin/python scripts/finetune_hgwind_development.py --steps 2000 --run
```

After completion, use `scripts/predict_hgwind_development.py --run-dir NEW_TRAINING_OUTPUT --run`, substituting the exact new training directory printed by this run. The prediction entry's default still points to the original 200-step checkpoint; do not inadvertently evaluate that again. Compare identical noise and prefixes, position mean/p95, orientation, closure-window gripper errors/transitions, mode accuracy and motion-reference errors. Reduced flow loss alone is insufficient. If this bounded increase does not meaningfully improve generated actions, inspect action representation/objective rather than repeatedly increasing steps. No temporal smoothing or expert-label substitution is applied to hide prediction errors.

This deliberately exposed single-episode fit is a pipeline check. It does not measure unseen-task generalization or establish that more training data is unnecessary.

## Observed fine-tuning result and next command

Historical command: this 200-step prediction has completed; use the new experiment instructions above.

The user run completes 200 optimizer steps, sampling 198 distinct anchor frames. Six probes using the same observations, noise and flow time improve from mean 4.8705 to 2.0003 (58.93% reduction); every paired probe improves. This is training-episode flow loss, not physical action error. The last stochastic training loss of 3.4736 uses different sampled frames/noise from earlier steps and alone is not evidence of regression.

The saved full checkpoint strictly reloads on CPU. Normalizers exactly match the source data, and the saved action projection update matches the training report. No learned simulation rollout has occurred. [Hashed result](../../results/summaries/2026-10-31-ball-development-finetune-result.json).

Next, generate actual 20-action chunks from all 269 recorded observations, comparing original pretrained and fine-tuned weights with the **same new normalizers and per-frame noise**:

```bash
cd /home/robophys/Capstone_Project_Yuan_Haoming
.venv_dynamicvla/bin/python scripts/predict_hgwind_development.py --run
```

This command performs inference only, with no training or simulation. It saves raw absolute29D chunks and reports position/quaternion errors, velocity/acceleration/reference errors, nullable/mode/gripper accuracy, scalar clipping and decoder rejection. Quaternion signs are equivalent; invalid/degenerate outputs are counted, not silently scored as valid. Offline prefixes 1, 4 and 20 use the recorded observation at each chunk anchor; they do not simulate feedback. Decoder acceptance is not physical feasibility. These differently defined29D channels make the official baseline an adaptation diagnostic, not an unchanged zero-shot benchmark.

`--check-load` performs strict CPU reload without inference; the default is CPU preflight. Prediction output is a fresh `offline_predictions/<timestamp>/` directory inside the fixed training run. No checkpoint, source data or controller is modified. Disable the new prediction entry to roll back.

## Immediate objective

Prioritize one complete data → pretrained fine-tuning → prediction → simulator loop using the already passing `local/traj_00018` development case. Its teacher and float32 29D action surrogate both complete269 intervals safely and pass independent grasp checks. This is a deliberately selected passing engineering case, not an unbiased evaluation sample. Keep `traj_00001` contact sensitivity and safety-criterion review as known unresolved work; no controller or safety change is made here.

The earlier requirement to resolve all sampled surrogate failures before any training is superseded **for this single-episode integration check only**. Production cohort readiness and held-out evaluation remain separate gates. A learned policy has not yet executed through the ball controller.

## Implemented data path

`prepare_hgwind_development.py` verifies the source-hashed batch reports, teacher capture and passing surrogate. All15 source trajectory arrays match the regenerated teacher and the staged rendered replay. Reconstruct all269 upper-command labels and require bit-exact equality to the successful float32 action file. Existing 60Hz original-observation RGB is read directly from its HDF5; this adapter is not a LeRobot export and does not use the 30Hz review videos. No rendering is needed for this step.

- Policy observations: `opst_cam` and `wrist_cam` RGB, measured robot state, verbatim source instruction.
- Robot state30: arm q7, arm qdot7, world pinch position3, quaternion wxyz4, initial live-posture anchor7, finger q2. EE position is converted to pinch position using the saved validated robot calibration (approximately12.3mm local-z offset). Do not silently call the saved EE point the pinch point. This fixed-case calibration must be reproduced by the future live adapter.
- Image history: frames `[k-2,k]`, clamp/pad only at the episode start. Upstream consumes both images but only the latest robot-state vector.
- Labels:29 **absolute** action fields for `[k,...,k+19]`, with end padding masked. Both loader semantics and model `use_delta_action` are false. The old7D pose-delta wrapper is not used.
- Source timing: the frozen collector reads q/qdot/EE and scene capture before `controller.step`; records those snapshots and command before advancing physics. Camera timestamps/source indices must exactly cover the same269 control frames.
- No object position, object velocity, contact or expert FSM state is passed as a policy observation. Privileged expert information is used only to construct supervision.
- Small `arrays.npz` and source-hashed `dataset.json` reference existing read-only RGB. No raw data or prior reports are overwritten.

## Model path

`finetune_hgwind_development.py` reuses the pinned official DynamicVLA policy loader and existing local weights, tokenizer and metadata. Configure action29/state30 within the pretrained32-channel architecture, absolute actions, frozen vision/connector/text and trainable action expert/state projection. This changes pretrained channel semantics; it does not imply zero-shot capability.

New statistics use only this development episode. Action statistics count valid absolute chunk targets, excluding padded repeats; standard deviations below1e-6 become1. Images use the existing upstream resize to360×480, then the policy's usual preprocessing. No augmentation is enabled for the integration check. The original Chinese task text is retained.

Optimization uses the mean flow-matching loss over **valid timesteps and the first29 channels**, excluding the three padded channels. The upstream32-channel aggregate is logged separately. This small loss wrapper is explicit; it is not an unchanged upstream training recipe. Same-frame fixed-noise flow probes before/after fitting measure an optimization check, not physical action error or held-out performance.

Default:200 AdamW steps, batch2, learning rate1e-4, gradient clipping10, seed31. Save the model configuration, full policy weights, new normalizers and provenance to a fresh run directory. There are no Isaac Sim calls, expert action execution, downloads or real-robot commands in this entry.

## Run

Prepared locally under `data/derived/hgwind_traj18_development_20261031/`:

```bash
cd /home/robophys/Capstone_Project_Yuan_Haoming
.venv_dynamicvla/bin/python scripts/finetune_hgwind_development.py --run
```

Without `--run`, only data/configuration preflight runs. `--check-load` additionally loads official weights and exercises normalization, image/language preprocessing and a synthetic masked-loss gradient on CPU, without model forward or CUDA. Optional `--steps 0 --run` runs one real forward/backward without an optimizer update or checkpoint.

To reproduce data preparation into a **new** directory:

```bash
data/derived/hf_pilot_render_repo/.venv/bin/python scripts/prepare_hgwind_development.py --run --output data/derived/hgwind_traj18_development_new
.venv_dynamicvla/bin/python scripts/finetune_hgwind_development.py --dataset data/derived/hgwind_traj18_development_new --run
```

Existing output is never overwritten. Normalizer, checkpoint-load, finite-loss/gradient and source-integrity failures stop the run. A completed training script does not imply the model grasps successfully.

## Next gate and limitations

After improving and rechecking the generated29D actions on this same episode, connect the model to the passing simulator executor using **live** camera/robot observations. A single successful grasp remains an integration demonstration, not generalization or robustness. The current training preparation does not implement that live29D policy adapter; the old7D egg adapter cannot be substituted unchanged.

The other39 stored ball trajectories lack all required upper-command labels; this work does not make them directly convertible. Contact sensitivity in other cases remains recorded. Do not silently fall back to saved expert actions during learned-policy evaluation.

Rollback: stop using these new training/data entry points. Existing simulation, original data, SDK and control defaults are unchanged.

## Verified preparation

374 portable CPU checks pass, including22 focused checks. The real CPU model load confirms exact new normalizers, state input2×32, action input2×20×32,154 instruction tokens and gradients only on valid29D targets; CUDA remains uninitialized. Those preparation checks preceded the user-run training reported above. [Source-hashed preparation](../../results/summaries/2026-10-31-ball-development-preparation.json).

Prediction preparation: 380 portable CPU checks pass, including 11 focused data/metric checks. Actual trained-checkpoint CPU reload passes strictly with exact normalizers; GPU prediction was subsequently run by the user and independently verified above.

## Startup target audit completed

The [CPU audit](BALL_STARTUP_TARGET_AUDIT.md) binds raw captured targets to training labels and identifies rebound/braking-clearance contributions. Native nominal-request comparison still shows0.569m/s mean startup discrepancy for the2,000-step model. Retain the checkpoint and interface; prepare live29D integration next. No label smoothing, extra training, controller retuning or new physical rollout was performed.

## Live integration prepared

The new [absolute29 live adapter](BALL_LIVE_POLICY.md) is CPU checked against the actual training preprocessing and strict checkpoint reload. It uses current simulator images and measured robot state, with no expert action fallback. The next user-run check is eight control intervals; learned grasp remains unverified.
