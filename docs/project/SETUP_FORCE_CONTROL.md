# Single pre-recording wind-force control

## Observed result: startup ball mismatch removed, finger response differs

User run `20261030_005620_189348_fresh60_omit1` verifies exactly one pinned omission and three remaining native setup force returns. No native startup abort occurs in this launch. Independent full-prefix rereading finds ball native transform and linear/angular velocity **exactly equal to successful history for rows0..500**; row501 first differs at2.7833334s. This is evidence that the specific force submission contributed to the earlier startup mismatch in this case. It does not establish an internal accumulation mechanism or a general reset correction.

The episode still fails:190 recorded frames/189 completed intervals, then motion safety stops joint offset4 at-2.6516rad/s against the absolute2.61 limit. Bilateral contact lasts0.2833s without sustained lift.

Both runs first command zero finger-position targets at frame154; those recorded targets match through166. At166, the omission run's finger joints remain2.192/2.490mm farther open while EE position differs by only0.0886mm. Positive raw right-finger/ball impulse begins at native step500 in history versus516 in omission (about88.9ms later); left-finger steps are511 versus520. Thus the remaining visible discrepancy includes finger closure before contact, not just a different free-moving ball path. Equal saved position targets do not establish equal native velocity targets, gains, force/speed limits or solver/mimic state.

`Expert.ensure_gripper_force` can change live drive gains/caps and USD drive attributes, while its free-travel branch preserves some authored properties. That is a code-supported reset/drive-history hypothesis, not a measured difference in gain values. Existing records lack the full actual drive properties at closure. The [passive drive pair](GRIPPER_DRIVE_DIAGNOSTIC.md) now prepares those measurements at initialization and closure, before changing parameters. Its simulator result is pending. Zero explicit finger entries in `get_applied_joint_efforts` are not measurements of the position drive's actual force.

This is an expert development experiment with modified forcing history, excluded from training. No VLA, new GPU run or rendering by the agent. [Source-hashed result](../../results/summaries/2026-10-30-setup-force-result.json).

## Question and scope

Matched fresh60 reproduces the old failure exactly, while history58/59/60 reproduces original success. Fresh60 observes two identical early force calls followed by two higher-force calls; history60 observes one early call followed by the same two higher-force calls. Both recordings begin with matching recorded positions/velocities, but their first native ball velocities differ. This motivates one force-history intervention; it does not establish force accumulation.

The opt-in experiment omits only event1 (the second, zero-based event) from the verified fresh60 pre-recording trace. First-call native return, script path, body, force, application point, mode, configured dt and simulation clock must match the pinned signature. A mismatch stops the experiment without choosing a substitute. Every other force call is forwarded through the existing passive proxy, including calls after its logging horizon. There is no force clearing, extra physics step, scene edit, solver retuning, safety-limit change or online action replacement. The frozen fan consumes the API call as a statement; the omitted call returns None.

This **intentionally changes applied-force history**. It is an exposed development diagnosis, not an original expert reproduction, general fan correction, training example, learned policy or physical-generalization result.

## Run exactly one attempt

```bash
cd /home/robophys/Capstone_Project_Yuan_Haoming

data/derived/hf_pilot_render_repo/.venv/bin/python \
  scripts/run_hgwind_history_diagnostic.py \
  --fresh-control --stage-load-barrier \
  --omit-setup-force-once --native-backtrace --run
```

Without `--run`, this performs CPU preflight only. Rendering remains off by default. Outputs use a new `_fresh60_omit1` directory and retain failed outcomes; no automatic retries occur. Omit `--omit-setup-force-once` to return to the unmodified fresh-control path. Original sources and baseline recordings are retained.

## Read the result in order

1. Require `SETUP_FORCE_CONTROL verified=true`. The audit and force trace must show one exact pre-recording omission and entry into recording. Otherwise the intervention was not evaluated.
2. Read `SETUP_FORCE_COMPARISON`: compare the first native ball velocity and all15 core arrays against the successful history reference. Do not judge from grasp outcome alone.
3. Read independent grasp, initial-state and motion-safety outcomes separately. A saved safety abort can still yield startup comparison evidence.

If the first-step mismatch disappears, this supports a contribution from this specific force submission. If the complete episode also matches history, the evidence is stronger for this case, but a general reset fix remains unverified. If only startup matches, inspect where the remaining rollout diverges. If startup still differs, this single omission is insufficient; reassess initialization evidence before another change. API call counts at equal clocks alone do not prove native impulse accumulation.

## Implementation and verification

- `setup_force_control.prepare_intervention` pins the verified control reports/raw-force trace and unchanged task/controller/collector dependencies.
- The separate opt-in worker installs the proxy before entering the existing history worker. Default startup and passive recording source remain unchanged.
- `setup_force_control.audit_intervention` blocks comparison/success certification if the intervention lacks evidence. Root results explicitly mark modified forcing history and exclude training eligibility.
- 287 portable CPU checks pass; the final focused56 checks and actual scene/native-assembly preflight pass. SDK Python3.10 parses the new entry/module and coordinator. The subsequent user-run outcome is documented above; these CPU checks are preparation evidence.

[Preparation provenance](../../results/summaries/2026-10-30-setup-force-control-prepared.json).
