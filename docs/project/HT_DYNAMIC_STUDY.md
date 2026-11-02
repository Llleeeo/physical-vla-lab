# Fast action prediction for dynamic manipulation

Research plan adopted on 2 November 2026. Implementation and evaluation of HT are pending. [Starting inventory](HT_READINESS_AUDIT.md).

## Question

Can heteroscedastic Student-t action regression retain the precision needed for dynamic contact while reducing observation-to-action delay? DynamicVLA is the proposed model baseline. Selected RoboPhys tasks can provide demonstrations and physical records for diagnosing premature contact, missed closure and loss after grasping.

HT-Policy already proposes efficient robust action regression. Fast policies for moving targets and critical-state weighting also have prior work. This project does not claim those ideas as new. The possible extension is conditional: determine whether HT suppresses useful supervision near consequential events, then test a change to its objective against simple controls. No improved method or novelty claim is established by the current audit.

## Experimental sequence

1. Reproduce one manageable official HT experiment in RoboMimic. Verify dataset and software revisions, evaluate available reference weights, and measure local resource requirements. A state-based Lift configuration is a candidate, subject to checking the release.
2. Validate one dynamic task using existing demonstrations. Expert commands must work through the same downstream interface as learned actions. Freeze episode-level splits before model comparison; related initializations stay in one split.
3. Compare the original DynamicVLA Flow predictor with direct MSE and direct HT regression. Use a shared observation/action definition and fixed controller. Converting the flow action expert requires a direct readout and scale prediction, not merely one flow integration step.
4. Evaluate each method with its measured end-to-end delay, then under a shared feasible delay and command schedule. Physics must advance during the represented inference interval. Preserve timestamp and missed-deadline records.
5. Relate action errors and HT scales/gradients to contact outcomes. If supported, test bounded event-focused auxiliary supervision against uniform supervision and simple event weighting. Simulator state can supply training labels and diagnostics, not privileged deployment observations.
6. Expand to a second task and repeated training seeds after the first comparison is valid. Use held-out initializations for final reporting. Hardware validation is optional after the simulation study.

The tentative final baseline matrix is two tasks, three methods, three training seeds, 30 shared test initializations and two timing conditions: 1,080 rollouts. This is a planning target, subject to profiling, not a completed result. Task/controller feasibility is a Week 4 decision point; a compatible DOM task is the fallback if RoboPhys remains blocked.

## Immediate work

The resource audit is complete at the artifact-inspection level. Next inspect and pin the official HT source/configuration, resolve the reproduction dataset and model files, and verify GPU visibility. Then prepare a separate environment with a small loader/model check. No new collection campaign or further tuning of the old wind-grasp controller is required for this step.

## Reading

- [HT-Policy: Residual Modeling Closes the Regression and Generative Policy Gap in Robot Learning](https://arxiv.org/abs/2610.12231v2), with the [official implementation](https://github.com/the-labone/regression-policy).
- [DynamicVLA](https://arxiv.org/abs/2601.22153v2).
- [FASTER](https://arxiv.org/abs/2603.19199v3) and [FODMP](https://arxiv.org/abs/2603.24806) for efficient action generation and time-dependent execution.
- [Where Small Mistakes Matter](https://arxiv.org/abs/2610.17709v2) and [Saliency-Aware Quantized Imitation Learning](https://arxiv.org/abs/2505.15304v2) for the distinction between ordinary prediction error and consequential errors.

The experimental question remains open. The first deliverable is a reproducible comparison, not a promised improvement over HT or DynamicVLA.
