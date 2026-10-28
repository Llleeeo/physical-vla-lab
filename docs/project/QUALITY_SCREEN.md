# Ball-cohort quality screen

Snapshot: 28 October 2026. All 40 rendered development episodes were screened using saved physical state/contact records and categorical target segmentation. Nine five-event contact sheets were visually inspected; this is not full-video review of all episodes.

## Descriptive observations

Each table entry is the median across episodes of the specified episode-level metric.

| Measurement | Main/local (21) | Remote/hongtao (19) |
|---|---:|---:|
| Final 0.5 s relative ball-center jitter, within-episode p95 | 0.266 mm | 0.402 mm |
| First LIFT entry to final relative center displacement | 4.04 mm | 2.33 mm |
| Minimum final-window lift from the first raw bilateral contact state | 52.38 mm | 52.29 mm |
| Final 0.5 s relative speed from recorded velocities, within-episode p95 | 0.01475 m/s | 0.01462 m/s |
| Episodes with more than one LIFT entry | 5 | 1 |

Local has a smaller typical final-window center jitter; the remote group has a smaller typical displacement from first LIFT. That displacement can include recapture and regrasp, so it is not a pure slip measurement. Relevant exploratory bootstrap intervals for differences in jitter, center displacement and final-window lift span zero. This small selected-success comparison does not establish an overall source-group quality advantage.

Groups have distinct base seeds (202610280 versus 202710280), no shared episode seeds, and different approach-depth settings (2 mm versus 0 mm). They are neither paired trials nor a causal depth-bias experiment. These exposed development episodes cannot become a clean held-out benchmark after inspection.

## Contact resolution matters

Positive impulse masks were rebuilt from the 180 Hz physical contact records, independently of defaultable controller telemetry. At the final 0.5 s, all 40 episodes had left and right positive contacts within every 60 Hz control bin. This is different from simultaneous bilateral impulse on each physical substep.

`local/traj_00019` alternates the two fingers at physical-substep resolution. Its final window has no same-substep bilateral positive impulse, while both fingers still make positive contact within each control bin. Its recorded relative-speed p95 is about 0.380 m/s, much larger than either group's median. It deserves targeted inspection for high-frequency contact dynamics; these observations alone do not prove it was ungrasped. `local/traj_00016` and `local/traj_00012` are additional relative-velocity cases for review.

Relative position uses the ball center in the rotating hand frame with the recorded pinch offset. Relative velocity subtracts hand linear velocity and the angular-motion contribution. Neither metric is material-point tangential slip, a friction-force certificate, or a requirement to stop ball rotation. Lift is referenced to first contact, not the initial spawn/drop height. Original task labels remain intact.

## Visibility findings

Segmentation was sampled at precontact, first bilateral contact, first LIFT, first HOLD and final frames, plus sparse final-window samples: 400 distinct episode/frame samples per camera in total.

- `opst_cam`, `side_left_cam` and revised `side_right_cam`: target visible and not touching the image boundary in all sampled frames.
- `wrist_cam`: target visible in all 400 samples, but its segmentation touched the image edge in 355 (203 local, 152 remote). Inspected sheets confirm frequent lower-edge truncation.
- `overhead_cam`: zero visible target mask in 223/400 samples, consistent with occlusion in inspected sheets; a zero mask alone cannot distinguish occlusion from an out-of-frame target.

The sampled images do not certify every video frame. Wide external views also make the ball small; camera input selection/preprocessing should be explicit before training. Existing images remain usable for inspection, and no automatic rerendering is triggered by this report.

## Decision and next checks

Retain both source groups and preserve episode-level provenance. Do not increase sampling weights simply because an episode came from the main machine. Establish uniform episode sampling on the eventual training split first, then compare any quality-aware weighting under a fixed training budget and fixed validation distribution.

Before freezing eligible training episodes, inspect the high-relative-speed cases and the recapture sequences, document episode-level decisions, and settle how wrist truncation is handled in the observation interface. No automatic exclusions, split assignment, action conversion or new training has occurred in this screen.

## Reproduction

`scripts/analyze_hgwind_cohort_quality.py` reads the existing render manifest and immutable recordings, then writes a detailed JSON report, CSV metrics, a comparison plot, one event sheet per episode, and a local HTML review index. Requires the documented CPU libraries plus Matplotlib (locally 3.11.1); does not launch Isaac, use a GPU or contact the Hub. Full source recordings are required separately.

`src/capstone_vla/trajectory_quality.py` contains the pure computations. Eleven focused synthetic tests cover rotating-hand compensation, first-contact height/acceleration anchors, missing events, time validity, substep alternation, categorical visibility and deterministic descriptive bootstrap results. See the [compact evidence summary](../../results/summaries/2026-10-28-quality-screen.json).
