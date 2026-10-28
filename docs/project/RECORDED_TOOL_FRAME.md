# Recorded Franka tool-frame audit

The wind-driven-ball cohort requires an explicit bridge from its recorded right-gripper frame to a policy action frame. This CPU audit resolves the kinematic convention before changing execution code.

## Verified differences

All 40 recordings identify the source arm mode as `operational_space_inverse_dynamics_effort`. The older egg-task runner and `pilot_recorded_actuators.restore_recorded_franka_pd` explicitly require position-drive sources. The older runner also assumes one dynamic object, whereas the ball scene includes a fan. It is inappropriate to relax those checks and quietly restore position drives for this cohort.

The recordings define `right_gripper` relative to `panda_link7`, with local translation approximately `[0, 0, 0.207]` metres and a recorded orientation. The earlier integration used `panda_hand` plus local Z 0.1034 m. Through the actual URDF fixed-joint chain, these tool conventions differ by about 3.4 mm and 180 degrees.

Both can be valid coordinate conventions if observation labels, actions and execution are consistently transformed. The difference does not prove a prior controller failure, nor establish the cause of earlier learned-policy failures.

## Implementation and CPU validation

`src/capstone_vla/recorded_franka.py` wraps the existing `FrankaCartesianAdapter`. It composes the complete fixed-joint tail after the last revolute joint, then applies the end-effector transform declared by the recording. The linear Jacobian is corrected for the displaced tool origin; the angular Jacobian and original action encoding are retained. The old adapter is unchanged.

`scripts/prepare_hgwind_action_probe.py` compares measured-joint FK against the recorded physical EE poses, using the scene's fixed robot-base transform. All 40 episodes (12,154 frames) pass the existing 1 mm / 0.001 rad FK acceptance thresholds. Maximum corrected discrepancies are **0.001593 mm** and **0.0001263 degrees**. Source-file checksums are verified before and after processing; scene SHA remains unchanged.

The script also writes a small, separate action probe for `local/traj_00000`: measured robot state and Cartesian targets in the recorded tool convention, with 60 Hz timestamps and preserved binary gripper transitions. No images are copied and no policy-facing object state is introduced. This is not a LeRobot export or a training-ready dataset.

Six synthetic regression checks cover the fixed-frame transformation, Jacobian finite differences, cancellation of the legacy virtual-tip offset, identity-goal IK and invalid transform rejection. Real-data FK validation is a separate, asset-dependent check.

## Remaining execution gate

Reuse and audit the frozen RoboPhys effort-controller path for the actuator baseline and converted-command comparison. In particular, preserve model compensation, command limits, control/physics cadence and physical gripper behavior. Do not substitute the earlier position-drive replay helper or interpret exact-state rendering as physical action replay.

Camera selection remains open: the quality screen identified wrist truncation, but a smaller visible target in external views also matters. No observation profile, episode eligibility split, sampling weights, physics replay, new policy training or grasp-success claim is certified by this audit.

Evidence: [recorded-tool summary](../../results/summaries/2026-10-28-recorded-tool-frame.json).
