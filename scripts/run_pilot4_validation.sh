#!/usr/bin/env bash
set -euo pipefail
pilot_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$pilot_root/data/derived/hf_pilot_render_repo/.venv/bin/python" \
  "$pilot_root/scripts/run_task2_interface.py" \
  --episode "$pilot_root/data/raw/hf_pilot_2026_10_24/RoboPhys001/Trajectory-004/Trajectory/1s1p/grasp_fixed_axis_rotation_1s1p_akrsd68/traj_00000" \
  --task "$pilot_root/data/derived/development_validation_20261025/pilot4_workspace/Tasks/1s1p/grasp_fixed_axis_rotation_1s1p_akrsd68.usd" \
  --mode cartesian --velocity-reference causal --ik-solver adaptive \
  --grasp-mode physical --enforce-frozen-controller "$@"
