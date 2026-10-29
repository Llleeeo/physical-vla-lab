#!/usr/bin/env python3
"""Fresh-only diagnostic gain setter after original drive initialization."""
import json
import os
from pathlib import Path
import runpy
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from capstone_vla.gripper_drive_trace import install as observe
from capstone_vla.gripper_gain_control import install as control
plan=json.loads(Path(os.environ['HGWIND_EFFORT_PROBE_MANIFEST']).read_text())
observe(plan)
control(plan)
print('GRIPPER_GAIN_CONTROL_ACTIVE one_frame0_reapplication=true mimic=retained production_fix=false',flush=True)
runpy.run_path(str(ROOT/'scripts/sim/run_hgwind_setup_force_worker.py'),run_name='__main__')
