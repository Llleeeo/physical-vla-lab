#!/usr/bin/env python3
"""Passive drive snapshots around the existing bounded history/fresh workers."""
import json
import os
from pathlib import Path
import runpy
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from capstone_vla.gripper_drive_trace import install
plan=json.loads(Path(os.environ['HGWIND_EFFORT_PROBE_MANIFEST']).read_text())
if plan.get('trace_gripper_drives') is not True:raise ValueError('Explicit observation flag required')
install(plan)
print('GRIPPER_TRACE_ACTIVE setters=false added_physics_steps=0',flush=True)
entry='run_hgwind_setup_force_worker.py' if plan.get('setup_force_control') else 'run_hgwind_history_worker.py'
runpy.run_path(str(ROOT/'scripts/sim'/entry),run_name='__main__')
