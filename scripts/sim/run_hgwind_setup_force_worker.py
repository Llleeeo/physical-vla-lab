#!/usr/bin/env python3
"""One fresh60 force-history intervention; no original source file changes."""
import json
import os
from pathlib import Path
import runpy
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from capstone_vla.setup_force_control import install
plan = json.loads(Path(os.environ['HGWIND_EFFORT_PROBE_MANIFEST']).read_text())
install(plan)
print('SETUP_FORCE_CONTROL_ACTIVE omit_event=1 phase=pre_recording modified_forcing_history=true', flush=True)
runpy.run_path(str(ROOT/'scripts/sim/run_hgwind_history_worker.py'), run_name='__main__')
