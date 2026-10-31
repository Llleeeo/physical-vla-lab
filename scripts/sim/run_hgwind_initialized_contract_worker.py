#!/usr/bin/env python3
"""One29D surrogate with explicit measured fresh60 setup; no online skill step."""
import json
import os
from pathlib import Path
import runpy
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from capstone_vla.initialized_contract import install
plan=json.loads(Path(os.environ['HGWIND_EFFORT_PROBE_MANIFEST']).read_text())
install(plan)
print('INITIALIZED_CONTRACT_ACTIVE force_omit1=true gain_reapplication=true online_skill_step=false',flush=True)
runpy.run_path(str(ROOT/'scripts/sim/run_hgwind_native_expert_worker.py'),run_name='__main__')
