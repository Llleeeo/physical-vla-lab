#!/usr/bin/env python3
"""Record a bounded original expert sequence within one native worker."""
import json
import os
from pathlib import Path
import runpy
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from capstone_vla import hgwind_expert_trace as trace
from capstone_vla import hgwind_history as history
from capstone_vla.hgwind_startup_trace import configure
plan = json.loads(Path(os.environ['HGWIND_EFFORT_PROBE_MANIFEST']).read_text())
history.configure(plan)
configure(Path(plan['output']) / 'pre_attempt_force_calls.jsonl')
original_patch = trace.patch_collector
trace.patch_collector = lambda source: history.patch_history_collector(original_patch(source))
runpy.run_path(str(ROOT / 'scripts/sim/run_hgwind_native_expert_worker.py'), run_name='__main__')
