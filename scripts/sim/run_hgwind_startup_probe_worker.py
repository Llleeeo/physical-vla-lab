#!/usr/bin/env python3
"""Passive startup tracing around the unchanged native-expert worker."""
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from capstone_vla import hgwind_expert_trace as trace
from capstone_vla.hgwind_startup_trace import configure, patch_startup_collector
plan = json.loads(Path(os.environ['HGWIND_EFFORT_PROBE_MANIFEST']).read_text())
if not plan.get('trace_startup_forces') or plan['tracking'] != 'native-expert':
    raise ValueError('Startup diagnostic requires explicitly enabled native expert')
configure(Path(plan['output']) / 'startup_force_calls.jsonl')
original_patch = trace.patch_collector
trace.patch_collector = lambda source: patch_startup_collector(original_patch(source))
runpy.run_path(str(ROOT / 'scripts/sim/run_hgwind_native_expert_worker.py'), run_name='__main__')
