#!/usr/bin/env python3
"""Execute the frozen collector with task-scoped in-memory probe callbacks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from capstone_vla.hgwind_effort_probe import patch_worker
from capstone_vla.hf_archive import sha256
plan=json.loads(Path(os.environ['HGWIND_EFFORT_PROBE_MANIFEST']).read_text())
parser=argparse.ArgumentParser(add_help=False)
parser.add_argument('--task');parser.add_argument('--grasp_mode');parser.add_argument('--arm_execution')
args,_=parser.parse_known_args()
if Path(args.task).resolve()!=Path(plan['task']) or args.grasp_mode!='physical' or args.arm_execution!='inverse_dynamics':
    raise RuntimeError('Unexpected task or actuator mode for isolated probe')
if any(x in sys.argv for x in ['--replay_state_exact','--replay_trajectories','--replay_trajectory','--pilot-comparison-out']):
    raise RuntimeError('Exact-state/position-drive probe flags are forbidden')
worker=Path(plan['worker']);sys.path.insert(0,str(worker.parents[1]))
if sha256(worker)!=plan['worker_sha256']:
    raise RuntimeError('Frozen worker changed after preparation')
source=patch_worker(worker.read_text())
if hashlib.sha256(source.encode()).hexdigest()!=plan['patch_sha256']:
    raise RuntimeError('Probe adapter changed after preparation')
print('HGWIND_EFFORT_PROBE_ACTIVE mode='+plan['mode']+' shadow_motor_references=discarded physical_grasp=true',flush=True)
exec(compile(source,str(worker),'exec'),{'__name__':'__main__','__file__':str(worker)})
