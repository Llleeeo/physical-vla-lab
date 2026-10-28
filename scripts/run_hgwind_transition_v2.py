#!/usr/bin/env python3
"""Run ten bounded local attempts with the added capture dwell removed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',help='GPU run; default CPU prepare only')
    args=parser.parse_args()
    config=json.loads((ROOT/'configs/hgwind_transition_v2.json').read_text())
    scene=ROOT/'data/derived/hgwind_scene_edit_20261027/Tasks/1s1p/grasp_wind_driven_motion_1s1p_hgprgew.usd'
    if hashlib.sha256(scene.read_bytes()).hexdigest()!=config['scene_sha256']:
        raise RuntimeError('Scene changed since this transition-only comparison was specified')
    command=[sys.executable,str(ROOT/'scripts/collect_hgwind_baseline.py'),
             '--scene',config['scene'],'--targeting-mode',config['targeting_mode'],
             '--seed',str(config['seed']),'--attempts',str(config['attempts']),
             '--control-profile',config['control_profile'],'--experiment-label',config['experiment_label'],
             '--baseline-report',str(ROOT/config['baseline_report'])]
    if args.run:command.append('--run')
    print('TRANSITION_V2 original_capture_gate=true added_dwell=false smooth_lift=true hard_guards=unchanged',flush=True)
    return subprocess.call(command,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))


if __name__=='__main__':raise SystemExit(main())
