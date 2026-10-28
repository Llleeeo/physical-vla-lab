#!/usr/bin/env python3
"""Apply an explicit capstone control profile around the unchanged collector."""
import argparse
import hashlib
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/'data/derived/hgwind_current_baseline_20261027/repo'
sys.path[:0]=[str(ROOT/'src'),str(REPO/'Scripts')]
parser=argparse.ArgumentParser(add_help=False)
parser.add_argument('--task');parser.add_argument('--grasp_mode')
args,_=parser.parse_known_args()
if Path(args.task or '').stem!='grasp_wind_driven_motion_1s1p_hgprgew' or args.grasp_mode!='physical':
    raise RuntimeError('This control experiment is restricted to HGwind physical grasp collection')
if os.environ.get('CAPSTONE_HGWIND_CONTROL_PROFILE') not in ('smooth-lift', 'smooth-only', 'mimic-safe'):
    raise RuntimeError('Explicit adapted control profile is required')
from expert_data.skills import catch
from capstone_vla.hgwind_control import install_profile,profile_summary,profile_configuration
configuration=install_profile(catch, profile_configuration(os.environ['CAPSTONE_HGWIND_CONTROL_PROFILE']))
worker=REPO/'Scripts/expert_data/collect_worker.py'
source=worker.read_text()
anchor='"skill_terminations": _jsonify(skill_terminations),'
if source.count(anchor)!=1:
    raise RuntimeError('Collector metadata insertion point changed')
source=source.replace(anchor,anchor+'\n            "capstone_expert_control": CAPSTONE_PROFILE_SUMMARY(pol),')

extra_globals={}
if os.environ['CAPSTONE_HGWIND_CONTROL_PROFILE']=='mimic-safe':
    if '--replay_trajectory' in sys.argv or '--replay_trajectories' in sys.argv:
        raise RuntimeError('Mimic actuation diagnostic is fresh collection only')
    from capstone_vla.hgwind_mimic_actuation import (patch_worker_source,
        install_after_policy_import,actuated_finger_indices,summary as mimic_summary)
    source=patch_worker_source(source)
    extra_globals={'CAPSTONE_INSTALL_MIMIC_ACTUATION':install_after_policy_import,
                   'CAPSTONE_ACTUATED_FINGER_INDICES':actuated_finger_indices,
                   'CAPSTONE_MIMIC_SUMMARY':mimic_summary}

if os.environ.get('CAPSTONE_HGWIND_CAPTURE_SPEED_GATE') == '1':
    if os.environ['CAPSTONE_HGWIND_CONTROL_PROFILE'] != 'mimic-safe':
        raise RuntimeError('Capture speed diagnostic requires mimic-safe physical actuation')
    from capstone_vla.hgwind_capture_gate import install_capture_speed_gate, summary as gate_summary
    install_capture_speed_gate(catch.Catch)
    if source.count(anchor) != 1:
        raise RuntimeError('Capture gate metadata insertion point changed')
    source = source.replace(anchor, anchor + '\n            "capstone_capture_speed_gate": CAPSTONE_CAPTURE_GATE_SUMMARY(pol),')
    extra_globals['CAPSTONE_CAPTURE_GATE_SUMMARY'] = gate_summary
    print('CAPSTONE_CAPTURE_SPEED_GATE enabled=true existing_relative_limit=true extra_dwell=false', flush=True)

print('CAPSTONE_HGWIND_CONTROL_ACTIVE profile=' + configuration['name']
      + ' settle_s=' + str(configuration['settle_s'])
      + ' lift_speed_scale=' + str(configuration['lift_speed_scale']), flush=True)
exec(compile(source,str(worker),'exec'),{'__name__':'__main__','__file__':str(worker),
     'CAPSTONE_PROFILE_SUMMARY':profile_summary,**extra_globals})
