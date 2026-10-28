#!/usr/bin/env python3
"""Read-only independent grasp evidence check; output is a new JSON report."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import h5py
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from capstone_vla.grasp_evidence import evaluate_grasp, GraspEvidenceCriteria
p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--motion',choices=('linear','angular'),default='linear');p.add_argument('--require-lift',action=argparse.BooleanOptionalAction,default=True);a=p.parse_args()
trace=a.run/'cartesian_trace.hdf5'
with h5py.File(trace) as f:
    keys=['timestamp_s','ee_pos','ee_quat','object_pos','object_vel','contact_L','contact_R','impulse_L','impulse_R','grasp_assistance_audit','object_quat','object_ang_vel']
    result=evaluate_grasp({key:f[key][:] for key in keys if key in f},GraspEvidenceCriteria(motion_requirement=a.motion,require_lift=a.require_lift))
    result['actual_grasp_mode']=str(f.attrs.get('actual_grasp_mode','not_recorded'))
    result['grasp_joint_candidates']=json.loads(f.attrs.get('grasp_joint_candidates_json','null'))
result['source_sha256']=hashlib.sha256(trace.read_bytes()).hexdigest()
result['verifier_sha256']=hashlib.sha256((ROOT/'src/capstone_vla/grasp_evidence.py').read_bytes()).hexdigest()
output=a.run/'independent_grasp_evidence.json'
if output.exists():raise FileExistsError(output)
output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
