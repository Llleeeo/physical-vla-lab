#!/usr/bin/env python3
"""Inspect per-physics-step controller evidence without certifying task success."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
with h5py.File(args.run/'cartesian_trace.hdf5') as f:
 d=f['servo_diagnostics'];index=d['command_index'][:]
 out={'schema':'capstone_task2_servo_diagnosis_v1','frames':len(f['actions']),
      'physics_ticks':len(index),'ik_solver':str(f.attrs.get('ik_solver','clipped')),
      'bounded_from_command':int(f.attrs.get('bounded_from_command',0)),
      'official_success_verified':False,
      'notes':['ik_delta_clipped/target_limit_clipped evaluate the unconstrained candidate, not the bounded solver output',
               'nominal_pd_effort_nm is an estimate; solver_projected_effort_nm is net solver joint effort, not isolated motor torque']}
 for name,mask in [('before_contact',index<115),('contact_lift',(index>=115)&(index<=160)),('late_lift',index>160)]:
  item={}
  for key in ['ik_delta_clipped','target_limit_clipped','velocity_clipped']:
   item[key+'_ticks_per_joint']=np.sum(d[key][:][mask],axis=0).tolist()
  for key in ['qd','nominal_pd_effort_nm','solver_projected_effort_nm','unclipped_ik_delta_rad']:
   item[key+'_absmax_per_joint']=np.max(np.abs(d[key][:][mask]),axis=0).tolist()
  for key in ['goal_position_error_m','goal_orientation_error_rad']:
   values=d[key][:][mask];item[key]={'p95':float(np.percentile(values,95)),'max':float(np.max(values))}
  out[name]=item
 obj=f['object_pos'][:,0,:]
 out['lift_evidence']={'initial_object_z_m':float(obj[0,2]),'final_object_z_m':float(obj[-1,2]),
                       'last_second_min_z_m':float(np.min(obj[-60:,2])),
                       'bilateral_contact_frames':int(np.sum(f['contact_L'][:]&f['contact_R'][:]))}
(args.run/'servo_diagnosis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
