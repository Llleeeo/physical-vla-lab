#!/usr/bin/env python3
"""Audit all saved HGwind attempts from immutable metadata and physical traces."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'data/derived/hgwind_current_baseline_20261027/repo/runs/20261027_165207_242720'


def first(mask):
    ids=np.flatnonzero(mask)
    return int(ids[0]) if len(ids) else None


def longest(mask,dt):
    edges=np.diff(np.r_[False,mask,False].astype(int))
    return float(np.max(np.flatnonzero(edges==-1)-np.flatnonzero(edges==1),initial=0)*dt)


def audit_episode(ep):
    m=json.loads((ep/'meta.json').read_text());a={}
    for line in (ep/'analyse.jsonl').open():
        record=json.loads(line)
        if record['type']=='control_telemetry':
            a[record['data']['name']]=np.asarray(record['data']['values'])
    term=m['skill_terminations']['prim_1'];safety=m['runtime_safety']
    if m['success']:category='success'
    elif not safety['passed']:category='runtime_safety_abort'
    elif 'T_stop' in str(term.get('fail_reason')):category='lost_capture_then_stopped'
    elif not np.any(a['close_request']):category='no_close_timeout'
    else:category='other_failure'
    with h5py.File(ep/'replay.hdf5') as f:
        t=f['timestamp'][:];dt=float(np.median(np.diff(t)));oi=m['object_paths'].index(m['policy']['roles']['A'])
        position=f['obs/object_pos'][:,oi,:];velocity=f['obs/object_vel'][:,oi,:]
        both=a['contact_L']&a['contact_R'];onset=first(both)
        q=f['obs/ee_quat'][:];ee=f['obs/ee_pos'][:]
        pinch=ee+Rotation.from_quat(q[:,[1,2,3,0]]).apply(m['validation']['pinch_frame']['offset_local_m'])
        distance=np.linalg.norm(pinch-a['grasp_center_world'],axis=1)
        phase=a['skill_phase'][:,0];changes=np.flatnonzero(np.r_[True,phase[1:]!=phase[:-1]])
        headers=f['evidence/contacts/headers'][:];points=f['evidence/contacts/points'][:];paths=f['evidence/contacts/paths'].asstr()[:].tolist()
        target_id=paths.index(m['policy']['roles']['A']);positive_ticks={}
        for finger in ('panda_leftfinger','panda_rightfinger'):
            ids={i for i,p in enumerate(paths) if p.endswith('/'+finger)};ticks=set()
            for h in headers:
                if h[7]<=0 or not ((h[1]==target_id and h[2] in ids) or (h[2]==target_id and h[1] in ids)):continue
                data=points[int(h[6]):int(h[6]+h[7])]
                if np.any(np.linalg.norm(data[:,6:9],axis=1)>1e-12):ticks.add(int(h[0]))
            positive_ticks[finger]=len(ticks)
        lift_starts=np.flatnonzero((phase==12)&np.r_[True,phase[:-1]!=12])
        jump=None
        if len(lift_starts):
            k=int(lift_starts[0]);v=a['live_grasp_target_vel']
            jump=float(np.linalg.norm(v[min(k+1,len(v)-1)]-v[max(0,k-1)]))
        effort=m['arm_execution']['dynamic_drive']['effort_feasibility']
        row={'episode':ep.name,'attempt_index':m['attempt_index'],'episode_seed':m.get('episode_seed'),
             'frames':len(t),'category':category,'collector_success':m['success'],
             'trajectory_valid':m['trajectory_validity']['passed'],'close_frames':int(a['close_request'].sum()),
             'first_close_frame':first(a['close_request']),'bilateral_frames':int(both.sum()),
             'first_bilateral_frame':onset,'longest_bilateral_s':longest(both,dt),
             'raw_positive_impulse_ticks':positive_ticks,
             'no_assistance_observed':bool(not a['attach_grasp_active'].any() and not m['fix_joint'] and not m['fixedforce']),
             'minimum_pinch_error_mm':float(distance.min()*1000),
             'post_contact_peak_lift_m':float(np.max(position[onset:,2])-position[onset,2]) if onset is not None else None,
             'post_contact_final_lift_m':float(position[-1,2]-position[onset,2]) if onset is not None else None,
             'pre_first_contact_speed_mps':float(np.linalg.norm(velocity[max(0,first(a['contact_L']|a['contact_R'])-1)])) if onset is not None else None,
             'lift_entry_command_velocity_change_mps':jump,
             'phase_transitions':[{'frame':int(k),'phase_code':int(phase[k])} for k in changes],
             'terminal_reason':safety['abort_reason'] or term.get('fail_reason'),
             'effort_failure':effort.get('last_failure'),
             'maximum_raw_joint_speed_ratio':m['arm_execution']['dynamic_drive']['maximum_raw_velocity_ratio'],
             'input_sha256':{n:hashlib.sha256((ep/n).read_bytes()).hexdigest() for n in ('meta.json','replay.hdf5','analyse.jsonl')}}
        return row


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,default=DEFAULT)
    args=parser.parse_args();run=args.run_dir.resolve();config=json.loads((run/'run.json').read_text())
    task=run/'Trajectory/1s1p/grasp_wind_driven_motion_1s1p_hgprgew'
    episodes=sorted(p for p in task.iterdir() if p.is_dir() and p.name.startswith(('traj_','failed_')))
    rows=sorted((audit_episode(p) for p in episodes),key=lambda r:r['attempt_index'])
    indices=[r['attempt_index'] for r in rows]
    report={'run':str(run),'attempt_budget':config.get('attempt_budget'),
            'run_exit_code':config.get('exit_code'),'saved_episodes':len(rows),
            'unique_attempt_indices':len(set(indices)),'attempt_indices':indices,
            'categories':dict(Counter(r['category'] for r in rows)),
            'scene_sha256':config['task_sha256'],'rows':rows,
            'scope':'Development expert collection; no VLA result or held-out generalization claim',
            'summary_note':'expert_collection_summary.json describes only the final worker after safety restarts; totals here are rebuilt from every saved episode.'}
    if len(set(indices))!=len(indices):raise RuntimeError('Duplicate attempt indices')
    out=run/'analysis';out.mkdir(exist_ok=True)
    (out/'batch_failure_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    fields=['attempt_index','episode','category','frames','first_close_frame','bilateral_frames',
            'longest_bilateral_s','post_contact_peak_lift_m','post_contact_final_lift_m',
            'lift_entry_command_velocity_change_mps','maximum_raw_joint_speed_ratio','terminal_reason']
    with (out/'attempts.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    print('BATCH_AUDIT '+json.dumps({k:report[k] for k in ('saved_episodes','unique_attempt_indices','categories','run_exit_code')},ensure_ascii=False))
    print('REPORT '+str(out/'batch_failure_audit.json'))


if __name__=='__main__':main()
