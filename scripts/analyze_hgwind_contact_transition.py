#!/usr/bin/env python3
"""Extract compact contact/transition evidence before user-authorized raw cleanup."""
import argparse
from collections import Counter
import json
from pathlib import Path

import h5py
import numpy as np

TASK='grasp_wind_driven_motion_1s1p_hgprgew'


def first(mask):
    ids=np.flatnonzero(mask)
    return int(ids[0]) if len(ids) else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();run=args.run_dir.resolve()
    baseline=json.loads((run/'analysis/batch_failure_audit.json').read_text())
    indexed={r['episode']:r for r in baseline['rows']};rows=[]
    for ep in sorted((run/'Trajectory/1s1p'/TASK).iterdir()):
        if ep.name not in indexed:continue
        row=dict(indexed[ep.name]);m=json.loads((ep/'meta.json').read_text())
        telemetry={}
        for line in (ep/'analyse.jsonl').open():
            d=json.loads(line)
            if d['type']=='control_telemetry':telemetry[d['data']['name']]=np.asarray(d['data']['values'])
        phase=telemetry['skill_phase'][:,0];row['last_phase']=int(phase[-1])
        row['entered_lift']=bool(np.any(phase==12))
        controls=m['capstone_expert_control']['primitives']
        row['control_configuration']=controls[0]['configuration'] if controls else None
        row['transition_measurements']=controls[0]['measurements'] if controls else {}
        row['ever_physical_grasp_confirmed']=bool(np.any(telemetry['physical_grasp']))
        row['last_capture_relative_speed_mps']=float(telemetry['capture_relative_speed'][-1])
        indices=m['arm_execution']['arm_dof_indices'];names=[m['joint_names'][i] for i in indices]
        failure=row.get('effort_failure') or {}
        row['infeasible_joints']=[names[i] for i in failure.get('joint_offsets',[]) if 0<=i<len(names)]
        with h5py.File(ep/'replay.hdf5') as f:
            time=f['evidence/motion/timestamp'][:]
            velocity=f['evidence/motion/obs/joint_vel'][:,indices]
            limits=np.asarray(m['physical_limit_contract']['max_velocity'])
            ratios=abs(velocity)/limits
            crossing=first(np.any(ratios>1+1e-6,axis=1))
            row['first_dense_velocity_limit_crossing_s']=float(time[crossing]) if crossing is not None else None
            row['max_joint_velocity_ratios']={name:float(value) for name,value in zip(names,ratios.max(axis=0))}
            row['max_joint_speeds_rad_s']={name:float(value) for name,value in zip(names,abs(velocity).max(axis=0))}
            both=first(telemetry['contact_L'] & telemetry['contact_R'])
            row['first_bilateral_contact_s']=float(f['timestamp'][both]) if both is not None else None
            row['same_recorded_start_state']={'joint_position':f['obs/joint_pos'][0].tolist(),
                                            'object_position':f['obs/object_pos'][0].tolist(),
                                            'object_velocity':f['obs/object_vel'][0].tolist()}
        rows.append(row)
    unsafe=[r for r in rows if r['category']=='runtime_safety_abort']
    counters={
        'all_attempts':len(rows),'successes':sum(r['collector_success'] and r['trajectory_valid'] for r in rows),
        'safety_aborts':len(unsafe),
        'safety_with_bilateral_contact':sum(r['bilateral_frames']>0 for r in unsafe),
        'safety_with_positive_raw_impulses_both_fingers':sum(min(r['raw_positive_impulse_ticks'].values())>0 for r in unsafe),
        'safety_last_phase_track':sum(r['last_phase']==3 for r in unsafe),
        'safety_last_phase_lift':sum(r['last_phase']==12 for r in unsafe),
        'safety_never_entered_lift':sum(not r['entered_lift'] for r in unsafe),
        'safety_lift_smoother_never_ran':sum(r['transition_measurements'].get('reference_updates',0)==0 for r in unsafe),
        'infeasible_joint_counts':dict(Counter(name for r in unsafe for name in r['infeasible_joints'])),
        'safety_without_explicit_joint_offset':sum(not r['infeasible_joints'] for r in unsafe)}
    output={'schema':'capstone_hgwind_contact_transition_audit_v1','run':str(run),
            'summary':counters,'rows':rows,
            'interpretation':'Post-contact TRACK persistence and wrist constraint conflicts are measured; improvement from removing the added dwell remains a hypothesis until physical reruns.',
            'raw_data_retention':'User requested deletion after this compact audit; no HDF5/video/action stream is embedded.'}
    dest=run/'analysis/contact_transition_audit.json';dest.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print('TRANSITION_AUDIT '+json.dumps(counters,ensure_ascii=False));print('REPORT '+str(dest))


if __name__=='__main__':main()
