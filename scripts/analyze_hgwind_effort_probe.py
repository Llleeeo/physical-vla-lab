#!/usr/bin/env python3
"""Audit a completed command probe; never relabel its shadow-FSM task outcome."""
import argparse
import json
from pathlib import Path
import sys
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from capstone_vla.hgwind_effort_probe import initial_state_differences
from capstone_vla.hgwind_cohort import require,verify_files
from capstone_vla.grasp_evidence import GraspEvidenceCriteria,evaluate_grasp


def control_contact_arrays(handle,target,times):
    c=handle['evidence/contacts'];paths=c['paths'].asstr()[:].tolist()
    headers,points=c['headers'][:],c['points'][:];step=c['step_time'][:];dt=c['step_dt'][:]
    bins=np.searchsorted(times,step-dt/2,side='right')-1
    require(np.all((bins>=0)&(bins<len(times))),'Raw contact times outside episode')
    target_ids={i for i,p in enumerate(paths) if p==target or p.startswith(target+'/')}
    impulses=[]
    for name in ['panda_leftfinger','panda_rightfinger']:
        ids={i for i,p in enumerate(paths) if '/'+name in p};tick=np.zeros(len(step))
        for h in headers:
            if h[7]>0 and ((h[1] in target_ids and h[2] in ids) or (h[2] in target_ids and h[1] in ids)):
                tick[int(h[0])]+=np.linalg.norm(points[int(h[6]):int(h[6]+h[7]),6:9],axis=1).sum()
        control=np.zeros(len(times));np.add.at(control,bins,tick);impulses.append(control)
    return impulses


def analyze(output):
    output=Path(output);plan=json.loads((output/'plan.json').read_text());source=Path(plan['source']);source_meta=json.loads((source.parent/'meta.json').read_text())
    verify_files(source.parent,plan['source_files'])
    episode_paths=[p.parent for p in (output/'worker_output/Trajectory/1s1p').glob('*/**/replay.hdf5')]
    require(len(episode_paths)==1,'Expected exactly one saved physical probe episode')
    ep=episode_paths[0];meta=json.loads((ep/'meta.json').read_text());tag=meta['capstone_effort_probe']
    require(tag['saved_commands_own_motors'] and not tag['shadow_expert_operational_reference_used']
            and not tag['shadow_expert_velocity_limit_used'],'Unexpected motor owner')
    require(meta['arm_execution']['mode']=='operational_space_inverse_dynamics_effort','Actuator mode changed')
    source_profile=source_meta['robot_limit_profile'];actual_profile=meta['robot_limit_profile']
    require(source_profile['sha256']==actual_profile['sha256'],'Manufacturer limit profile changed')
    n=int(meta['num_steps']);require(n>=2,'Probe saved fewer than two states; inspect simulation.log');callback_count=int(meta.get('arm_execution',{}).get('dynamic_drive',{}).get('physics_callback_count',-1));cadence_ok=callback_count==3*n;full=(n==plan['frames'] and plan['frames']==plan['source_frames'])
    state_keys=['joint_pos','joint_vel','object_pos','object_quat','object_vel','object_ang_vel']
    with h5py.File(source,'r') as old,h5py.File(ep/'replay.hdf5','r') as new:
        initial=initial_state_differences({k:old['obs/'+k][0] for k in state_keys},{k:new['obs/'+k][0] for k in state_keys},source_meta['object_paths'],meta['object_paths'])
        initial_match=all(value<=1e-5 for value in initial.values())
        times=new['timestamp'][:];require(len(times)==n and np.allclose(np.diff(times),1/60,atol=1e-7,rtol=0),'Probe clock changed')
        require(abs(float(meta['sim']['physics_dt'])-1/180)<1e-8,'Physics cadence changed')
        oi=meta['object_paths'].index(source_meta['policy']['roles']['A']);old_oi=source_meta['object_paths'].index(source_meta['policy']['roles']['A'])
        left,right=control_contact_arrays(new,source_meta['policy']['roles']['A'],times)
        diag=np.load(output/'command_diagnostics.npz');assistance=diag['assistance']
        require(assistance.shape==(n,3) and not assistance.any(),'Missing/nonzero assistance evidence')
        params=source_meta['success_parameters']['1']
        criteria=GraspEvidenceCriteria(minimum_bilateral_duration_s=float(params['hold_duration_s']),final_hold_duration_s=float(params['hold_duration_s']),
                    minimum_lift_m=float(params['Dh_min']),maximum_relative_position_jitter_m=float(params['r_hold']),minimum_precontact_speed_m_s=float(params['v_eps']))
        data={'timestamp_s':times,'ee_pos':new['obs/ee_pos'][:],'ee_quat':new['obs/ee_quat'][:],
              'object_pos':new['obs/object_pos'][:,oi:oi+1],'object_vel':new['obs/object_vel'][:,oi:oi+1],
              'contact_L':left>1e-12,'contact_R':right>1e-12,'impulse_L':left,'impulse_R':right,'grasp_assistance_audit':assistance}
        grasp=evaluate_grasp(data,criteria) if n>=2 else None
        comparable=min(n,len(old['timestamp']));position_error=np.linalg.norm(new['obs/ee_pos'][:comparable]-old['obs/ee_pos'][:comparable],axis=1)*1000
        orientation_error=(Rotation.from_quat(new['obs/ee_quat'][:comparable][:,[1,2,3,0]])*Rotation.from_quat(old['obs/ee_quat'][:comparable][:,[1,2,3,0]]).inv()).magnitude()
        tracking={'ee_position_mean_mm':float(position_error.mean()),'ee_position_p95_mm':float(np.percentile(position_error,95)),
                  'ee_orientation_p95_deg':float(np.rad2deg(np.percentile(orientation_error,95)))}
    report={'schema':'physical-vla-lab.hgwind-effort-probe-report.v1','mode':plan['mode'],'source_reference_sha256':plan['source_reference_sha256'],
            'task_sha256':plan['task_sha256'],'execution_fingerprints':{k:plan[k] for k in ['worker_sha256','patch_sha256','controller_config_sha256','urdf_sha256']},'completed_frames':n,'requested_frames':plan['frames'],'full_source_length_completed':full,'physics_callback_count':callback_count,'servo_cadence_passed':cadence_ok,
            'initial_state_differences':initial,'initial_state_match':initial_match,'runtime_safety_passed':meta['runtime_safety']['passed'],
            'independent_grasp_evidence':grasp,'source_state_comparison':tracking,'probe_episode':str(ep),'motor_reference_audit':tag,
            'baseline_ready_for_cartesian':bool(plan['mode']=='joint' and full and cadence_ok and initial_match and meta['runtime_safety']['passed'] and grasp and grasp['status']=='pass'),
            'official_benchmark_success_verified':False,'vla_policy':False,'training_eligible':False,
            'limits':['This uses the original effort/limit chain with saved joint-goal references, not reconstructed original operational references.','Shadow task FSM outcomes are not the independent probe result.','A short smoke cannot establish grasp performance.']}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('HGWIND_EFFORT_PROBE_COMPLETED '+json.dumps({k:report[k] for k in ['mode','completed_frames','full_source_length_completed','initial_state_match','baseline_ready_for_cartesian']}))
    print('INDEPENDENT_GRASP '+str(None if grasp is None else grasp['status']));print('REPORT '+str(output/'report.json'))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);analyze(parser.parse_args().output)
