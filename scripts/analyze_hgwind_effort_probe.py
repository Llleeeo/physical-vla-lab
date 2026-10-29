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
from capstone_vla.hgwind_effort_probe import initial_state_differences,controller_branch_verified
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


def find_probe_episodes(worker_output):
    # The collector's --out_root is already the trajectory root.
    return [p.parent for p in Path(worker_output).glob('1s1p/*/*/replay.hdf5')]


def analyze(output):
    output=Path(output);plan=json.loads((output/'plan.json').read_text());source=Path(plan['source']);source_meta=json.loads((source.parent/'meta.json').read_text())
    verify_files(source.parent,plan['source_files'])
    episode_paths=find_probe_episodes(output/'worker_output')
    require(len(episode_paths)==1,'Expected exactly one saved physical probe episode')
    ep=episode_paths[0];meta=json.loads((ep/'meta.json').read_text());tag=meta['capstone_effort_probe']
    if plan.get('tracking')=='native-expert':
        if plan.get('independent_packets'):
            require(tag.get('packet_requests_own_motors') is True and tag.get('original_expert_used') is False and not tag['saved_commands_own_motors'],'Explicit packet motor owner changed')
        else:
            require(tag.get('original_expert_used') and not tag['saved_commands_own_motors'],'Original expert command owner changed')
        require(meta['capstone_expert_control']['profile']==source_meta['capstone_expert_control']['profile'] and tag['depth_bias_m']==plan['original_depth_bias_m'],'Original expert control profile changed')
    else:
        require(tag['saved_commands_own_motors'] and not tag['shadow_expert_operational_reference_used']
                and not tag['shadow_expert_velocity_limit_used'],'Unexpected motor owner')
    require(meta['arm_execution']['mode']=='operational_space_inverse_dynamics_effort','Actuator mode changed')
    source_profile=source_meta['robot_limit_profile'];actual_profile=meta['robot_limit_profile']
    require(source_profile['sha256']==actual_profile['sha256'],'Manufacturer limit profile changed')
    n=int(meta['num_steps']);require(n>=2,'Probe saved fewer than two states; inspect simulation.log');callback_count=int(meta.get('arm_execution',{}).get('dynamic_drive',{}).get('physics_callback_count',-1));cadence_ok=callback_count==3*n;full=(n==plan['frames'] and plan['frames']==plan.get('reference_frames',plan['source_frames']))
    task_callbacks=int(meta.get('arm_execution',{}).get('dynamic_drive',{}).get('operational_space_callback_count',0))
    branch_verified=controller_branch_verified(plan.get('tracking','joint'),callback_count,task_callbacks,n,tag,not meta['runtime_safety']['passed'])
    if plan.get('independent_packets'):
        branch_verified=bool(tag.get('packet_requests_own_motors') is True and tag.get('segment_trace_count')==n and task_callbacks==callback_count)
    if plan.get('tracking') in ('logged-command','live-jacobian') and cadence_ok:
        branch_verified=branch_verified and task_callbacks==tag.get('expected_logged_osc_callbacks')
    if plan.get('tracking')=='live-jacobian':
        branch_verified=branch_verified and tag.get('cached_original_jacobians_replayed') is False and tag.get('live_jacobian_control_updates')==tag.get('expected_issued_live_jacobian_updates',-1)
    completed_segments=int(meta.get('arm_execution',{}).get('dynamic_drive',{}).get('completed_segment_count',0))
    full=full and completed_segments==n
    state_keys=['joint_pos','joint_vel','object_pos','object_quat','object_vel','object_ang_vel']
    with h5py.File(source,'r') as old,h5py.File(ep/'replay.hdf5','r') as new:
        initial=initial_state_differences({k:old['obs/'+k][0] for k in state_keys},{k:new['obs/'+k][0] for k in state_keys},source_meta['object_paths'],meta['object_paths'])
        initial_match=all(value<=1e-5 for value in initial.values())
        times=new['timestamp'][:];require(len(times)==n and np.allclose(np.diff(times),1/60,atol=1e-7,rtol=0),'Probe clock changed')
        require(abs(float(meta['sim']['physics_dt'])-1/180)<1e-8,'Physics cadence changed')
        oi=meta['object_paths'].index(source_meta['policy']['roles']['A'])
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
                  'ee_orientation_p95_deg':float(np.rad2deg(np.percentile(orientation_error,95))),
                  'first_position_discrepancy_above_0_01mm_frame':next((i for i,x in enumerate(position_error) if x>.01),None),
                  'position_error_at_frame_90_mm':float(position_error[90]) if comparable>90 else None}
        source_closures=np.flatnonzero((old['actions'][:,7]<.02)&np.r_[True,old['actions'][:-1,7]>=.02])
        source_first_close=int(source_closures[0]) if len(source_closures) else None
        motor_context={'source_first_close_frame':source_first_close,'probe_closed_command_frames':int(np.sum(diag['requested_joints'][:,7]<.02)),
                       'source_operational_space_callbacks':source_meta['arm_execution']['dynamic_drive'].get('operational_space_callback_count'),
                       'probe_operational_space_callbacks':meta['arm_execution']['dynamic_drive'].get('operational_space_callback_count'),
                       'abort_reason':meta['runtime_safety'].get('abort_reason')}
    report={'schema':'physical-vla-lab.hgwind-effort-probe-report.v1','mode':plan['mode'],'reference':plan.get('reference','saved-targets'),'tracking':plan.get('tracking','joint'),'tracking_branch_verified':branch_verified,'source_reference_sha256':plan['source_reference_sha256'],
            'task_sha256':plan['task_sha256'],'execution_fingerprints':{k:plan.get(k) for k in ['worker_sha256','patch_sha256','controller_config_sha256','urdf_sha256','adapter_sha256']},'recorded_frames':n,'completed_frames':completed_segments,'requested_frames':plan['frames'],'full_source_length_completed':None if plan.get('tracking')=='native-expert' else full,'control_budget_exhausted':n>=plan['frames'],'expert_early_success':bool(plan.get('tracking')=='native-expert' and meta.get('success') is True and n<plan['frames']),'physics_callback_count':callback_count,'servo_cadence_passed':cadence_ok,'completed_intervals_cadence_passed':callback_count==3*completed_segments,'attempted_osc_commands':task_callbacks,'rejected_final_osc_command':task_callbacks==callback_count+1 and not meta['runtime_safety']['passed'],'incomplete_control_interval':completed_segments<n,
            'initial_state_differences':initial,'initial_state_match':initial_match,'runtime_safety_passed':meta['runtime_safety']['passed'],
            'independent_grasp_evidence':grasp,'source_state_comparison':tracking,'probe_episode':str(ep),'motor_reference_audit':tag,'control_context':motor_context,'worker_exit_code':json.loads((output/'worker_exit.json').read_text())['exit_code'] if (output/'worker_exit.json').exists() else None,
            'task_feedforward_ablation_passed':None if plan.get('task_feedforward','logged')!='zero' else bool(full and branch_verified and cadence_ok and initial_match and meta['runtime_safety']['passed'] and grasp and grasp['status']=='pass' and tag.get('task_feedforward')=='zero' and tag.get('task_feedforward_reference_count')==tag.get('live_jacobian_control_updates') and tag.get('maximum_applied_task_feedforward_abs')==0 and tag.get('removed_nonzero_task_feedforward_commands')==plan['expected_removed_feedforward_commands']),
            'live_kinematics_component_passed':None if plan.get('tracking')!='live-jacobian' or plan.get('task_feedforward','logged')!='logged' else bool(full and branch_verified and cadence_ok and initial_match and meta['runtime_safety']['passed'] and grasp and grasp['status']=='pass'),
            'control_reference_replay_passed':None if plan.get('tracking')!='logged-command' else bool(plan.get('tracking')=='logged-command' and full and branch_verified and cadence_ok and initial_match and meta['runtime_safety']['passed'] and grasp and grasp['status']=='pass'),
            'native_expert_reference_passed':None if plan.get('tracking')!='native-expert' or plan.get('cartesian_target_capture') else bool(plan.get('tracking')=='native-expert' and branch_verified and cadence_ok and initial_match and meta['runtime_safety']['passed'] and meta.get('success') is True and meta.get('trajectory_validity',{}).get('passed') is True and grasp and grasp['status']=='pass'),
            'motion_tracking_passed':None if plan.get('reference')!='recorded-motion' else bool(plan.get('reference')=='recorded-motion' and full and branch_verified and cadence_ok and initial_match and meta['runtime_safety']['passed'] and grasp and grasp['status']=='pass'),
            'baseline_ready_for_cartesian':None if plan.get('reference')!='saved-targets' else bool(plan.get('reference','saved-targets')=='saved-targets' and plan['mode']=='joint' and full and branch_verified and cadence_ok and initial_match and meta['runtime_safety']['passed'] and grasp and grasp['status']=='pass'),
            'official_benchmark_success_verified':False,'vla_policy':False,'training_eligible':False,
            'limits':['This uses the original effort/limit chain with saved joint-goal references, not reconstructed original operational references.','Shadow task FSM outcomes are not the independent probe result.','A short smoke cannot establish grasp performance.']}
    report['cartesian_target_capture_verified']=None
    if plan.get('trace_cartesian_targets'):
        from capstone_vla.cartesian_target_trace import audit_capture
        capture=json.loads((output/'cartesian_target_inputs.json').read_text())
        segments=json.loads((output/'original_expert_control_references.json').read_text())
        report['cartesian_target_capture_audit']=audit_capture(capture,segments,n)
        with h5py.File(source,'r') as original,h5py.File(ep/'replay.hdf5','r') as captured:
            keys=['actions','timestamp','is_first','is_last','is_terminal']+['obs/'+key for key in original['obs']]
            equality={key:bool(np.array_equal(original[key][:],captured[key][:])) for key in keys}
        report['cartesian_target_capture_audit']['core_array_exact_equality']=equality
        report['cartesian_target_capture_verified']=bool(report['native_expert_reference_passed'] and report['cartesian_target_capture_audit']['capture_alignment_passed'] and all(equality.values()))
    report['cartesian_feedback_replay_passed']=None
    if plan.get('cartesian_target_capture') and not plan.get('independent_packets'):
        from capstone_vla.cartesian_target_replay import audit_reexecution
        from capstone_vla.hf_archive import sha256
        for field in ['cartesian_target_capture','cartesian_target_capture_report','cartesian_target_source_segments']:
            require(sha256(plan[field])==plan[field+'_sha256'],'Upper replay reference changed: '+field)
        capture=json.loads(Path(plan['cartesian_target_capture']).read_text())
        payload=json.loads((output/'cartesian_target_reexecution.json').read_text())
        segments=json.loads((output/'original_expert_control_references.json').read_text())
        replay_audit=audit_reexecution(payload,capture,segments,n)
        with h5py.File(source,'r') as original,h5py.File(ep/'replay.hdf5','r') as captured:
            keys=['actions','timestamp','is_first','is_last','is_terminal']+['obs/'+key for key in original['obs']]
            replay_audit['core_array_exact_equality']={key:bool(np.array_equal(original[key][:],captured[key][:])) for key in keys}
        report['cartesian_feedback_replay_audit']=replay_audit
        report['cartesian_feedback_replay_passed']=bool(replay_audit['reexecution_verified'] and branch_verified
            and cadence_ok and task_callbacks==3*len(payload['calls']) and completed_segments==n and initial_match and meta['runtime_safety']['passed']
            and tag.get('cartesian_upper_target_reexecution') is True and tag.get('saved_inner_control_outputs_used') is False
            and grasp and grasp['status']=='pass')
        report['limits']=['Recorded upper target/motion/settings schedule, native expert call timing, live safety callbacks, gripper and joint-only terminal fallback retained.',
            'Inner arm/OSC commands regenerated by current-state native feedback; no saved inner references execute.',
            'One exposed expert diagnostic; not a standalone VLA policy, full benchmark certification or generalization result.']
    report['packet_controller_probe_passed']=None
    if plan.get('independent_packets'):
        from capstone_vla.cartesian_packets import build_packets,audit_packets
        from capstone_vla.hf_archive import sha256
        for field in ['cartesian_target_capture','packet_baseline_report']:
            require(sha256(plan[field])==plan[field+'_sha256'],'Packet reference changed: '+field)
        capture=json.loads(Path(plan['cartesian_target_capture']).read_text())
        with h5py.File(source,'r') as original:
            expected_anchor = None
            if plan.get('packet_posture_profile')=='latched':
                from capstone_vla.posture_anchor import compute_anchor
                expected_anchor=compute_anchor(original['obs/joint_pos'][0,:7],plan['robot_posture_configuration'])
            packets=build_packets(capture,original['actions'][:,7:],plan.get('packet_gain_profile','recorded'),plan.get('packet_posture_profile','recorded'),expected_anchor)
        if plan.get('action_contract'):
            from capstone_vla.ball_action_contract import decode_stream
            require(sha256(plan['action_vectors'])==plan['action_vectors_sha256'],'Action vector labels changed')
            packets=decode_stream(np.load(plan['action_vectors'],allow_pickle=False),expected_anchor)
        require(json.loads((output/'packet_requests.json').read_text())==packets,'Executed packet stream differs from expected decoded inputs')
        payload=json.loads((output/'cartesian_packet_execution.json').read_text())
        segments=json.loads((output/'original_expert_control_references.json').read_text())
        require(payload.get('packet_gain_profile','recorded')==plan.get('packet_gain_profile','recorded'),'Executed gain profile differs from plan')
        require(payload.get('packet_posture_profile','recorded')==plan.get('packet_posture_profile','recorded'),'Executed posture profile differs from plan')
        packet_audit=audit_packets(payload,packets,segments,n)
        report['packet_controller_audit']=packet_audit
        report['full_source_length_completed']=full
        report['expert_early_success']=False
        report['packet_controller_probe_passed']=bool(packet_audit['packet_execution_verified'] and full and branch_verified
            and cadence_ok and completed_segments==n and initial_match and meta['runtime_safety']['passed']
            and grasp and grasp['status']=='pass')
        report['limits']=['Recorded numeric pose/motion/controller-settings/gripper stream remains the request source; not VLA predictions.',
            'Online task and skill decisions bypassed; native initialized robot runtime and geometry-only skill helpers retained.',
            'Independent safety uses non-target scene geometry and robot state; target pose/velocity access blocked in its world adapter.',
            'Last Cartesian goal is explicitly held after the 194 source calls; no expert joint-only fallback executes.',
            'Collector expert/FSM success labels are not this probe result. Use the independent contact/lift/retention outcome.']
    report['packet_fixed_gains_probe_passed']=None
    if plan.get('packet_gain_profile','recorded')!='recorded' and plan.get('packet_posture_profile','recorded')=='recorded':
        from capstone_vla.cartesian_packets import audit_gain_profile
        for key in ['packet_gain_reference_report','packet_gain_reference_requests']:
            require(sha256(plan[key])==plan[key+'_sha256'],'Gain reference changed: '+key)
        source_packets=json.loads(Path(plan['packet_gain_reference_requests']).read_text())
        isolated=audit_gain_profile(source_packets,packets,plan['packet_gain_profile'])
        report['packet_gain_profile']=plan['packet_gain_profile']
        report['only_pose_gains_changed']=isolated
        report['changed_gain_frames']=sum(a!=b for a,b in zip(source_packets,packets))
        report['packet_fixed_gains_probe_passed']=bool(isolated and report['packet_controller_probe_passed'])
        report['limits'][0]='Only pose gains are constant at 9/8; all other recorded numeric settings/motion/gripper schedules remain. Not VLA predictions.'
    report['packet_measured_posture_probe_passed']=None
    if plan.get('packet_posture_profile') in ('measured','latched') and not plan.get('action_contract'):
        from capstone_vla.cartesian_packets import audit_posture_profile
        for key in ['packet_posture_reference_report','packet_posture_reference_requests']:
            require(sha256(plan[key])==plan[key+'_sha256'],'Posture reference changed: '+key)
        source_packets=json.loads(Path(plan['packet_posture_reference_requests']).read_text())
        isolated=audit_posture_profile(source_packets,packets,plan['packet_posture_profile'],expected_anchor)
        posture_probe_passed=bool(isolated and packet_audit['posture_profile_verified'] and report['packet_controller_probe_passed'])
        report.update(packet_posture_profile=plan['packet_posture_profile'],packet_gain_profile=plan['packet_gain_profile'],only_posture_target_changed=isolated,
            changed_posture_frames=sum(a!=b for a,b in zip(source_packets,packets)),
            packet_measured_posture_probe_passed=posture_probe_passed if plan['packet_posture_profile']=='measured' else None)
        report['limits'][0]='Only posture handling differs from the passing fixed-9/8 baseline; other recorded motion/settings/gripper schedules remain. Not VLA predictions.'
    report['packet_latched_posture_probe_passed']=None
    if plan.get('packet_posture_profile')=='latched' and not plan.get('action_contract'):
        initialization=payload['posture_initialization']
        require(initialization['configuration']==plan['robot_posture_configuration'],'Executed posture configuration differs from plan')
        with h5py.File(ep/'replay.hdf5','r') as recorded:
            initial_q=recorded['obs/joint_pos'][0,:7]
            robot_q=np.array([e['latched_posture_context']['measured_joint_position_rad'] for e in payload['calls']])
            robot_context_matches=bool(np.array_equal(initial_q,np.asarray(initialization['initial_joint_position_rad'])) and np.array_equal(robot_q,recorded['obs/joint_pos'][:,:7]))
        report['packet_latched_posture_probe_passed']=bool(posture_probe_passed and robot_context_matches)
        report['packet_measured_posture_probe_passed']=None
        report['latched_posture_matches_raw_robot_context']=robot_context_matches
        report['limits'][0]='Posture is initialized from measured robot joints and declared sampled robot variation, then held; target motion/other settings remain recorded. Not VLA predictions.'
    report['ball_action_contract_passed']=None
    if plan.get('action_contract'):
        from capstone_vla.ball_action_contract import audit_contract
        contract_audit=audit_contract(payload,np.load(plan['action_vectors'],allow_pickle=False))
        with h5py.File(ep/'replay.hdf5','r') as recorded:
            raw_q=recorded['obs/joint_pos'][:,:7]
        contexts=np.asarray([e['latched_posture_context']['measured_joint_position_rad'] for e in payload['calls']])
        robot_context_matches=bool(np.array_equal(raw_q,contexts) and np.array_equal(
            raw_q[0],np.asarray(payload['posture_initialization']['initial_joint_position_rad'])))
        contract_audit['robot_context_matches_raw_recording']=robot_context_matches
        report['ball_action_contract_audit']=contract_audit
        report['ball_action_contract_passed']=bool(report['packet_controller_probe_passed']
            and all(contract_audit[k] for k in ['decoded_actions_match_motor_boundary',
                'fixed_collision_callbacks_always_enabled','live_posture_verified',
                'robot_context_matches_raw_recording']))
        report['limits']=[
            'Expert-labelled 29D float32 command surrogate, not a trained model or a learned-policy success.',
            'Absolute world pose, velocity, acceleration, relative-motion references and explicit control modes; no hidden per-frame settings.',
            'Fixed 9/8 gains, live-initialized declared robot posture, always-on non-target collision callbacks and unchanged native hard limits.',
            'A single exposed initial condition; validate 3-5 different development episodes before cohort conversion/training.',
            'Original collector/FSM labels remain untouched; independent contact/lift/retention checks decide this engineering gate.']
        print('BALL_ACTION_CONTRACT_PASSED '+str(report['ball_action_contract_passed']),flush=True)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('HGWIND_EFFORT_PROBE_COMPLETED '+json.dumps({k:report[k] for k in ['mode','reference','tracking','tracking_branch_verified','recorded_frames','completed_frames','full_source_length_completed','initial_state_match','motion_tracking_passed','native_expert_reference_passed','control_reference_replay_passed','live_kinematics_component_passed','task_feedforward_ablation_passed','cartesian_target_capture_verified','cartesian_feedback_replay_passed','packet_controller_probe_passed','packet_fixed_gains_probe_passed','packet_measured_posture_probe_passed','packet_latched_posture_probe_passed','expert_early_success','baseline_ready_for_cartesian']}))
    print('INDEPENDENT_GRASP '+str(None if grasp is None else grasp['status']));print('REPORT '+str(output/'report.json'))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);analyze(parser.parse_args().output)
