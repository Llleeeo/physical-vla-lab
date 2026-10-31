#!/usr/bin/env python3
"""Independent live-chain audit, separate from the unused expert FSM outcome."""
import json
from pathlib import Path
import sys
import h5py
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from analyze_hgwind_effort_probe import control_contact_arrays,find_probe_episodes
from capstone_vla.hgwind_effort_probe import initial_state_differences
from capstone_vla.hgwind_development_data import states_from_robot
from capstone_vla.ball_live_protocol import decoded_request
from capstone_vla.ball_action_contract import native_inputs_match,normalization_audit_copy
from capstone_vla.cartesian_packets import audit_packets
from capstone_vla.grasp_evidence import GraspEvidenceCriteria,evaluate_grasp
from capstone_vla.hf_archive import sha256
from capstone_vla.ball_smoothing import smooth_actions


def analyze(out,exit_code):
    out=Path(out);cfg=json.loads((out/'run_config.json').read_text())
    result=dict(status='BALL_RECORDED_INCOMPLETE',requested_frames=cfg['frames'],worker_exit_code=exit_code,
        learned_policy_evaluated=False,source_actions_used_for_control=False,heldout_evaluation=False,
        real_time_evaluated=False,official_benchmark_success=None,independent_grasp=None,
        scope='Exposed training-case synchronous live integration; not a benchmark success-rate estimate')
    try:
        diagnostic=True
        result.update(source_actions_used_for_control=True,standalone_policy_evaluation=False,diagnostic_only=True,
            scope='Recorded-model command reexecution, not live policy evaluation')
        if cfg.get('model_inference_used') is not False or cfg.get('learned_policy_evaluated') is not False:
            raise ValueError('Invalid diagnostic provenance')
        if sha256(cfg['recorded_actions_path'])!=cfg['recorded_actions_sha256']:raise ValueError('Action file changed')
        with np.load(cfg['recorded_actions_path'],allow_pickle=False) as saved:
            expected_actions=smooth_actions(saved['actions'],cfg['smoothing_alpha'])
        for path,digest in cfg['runtime_hashes'].items():
            if sha256(path)!=digest:raise ValueError('Runtime changed: '+path)
        eps=find_probe_episodes(out/'worker_output')
        if len(eps)!=1:raise ValueError('Expected one saved episode; inspect simulation.log')
        ep=eps[0];meta=json.loads((ep/'meta.json').read_text())
        tag=meta['capstone_effort_probe']
        if abs(float(meta['sim']['physics_dt'])-1/180)>1e-8 or float(meta['control_hz'])!=60:
            raise ValueError('Native physics/control cadence changed')
        if tag.get('vla_policy') is not False or not tag.get('recorded_model_actions') or tag.get('source_actions_used_for_control') is not True:
            raise ValueError('Motor provenance mismatch')
        original=json.loads(Path(cfg['source']).with_name('meta.json').read_text())
        if meta['robot_limit_profile']['sha256']!=original['robot_limit_profile']['sha256']:raise ValueError('Native motion limits differ')
        payload=json.loads((out/'cartesian_packet_execution.json').read_text())
        packets=json.loads((out/'packet_requests.json').read_text())
        rows=json.loads((out/'live_control_references.json').read_text())
        timing=json.loads((out/'policy_timing.json').read_text())
        result['learned_policy_evaluated']=bool(timing and payload.get('learned_policy_evaluated'))
        native=meta['arm_execution']['dynamic_drive'];complete=int(native['completed_segment_count'])
        callbacks=int(native['physics_callback_count']);osc=int(native['operational_space_callback_count'])
        result.update(recorded_frames=meta['num_steps'],completed_frames=complete,physics_callbacks=callbacks,
            runtime_safety_passed=meta['runtime_safety']['passed'],abort_reason=meta['runtime_safety'].get('abort_reason'),
            original_collector_success=meta.get('success'),probe_episode=str(ep))
        with h5py.File(ep/'replay.hdf5','r') as actual,h5py.File(cfg['source'],'r') as source,h5py.File(out/'live_observations.hdf5','r') as obs:
            keys=['joint_pos','joint_vel','object_pos','object_quat','object_vel','object_ang_vel']
            diffs=initial_state_differences({k:source['obs/'+k][0] for k in keys},
                {k:actual['obs/'+k][0] for k in keys},original['object_paths'],meta['object_paths'])
            initial_ok=all(v<=1e-5 for v in diffs.values());result.update(initial_state_differences=diffs,initial_state_match=initial_ok)
            n=len(actual['timestamp']);m=len(obs['state']);actions=obs['action'][:]
            anchor=payload['posture_initialization']['anchor_rad']
            measured=states_from_robot(actual['obs/joint_pos'][:],actual['obs/joint_vel'][:],actual['obs/ee_pos'][:],actual['obs/ee_quat'][:],anchor,np.asarray(cfg['pinch_offset_local_m']))
            if m!=n:raise ValueError('Saved live and physics frame counts differ')
            np.testing.assert_array_equal(obs['state'][:],measured)
            np.testing.assert_allclose(obs['timestamp_s'][:],np.arange(n)/60,rtol=0,atol=1e-9)
            np.testing.assert_allclose(actual['timestamp'][:],np.arange(n)/60,rtol=0,atol=1e-7)
            if any(len(obs[name])!=n for name in cfg['cameras']):raise ValueError('Missing live RGB')
            if len(timing)!=n:raise ValueError('Incomplete policy timing records')
            # Exact binding to one saved model stream and declared causal transform.
            np.testing.assert_array_equal(actions,expected_actions[:n])
            for i,event in enumerate(timing):
                if event['frame']!=i or event['anchor_frame']!=i:raise ValueError('Recorded command timing changed')
                expected=decoded_request(actions[i],i,anchor)
                if i>=len(packets) or not native_inputs_match(packets[i]['inputs'],expected['inputs']):raise ValueError('Consumed action differs from native request')
            adjusted=normalization_audit_copy(payload,packets)
            audit=audit_packets(adjusted,packets,rows,n)
            audit['learned_policy_evaluated']=not diagnostic
            audit['recorded_model_reexecution']=True
            assistance=np.load(out/'command_diagnostics.npz',allow_pickle=False)['assistance']
            if assistance.shape!=(n,3) or assistance.any():raise ValueError('Unexpected/missing grasp assistance record')
            target=original['policy']['roles']['A'];idx=meta['object_paths'].index(target)
            left,right=control_contact_arrays(actual,target,actual['timestamp'][:])
            params=original['success_parameters']['1']
            criteria=GraspEvidenceCriteria(minimum_bilateral_duration_s=float(params['hold_duration_s']),final_hold_duration_s=float(params['hold_duration_s']),minimum_lift_m=float(params['Dh_min']),maximum_relative_position_jitter_m=float(params['r_hold']),minimum_precontact_speed_m_s=float(params['v_eps']))
            evidence=evaluate_grasp(dict(timestamp_s=actual['timestamp'][:],ee_pos=actual['obs/ee_pos'][:],ee_quat=actual['obs/ee_quat'][:],object_pos=actual['obs/object_pos'][:,idx:idx+1],object_vel=actual['obs/object_vel'][:,idx:idx+1],contact_L=left>1e-12,contact_R=right>1e-12,impulse_L=left,impulse_R=right,grasp_assistance_audit=assistance),criteria)
            result['partial_contact_evidence']=evidence
            if cfg['frames']==269 and n==269:result['independent_grasp']=evidence
            good=(exit_code==0 and n==m==complete==cfg['frames'] and callbacks==osc==3*n
                  and meta['runtime_safety']['passed'] and initial_ok and audit['packet_execution_verified']
                  and payload['learned_policy_evaluated'] is (not diagnostic)
                  and payload['source_actions_used_for_control'] is diagnostic
                  and not payload['source_motion_and_gripper_stream_retained'])
            result.update(live_state_alignment_verified=True,saved_command_binding_verified=True,
                packet_audit=audit,full_requested_integration_passed=bool(good))
            result['smoothing_alpha']=cfg['smoothing_alpha']
            error=np.linalg.norm(actions[:,:3]-measured[:,14:17],axis=1)*1000
            result['command_to_measured_pinch_error_mm']=dict(mean=float(error.mean()),p95=float(np.percentile(error,95)))
            result['gripper_transition_frames']=(np.flatnonzero(np.diff(actions[:,28]>=.5))+1).tolist()
            result['finger_contact_frames']=dict(left=np.flatnonzero(left>1e-12).tolist(),right=np.flatnonzero(right>1e-12).tolist())
            if good:result['status']='BALL_RECORDED_INTEGRATION_PASS'
        result['artifacts']={str(p.relative_to(out)):sha256(p) for p in [ep/'replay.hdf5',ep/'meta.json',out/'live_observations.hdf5',out/'policy_timing.json']}
    except Exception as exc:
        result['analysis_error']=type(exc).__name__+': '+str(exc)
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result['status']+' report='+str(out/'report.json'),flush=True)
    return result
