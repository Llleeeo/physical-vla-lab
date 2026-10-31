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
from capstone_vla.ball_vertical_diagnostic import audit_height, validate_provenance
from capstone_vla.ball_approach_follow import make_approach_guard
from capstone_vla.ball_visual_geometry import locate_ball
from capstone_vla.ball_near_capture import locate_for_plan


# Named prior analyzers are retained in commits 3fadd5a and 469fd68.
# Controller hashes must still match; only post-run numerical verification changed.
LEGACY_APPROACH_ANALYZER_SHA256 = '36291fdf6055aa9f7470bd6ba57177df68578a20b0bd46afe1e5c441e10dcbb8'
NEAR_CAPTURE_ANALYZER_SHA256 = "20f7bbcbd59a7ab609a138fda94a48fce0ea6052faf13e026b19c55afae65a44"
APPROACH_AUDIT_ATOL = 1e-12
NEAR_CAPTURE_AUDIT_ATOL = 1e-10


def audit_approach_value(expected, recorded, path='decision', atol=APPROACH_AUDIT_ATOL):
    """Discrete decisions exact; finite floating geometry tolerates roundoff only."""
    if type(expected) is not type(recorded):raise ValueError('Approach type mismatch: '+path)
    if isinstance(expected, dict):
        if set(expected)!=set(recorded):raise ValueError('Approach keys mismatch: '+path)
        for key in expected:audit_approach_value(expected[key],recorded[key],path+'/'+key,atol)
    elif isinstance(expected, list):
        if len(expected)!=len(recorded):raise ValueError('Approach length mismatch: '+path)
        for i,(a,b) in enumerate(zip(expected,recorded)):audit_approach_value(a,b,path+'/'+str(i),atol)
    elif isinstance(expected, float):
        if not (np.isfinite(expected) and np.isfinite(recorded) and abs(expected-recorded)<=atol):
            raise ValueError('Approach numeric mismatch: '+path)
    elif expected!=recorded:raise ValueError('Approach decision mismatch: '+path)


def verify_analysis_revision(path, digest, result):
    actual=sha256(path)
    if actual==digest:return
    if Path(path).resolve()==Path(__file__).resolve() and digest in (LEGACY_APPROACH_ANALYZER_SHA256,NEAR_CAPTURE_ANALYZER_SHA256):
        result['posthoc_analyzer_correction']=dict(original_sha256=digest,current_sha256=actual,
            reason='Numerical audit only: 1e-12 legacy, 1e-10 near-capture nonlinear fitting; decisions/jaw/provenance exact')
        return
    raise ValueError('Runtime changed: '+str(path))


def analyze(out,exit_code):
    out=Path(out);cfg=json.loads((out/'run_config.json').read_text())
    result=dict(status='BALL_LIVE_INCOMPLETE',requested_frames=cfg['frames'],worker_exit_code=exit_code,
        learned_policy_evaluated=False,source_actions_used_for_control=False,heldout_evaluation=False,
        real_time_evaluated=False,official_benchmark_success=None,independent_grasp=None,
        scope='Exposed training-case synchronous live integration; not a benchmark success-rate estimate')
    try:
        diagnostic=validate_provenance(cfg)
        approach_plan=cfg.get('approach_guard');guard=make_approach_guard(approach_plan)
        approach_atol=NEAR_CAPTURE_AUDIT_ATOL if approach_plan and 'near_capture' in approach_plan else APPROACH_AUDIT_ATOL
        if guard is not None:
            if diagnostic:raise ValueError('Combined approach/height intervention')
            result.update(visual_guidance=True,policy_postprocessing=True,unmodified_model_actions=False,
                simulator_target_state_used=False,scope='Exposed VLA+RGB-guided approach filter; not unmodified-policy or certified collision avoidance')
            guard_events=json.loads((out/'approach_guard_events.json').read_text())
        result.update(source_actions_used_for_control=diagnostic,standalone_policy_evaluation=not diagnostic and guard is None,
            diagnostic_only=diagnostic)
        if diagnostic:
            result['scope']='Recorded-height intervention with live model feedback; any grasp is hybrid diagnostic success, not unassisted VLA success'
            plan=cfg['vertical_diagnostic']
            if sha256(plan['reference_payload'])!=plan['reference_payload_sha256']:raise ValueError('Height reference payload changed')
            with np.load(plan['reference_payload'],allow_pickle=False) as reference:
                np.testing.assert_array_equal(plan['target_z_m'],reference['actions'][plan['start_frame']:plan['stop_frame_exclusive'],2])
        for path,digest in cfg['runtime_hashes'].items():
            verify_analysis_revision(path,digest,result)
        eps=find_probe_episodes(out/'worker_output')
        if len(eps)!=1:raise ValueError('Expected one saved episode; inspect simulation.log')
        ep=eps[0];meta=json.loads((ep/'meta.json').read_text())
        tag=meta['capstone_effort_probe']
        if abs(float(meta['sim']['physics_dt'])-1/180)>1e-8 or float(meta['control_hz'])!=60:
            raise ValueError('Native physics/control cadence changed')
        if not tag.get('vla_policy') or tag.get('source_actions_used_for_control') is not diagnostic:
            raise ValueError('Motor provenance mismatch')
        original=json.loads(Path(cfg['source']).with_name('meta.json').read_text())
        if meta['robot_limit_profile']['sha256']!=original['robot_limit_profile']['sha256']:raise ValueError('Native motion limits differ')
        payload=json.loads((out/'cartesian_packet_execution.json').read_text())
        if guard is not None and (payload.get('approach_guard')!=approach_plan or tag.get('approach_guard')!=approach_plan):
            raise ValueError('Approach provenance mismatch')
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
            if guard is not None:
                if len(guard_events)!=n:raise ValueError('Incomplete approach decisions')
                if json.loads(obs.attrs['approach_guard_json'])!=approach_plan:raise ValueError('Approach archive plan mismatch')
                np.testing.assert_allclose(obs['robot_finger_points'][:],actual['obs/finger_segment_world'][:],rtol=0,atol=1e-5)
                camera_order=json.loads(obs.attrs['camera_pose_order_json'])
                optics={c['name']:c['intrinsics'] for c in approach_plan['cameras']}
            # Bind consumed commands to persisted model IPC responses, never source actions.
            for i,event in enumerate(timing):
                origin=(i//cfg['prefix'])*cfg['prefix']
                if event['frame']!=i or event['anchor_frame']!=origin:raise ValueError('Queue timing changed')
                if i==origin:
                    from capstone_vla.ball_live_protocol import audit_request_history
                    with np.load(out/'ipc'/f'request_{origin:06d}.npz',allow_pickle=False) as request:
                        ids=[max(0,origin-2),origin]
                        audit_request_history(request,origin,np.stack([obs['state'][j] for j in ids]),
                            {name:np.stack([obs[name][j] for j in ids]) for name in cfg['cameras']})
                with np.load(out/'ipc'/f'response_{origin:06d}.npz',allow_pickle=False) as response:
                    if int(response['frame'])!=origin:raise ValueError('Response index mismatch')
                    raw=response['absolute'][i-origin]
                    if diagnostic:
                        np.testing.assert_array_equal(obs['model_action'][i],raw)
                        audit_height(raw,actions[i],i,cfg['vertical_diagnostic'],cfg['frames'])
                    elif guard is not None:
                        np.testing.assert_array_equal(obs['model_action'][i],raw)
                        detection=locate_for_plan([obs[name][i] for name in camera_order],obs['camera_pose_world'][i],
                            [optics[name] for name in camera_order],camera_order,approach_plan)
                        expected_action,event=guard.apply(raw,i,obs['state'][i,14:21],obs['robot_finger_points'][i],
                            obs['robot_finger_corners'][i],detection)
                        np.testing.assert_allclose(actions[i],expected_action,rtol=0,atol=approach_atol)
                        np.testing.assert_array_equal(actions[i,28],raw[28])
                        untouched=np.flatnonzero(expected_action==raw)
                        np.testing.assert_array_equal(actions[i,untouched],raw[untouched])
                        audit_approach_value(dict(event,detection=detection),guard_events[i],atol=approach_atol)
                    else:np.testing.assert_array_equal(actions[i],raw)
                expected=decoded_request(actions[i],i,anchor)
                if i>=len(packets) or not native_inputs_match(packets[i]['inputs'],expected['inputs']):raise ValueError('Consumed action differs from native request')
                if packets[i]['gripper_closed'] is not expected['gripper_closed']:raise ValueError('Consumed jaw differs from native request')
            if guard is not None:
                result['approach_audit']=dict(verified=True,raw_model_responses_bound=True,jaw_scores_unchanged=True,
                    visual_valid_frames=sum(e['vision_valid'] for e in guard_events),
                    modes={mode:sum(e['mode']==mode for e in guard_events) for mode in set(e['mode'] for e in guard_events)},
                    first_changed_frame=next((e['frame'] for e in guard_events if e['changed_channels']),None),
                    visual_follow_frames=sum(e.get('follow',{}).get('active',False) for e in guard_events))
            adjusted=normalization_audit_copy(payload,packets)
            audit=audit_packets(adjusted,packets,rows,n)
            audit['learned_policy_evaluated']=not diagnostic
            audit['recorded_height_intervention']=diagnostic
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
            result.update(live_state_alignment_verified=True,model_response_binding_verified=True,
                packet_audit=audit,full_requested_integration_passed=bool(good))
            if diagnostic:
                plan=cfg['vertical_diagnostic'];raw=obs['model_action'][:]
                result['vertical_intervention_audit']=dict(verified=True,channel=2,
                    start_frame=plan['start_frame'],stop_frame_exclusive=plan['stop_frame_exclusive'],
                    changed_frames=np.flatnonzero(actions[:,2]!=raw[:,2]).tolist(),
                    other_channels_exact=True,raw_model_responses_bound=True)
            if good:result['status']='BALL_VERTICAL_DIAGNOSTIC_PASS' if diagnostic else 'BALL_LIVE_INTEGRATION_PASS'
        artifacts=[ep/'replay.hdf5',ep/'meta.json',out/'live_observations.hdf5',out/'policy_timing.json']
        if guard is not None:artifacts.append(out/'approach_guard_events.json')
        result['artifacts']={str(p.relative_to(out)):sha256(p) for p in artifacts}
    except Exception as exc:
        result['analysis_error']=type(exc).__name__+': '+str(exc)
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result['status']+' report='+str(out/'report.json'),flush=True)
    return result
