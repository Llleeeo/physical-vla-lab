"""Numeric precision changes must not hide different modes, signs or inputs."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from capstone_vla import action_precision_control as control
from capstone_vla.ball_action_contract import decode_action,decode_stream,encode_stream,SLICES


def packets():
    result=[]
    for i in range(3):
        a=np.zeros(29,dtype=np.float32);a[SLICES['target_quaternion']]=[1,0,0,0]
        p=decode_action(a,i,[0.]*7)
        p['inputs'].update(target_position=[-3.1234567890123,i/7.,.023456789123],
            target_linear_velocity=[.123456789123,0.,0.],
            target_linear_acceleration=None if i==0 else [0.,0.,0.],
            target_quaternion=[-1.,0.,0.,0.] if i==1 else [1.,0.,0.,0.])
        p['gripper_closed']=i==1;result.append(p)
    return result


def test_precision_restores_original_values_not_widened_float32():
    original=packets();a32=encode_stream(original);a64=control.restore_precision(original,a32)
    assert a64.dtype==np.float64 and np.array_equal(a64.astype(np.float32),a32)
    assert not np.array_equal(a64,a32.astype(np.float64))
    decoded=decode_stream(a64,[0.]*7)
    for p,q in zip(original,decoded):
        assert p['inputs']['target_position']==q['inputs']['target_position']
        assert p['inputs']['target_linear_acceleration']==q['inputs']['target_linear_acceleration']
        assert p['gripper_closed'] is q['gripper_closed']
        assert p['safety_callbacks']==q['safety_callbacks']
    assert all(q['inputs']['target_quaternion']==[1.,0.,0.,0.] for q in decoded)


@pytest.mark.parametrize('kind',['gripper','nullable','gain','position','nan','dtype'])
def test_changed_semantics_or_invalid_labels_rejected(kind):
    p=packets();a=encode_stream(p)
    if kind=='gripper':p[1]['gripper_closed']=False
    elif kind=='nullable':p[1]['inputs']['target_linear_acceleration']=None
    elif kind=='gain':p[0]['inputs']['position_gain']=4.
    elif kind=='position':p[0]['inputs']['target_position'][0]+=.01
    elif kind=='nan':a[0,0]=np.nan
    elif kind=='dtype':a=a.astype(np.float64)
    with pytest.raises(ValueError):control.restore_precision(p,a)


def test_plan_comparison_rejects_hidden_controller_change(tmp_path):
    keys=['seed','attempt_offset','frames','source','source_reference_sha256','task_sha256',
        'physics_sha256','patch_sha256','worker_sha256','controller_config_sha256','urdf_sha256',
        'packet_gain_profile','packet_posture_profile','robot_posture_configuration',
        'cartesian_target_capture_sha256','teacher_replay_sha256','original_depth_bias_m',
        'stage_load_barrier','trace_gripper_drives','history_attempts','initialization_reference_report_sha256']
    base={k:'same' for k in keys};base['adapter_sha256']={}
    for key in ['setup_force_control','gripper_gain_control']:base[key]={'audit':'old','gain':1}
    plan=copy.deepcopy(base)
    for key in ['setup_force_control','gripper_gain_control']:plan[key]['audit']='new'
    control.matched_plan(plan,base)
    plan['gripper_gain_control']['gain']=2
    with pytest.raises(ValueError,match='gripper_gain_control'):control.matched_plan(plan,base)
    plan=copy.deepcopy(base);plan['patch_sha256']='different'
    with pytest.raises(ValueError,match='patch_sha256'):control.matched_plan(plan,base)


@pytest.mark.parametrize('tamper',[False,True])
def test_partial_audit_does_not_confuse_truncation_with_wrong_fields(tmp_path,monkeypatch,tamper):
    actions=control.restore_precision(packets(),encode_stream(packets()))
    decoded=decode_stream(actions,[0.]*7)
    p=decoded[0];event=dict(frame=0,inputs=dict(p['inputs'],**p['safety_callbacks']),
        gripper_closed=False,safety_gripper_closed=False,
        jaw_action={'joint_indices':[7,8],'joint_positions':[.04,.04]})
    if tamper:event['inputs']['target_position']=[99.,0.,0.]
    (tmp_path/'cartesian_packet_execution.json').write_text(json.dumps(dict(
        calls=[event],posture_initialization={'anchor_rad':[0.]*7})))
    (tmp_path/'report.json').write_text(json.dumps(dict(completed_frames=0,
        independent_grasp_evidence={'status':'fail'})))
    monkeypatch.setattr(control,'verify_runtime',lambda plan:actions)
    result=control.audit({'output':str(tmp_path),'frames':3},dict(initialization_verified=True,
        execution_complete=False,grasp='fail',combined_passed=False))
    assert result['issued_prefix_inputs_match'] is (not tamper)
    assert not result['full_execution'] and not result['combined_passed']
    assert result['grasp']=='fail'


def test_missing_episode_remains_not_evaluated(tmp_path,monkeypatch):
    monkeypatch.setattr(control,'verify_runtime',lambda plan:np.zeros((3,29)))
    result=control.audit({'output':str(tmp_path)},dict(initialization_verified=False,
        execution_complete=False,grasp='not_evaluated',combined_passed=False))
    assert result['grasp']=='not_evaluated' and result['errors']


@pytest.mark.parametrize('kind',['original64','widened32','float32','tampered64'])
def test_runtime_rejects_precision_lost_or_changed_labels(tmp_path,monkeypatch,kind):
    baseline=tmp_path/'baseline';baseline.mkdir();out=tmp_path/'current';out.mkdir()
    a32=encode_stream(packets());a64=control.restore_precision(packets(),a32)
    np.save(baseline/'actions.npy',a32)
    for name in ['plan.json','report.json']:(baseline/name).write_text('{}')
    actions=a64.copy()
    if kind=='widened32':actions=a32.astype(np.float64)
    elif kind=='float32':actions=a32
    elif kind=='tampered64':actions[0,0]+=1.e-10
    np.save(out/'actions.npy',actions)
    declaration=dict(schema=control.SCHEMA,baseline=str(baseline),source_dtype='float64',
        comparison_dtype='float32',same_decoder=True,learned_policy=False)
    for name,key in [('plan.json','baseline_plan_sha256'),('actions.npy','baseline_actions_sha256'),
                     ('report.json','baseline_report_sha256')]:
        declaration[key]=control.sha256(baseline/name)
    plan=dict(precision_control=declaration,action_vectors=str(out/'actions.npy'),
        action_vectors_sha256=control.sha256(out/'actions.npy'))
    monkeypatch.setattr(control,'BASELINE',baseline)
    monkeypatch.setattr(control,'matched_plan',lambda a,b:None)
    monkeypatch.setattr(control,'build',lambda p,a:a64)
    if kind=='original64':assert np.array_equal(control.verify_runtime(plan),a64)
    else:
        with pytest.raises(ValueError,match='original float64'):control.verify_runtime(plan)
