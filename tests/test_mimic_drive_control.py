"""Mimic control changes one native drive only; never count a missing run as failure."""
import copy
import json
from pathlib import Path
import h5py
import numpy as np
import pytest
from capstone_vla import mimic_drive_control as control
from capstone_vla import gripper_gain_control as gain
from test_gripper_gain_control import rig


def fingers():
    first='/Root/Dynamics/franka/panda_hand/panda_finger_joint1'
    key='physxMimicJoint:rotX:'
    return [dict(path=first,schemas=['PhysicsDriveAPI:linear'],attributes={},relationships={}),
        dict(path=first[:-1]+'2',schemas=['PhysxMimicJointAPI:rotX'],
             attributes={key+'gearing':-1.,key+'offset':0.,key+'referenceJointAxis':'rotX'},
             relationships={key+'referenceJoint':[first]})]


@pytest.mark.parametrize('bad',[None,'missing','reference','gearing','leader_mimic'])
def test_real_setter_requires_known_mimic_and_preserves_arm(rig,monkeypatch,bad):
    original=gain.read_drive
    rows=fingers()
    if bad=='missing':rows[1]['schemas']=[]
    if bad=='reference':rows[1]['relationships']['physxMimicJoint:rotX:referenceJoint']=['other']
    if bad=='gearing':rows[1]['attributes']['physxMimicJoint:rotX:gearing']=1.
    if bad=='leader_mimic':rows[0]['schemas'].append('PhysxMimicJointAPI:rotX')
    def read(expert):
        value=original(expert);value['usd_fingers']=copy.deepcopy(rows);return value
    monkeypatch.setattr(gain,'read_drive',read)
    rig.cfg['requested_gains']=copy.deepcopy(control.GAINS)
    gate=control.SingleDriveGainControl(rig.cfg);gate.observe(rig.expert,0)
    if bad:
        with pytest.raises(ValueError,match='Mimic'):rig.expert.ensure_gripper_force(70.,speed=.05,reconfigure=True)
        assert not any(c[0]=='setter' for c in rig.calls)
        return
    assert rig.expert.ensure_gripper_force(70.,speed=.05,reconfigure=True) is True
    gate.observe(rig.expert,1);gate.observe(rig.expert,154)
    assert rig.kp[0,7:].tolist()==[10500.,0.] and rig.kd[0,7:].tolist()==[140.,0.]
    assert np.all(rig.kp[0,:7]==1) and np.all(rig.kd[0,:7]==2)
    assert gate.state['authored_mimic_checked'] and gate.state['caps_targets_usd_state_preserved']
    assert len([c for c in rig.calls if c[0]=='setter'])==1
    rig.kp[0,8]=10500.
    with pytest.raises(ValueError,match='changed after'):gate.observe(rig.expert,155)


def matched_fixture(tmp_path,monkeypatch):
    base={k:'same' for k in ['seed','attempt_offset','frames','source','source_reference_sha256','task_sha256',
        'physics_sha256','patch_sha256','worker_sha256','controller_config_sha256','urdf_sha256',
        'packet_gain_profile','packet_posture_profile','robot_posture_configuration',
        'cartesian_target_capture_sha256','teacher_replay_sha256','original_depth_bias_m',
        'stage_load_barrier','trace_gripper_drives','history_attempts','initialization_reference_report_sha256']}
    base.update(adapter_sha256={},setup_force_control={'audit':'old','setting':1},
        gripper_gain_control={'audit':'old','requested_gains':[[[10500.,10500.]],[[140.,140.]]], 'cap':70.})
    (tmp_path/'plan.json').write_text(json.dumps(base));(tmp_path/'report.json').write_text('{}')
    actions=np.zeros((319,29),np.float32);np.save(tmp_path/'actions.npy',actions)
    monkeypatch.setattr(control,'BASELINE',tmp_path)
    plan=copy.deepcopy(base)
    plan['gripper_gain_control']['requested_gains']=copy.deepcopy(control.GAINS)
    plan['mimic_drive_control']=dict(schema=control.SCHEMA,baseline=str(tmp_path),training_eligible=False,
        baseline_plan_sha256=control.sha256(tmp_path/'plan.json'),
        baseline_report_sha256=control.sha256(tmp_path/'report.json'),
        baseline_actions_sha256=control.sha256(tmp_path/'actions.npy'))
    return plan,actions


@pytest.mark.parametrize('tamper',[None,'leader_gain','force_cap','precision','patch','arm_profile'])
def test_comparison_allows_only_declared_follower_change(tmp_path,monkeypatch,tamper):
    plan,_=matched_fixture(tmp_path,monkeypatch)
    plan['setup_force_control']['audit']='new';plan['gripper_gain_control']['audit']='new'
    if tamper=='leader_gain':plan['gripper_gain_control']['requested_gains'][0][0][0]=2000.
    if tamper=='force_cap':plan['gripper_gain_control']['cap']=100.
    if tamper=='precision':plan['precision_control']={'active':True}
    if tamper=='patch':plan['patch_sha256']='changed'
    if tamper=='arm_profile':plan['packet_gain_profile']='changed'
    if tamper:
        with pytest.raises(ValueError):control.verify_plan(plan)
    else:control.verify_plan(plan)


@pytest.mark.parametrize('bad',[None,'dtype','values'])
def test_runtime_checks_actual_action_file(tmp_path,monkeypatch,bad):
    plan,actions=matched_fixture(tmp_path,monkeypatch)
    if bad=='dtype':actions=actions.astype(np.float64)
    if bad=='values':actions[0,0]=.1
    np.save(tmp_path/'new.npy',actions)
    plan.update(action_vectors=str(tmp_path/'new.npy'),action_vectors_sha256=control.sha256(tmp_path/'new.npy'))
    if bad:
        with pytest.raises(ValueError,match='stream changed'):control.verify_runtime(plan)
    else:control.verify_runtime(plan)


def test_motion_metric_uses_dt_and_fixed_window(tmp_path):
    t=np.array([2.83,2.85,2.87,2.99,3.01]);v=np.zeros((5,9));v[:,4]=[0,1,-1,200,500]
    with h5py.File(tmp_path/'replay.hdf5','w') as h:
        h['evidence/motion/timestamp']=t;h['evidence/motion/obs/joint_vel']=v
    f=dict(projection_count=2,rejection_count=0,maximum_model_velocity_prediction_error_rad_s_per_joint=[0]*7)
    (tmp_path/'meta.json').write_text(json.dumps({'arm_execution':{'dynamic_drive':{'effort_feasibility':f}}}))
    result=control.motion_summary(tmp_path)
    assert result['window_covered'] and result['window_samples']==2
    assert result['joint5_max_abs_velocity_rad_s']==1
    assert result['joint5_max_abs_secant_acceleration_rad_s2']==pytest.approx(100)


def test_missing_run_never_reports_grasp_failure_or_success(tmp_path,monkeypatch):
    monkeypatch.setattr(control,'verify_runtime',lambda p:None)
    result=control.audit({'output':str(tmp_path)},dict(execution_complete=False,grasp='not_evaluated'))
    assert result['errors'] and result['grasp']=='not_evaluated' and not result['combined_passed']
