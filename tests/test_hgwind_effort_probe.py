import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import h5py
import numpy as np
import pytest

import capstone_vla.hgwind_effort_probe as probe
from capstone_vla.hgwind_cohort import TASK_ID


def source():
    plan={'schema':'physical-vla-lab.hgwind-effort-probe.v1','mode':'joint'}
    meta={'task':{'id':TASK_ID},'task_plan':{'steps':[{'skill':'Grasp','target_role':'A'}]},
          'robot':{'type':'franka'},'arm_execution':{'mode':'operational_space_inverse_dynamics_effort'},
          'joint_names':[f'panda_joint{i}' for i in range(1,8)]+['panda_finger_joint1','panda_finger_joint2'],
          'success':True,'runtime_safety':{'passed':True},'grasp_mode':'physical_finger_contact_only',
          'fix_joint':False,'fixedforce':False,'control_hz':60}
    actions=np.zeros((3,9));actions[:,7:]=.04
    return plan,meta,actions,np.arange(3)/60


def test_valid_effort_source_accepted():
    probe.validate_source(*source())


@pytest.mark.parametrize('field,value',[('grasp_mode','fixedforce'),('fixedforce',True),('control_hz',30),('success',False)])
def test_ineligible_reference_rejected(field,value):
    p,m,a,t=source();m[field]=value
    with pytest.raises(ValueError):probe.validate_source(p,m,a,t)


def test_position_drive_source_and_wrong_task_rejected():
    p,m,a,t=source();m['arm_execution']['mode']='position_drive'
    with pytest.raises(ValueError,match='actuation'):probe.validate_source(p,m,a,t)
    p,m,a,t=source();m['task']['id']='another-task'
    with pytest.raises(ValueError,match='HGwind'):probe.validate_source(p,m,a,t)


def test_gripper_or_clock_corruption_rejected():
    p,m,a,t=source();a[0,8]=0
    with pytest.raises(ValueError,match='gripper'):probe.validate_source(p,m,a,t)
    p,m,a,t=source();t[1]=t[0]
    with pytest.raises(ValueError,match='clock'):probe.validate_source(p,m,a,t)


def test_joint_branch_keeps_source_goals_and_uses_no_shadow_velocity(monkeypatch):
    p,m,a,t=source();monkeypatch.setattr(probe,'_STATE',{'plan':p,'requests':[],'poses':np.zeros((3,7))})
    target,velocity=probe.reference_frame(a,0,m['joint_names'],m['joint_names'],measured_positions=np.zeros(9))
    np.testing.assert_array_equal(target,a[0].astype(np.float32));assert np.isnan(velocity).all()
    target[0]=123;assert a[0,0]==0


def test_cartesian_branch_uses_measured_robot_joints(monkeypatch):
    p,m,a,t=source();p['mode']='cartesian';seen=[]
    def ik(q,action):seen.append(q.copy());return q+.01,0.
    monkeypatch.setattr(probe,'_STATE',{'plan':p,'requests':[],'poses':np.zeros((3,7)),
                                     'adapter':SimpleNamespace(joint_target=ik)})
    q=np.r_[np.full(7,.2),.04,.04]
    target,velocity=probe.reference_frame(a,1,m['joint_names'],m['joint_names'],measured_positions=q)
    np.testing.assert_array_equal(seen[0],q[:7]);np.testing.assert_allclose(target[:7],.21);np.testing.assert_array_equal(target[7:],0);assert np.isnan(velocity).all()


def test_reference_rejects_reordered_runtime_joints(monkeypatch):
    p,m,a,t=source();monkeypatch.setattr(probe,'_STATE',{'plan':p})
    with pytest.raises(ValueError,match='joint order'):
        probe.reference_frame(a,0,m['joint_names'],m['joint_names'][::-1],measured_positions=np.zeros(9))


def test_initial_state_comparison_handles_object_order_and_quaternion_sign():
    d={'joint_pos':np.zeros(9),'joint_vel':np.zeros(9),'object_pos':np.arange(6).reshape(2,3),
       'object_vel':np.zeros((2,3)),'object_ang_vel':np.zeros((2,3)),'object_quat':np.tile([1,0,0,0],(2,1))}
    c={k:(v[::-1].copy() if k.startswith('object_') else v.copy()) for k,v in d.items()};c['object_quat']*=-1
    assert not any(probe.initial_state_differences(d,c,['a','b'],['b','a']).values())
    c['object_vel'][0,0]=.1
    assert probe.initial_state_differences(d,c,['a','b'],['b','a'])['object_vel_max_norm']==pytest.approx(.1)


def test_unknown_worker_source_is_not_silently_patched():
    with pytest.raises(ValueError,match='suppression changed'):probe.patch_worker('print("unrelated")')


def test_substep_contact_alternation_is_reported_in_control_bins(tmp_path):
    spec=importlib.util.spec_from_file_location('effort_analysis',Path(__file__).resolve().parents[1]/'scripts/analyze_hgwind_effort_probe.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    p=tmp_path/'contacts.hdf5'
    with h5py.File(p,'w') as f:
        c=f.create_group('evidence/contacts');c['paths']=np.asarray(['/ball','/robot/panda_leftfinger','/robot/panda_rightfinger'],dtype=h5py.string_dtype())
        c['step_time']=np.arange(1,7)/180;c['step_dt']=np.full(6,1/180)
        c['headers']=np.asarray([[i,0,1+i%2,0,1+i%2,2,i,1] for i in range(6)])
        points=np.zeros((6,10));points[:,6]=1;c['points']=points
        l,r=module.control_contact_arrays(f,'/ball',np.arange(2)/60)
        np.testing.assert_array_equal(l,[2,1]);np.testing.assert_array_equal(r,[1,2])


def test_absent_finger_contact_is_zero_not_an_analysis_exception(tmp_path):
    spec=importlib.util.spec_from_file_location('effort_analysis_zero',Path(__file__).resolve().parents[1]/'scripts/analyze_hgwind_effort_probe.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with h5py.File(tmp_path/'empty.hdf5','w') as f:
        c=f.create_group('evidence/contacts');c['paths']=np.asarray(['/floor'],dtype=h5py.string_dtype());c['step_time']=np.arange(1,7)/180;c['step_dt']=np.full(6,1/180)
        c['headers']=np.zeros((0,8),np.int64);c['points']=np.zeros((0,10))
        l,r=module.control_contact_arrays(f,'/ball',np.arange(2)/60)
        assert not l.any() and not r.any()


def test_recorded_motion_targets_next_state_with_matching_velocity_and_current_grip():
    p,m,a,t=source();a[1,7:]=0
    q=np.arange(27,dtype=float).reshape(3,9)/100;v=q/10
    target,velocity=probe.build_reference(a,q,v,'recorded-motion')
    assert target.shape==(2,9)  # There is no observed state 3.
    np.testing.assert_allclose(target[:,:7],q[1:,:7])
    np.testing.assert_array_equal(target[:,7:],a[:-1,7:].astype(np.float32))
    np.testing.assert_allclose(velocity[:,:7],v[1:,:7]);assert not velocity[:,7:].any()
    assert np.isfinite(velocity).all()


def test_recorded_motion_rejects_invalid_robot_state():
    p,m,a,t=source();q=a.copy();q[1,0]=np.nan
    with pytest.raises(ValueError,match='robot-motion'):
        probe.build_reference(a,q,a,'recorded-motion')
    with pytest.raises(ValueError,match='Unknown reference'):
        probe.build_reference(a,a,a,'invented')


def test_saved_targets_reference_remains_unchanged_and_copied():
    p,m,a,t=source();target,velocity=probe.build_reference(a,None,None,'saved-targets')
    np.testing.assert_array_equal(target,a.astype(np.float32));assert np.isnan(velocity).all()
    target[0,0]=12;assert a[0,0]==0


def test_explicit_motion_velocity_is_not_replaced_by_position_error(monkeypatch):
    p,m,a,t=source();p['reference']='recorded-motion';v=np.full((3,9),.12)
    monkeypatch.setattr(probe,'_STATE',{'plan':p,'requests':[],'poses':np.zeros((3,7)),'velocities':v})
    _,actual=probe.reference_frame(a,1,m['joint_names'],m['joint_names'],measured_positions=np.zeros(9))
    np.testing.assert_array_equal(actual,v[1]);actual[0]=3;assert v[1,0]==.12


def test_analysis_finds_actual_collector_output_layout(tmp_path):
    spec=importlib.util.spec_from_file_location('effort_paths',Path(__file__).resolve().parents[1]/'scripts/analyze_hgwind_effort_probe.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    ep=tmp_path/'1s1p'/'task'/'failed_00000';ep.mkdir(parents=True);(ep/'replay.hdf5').touch()
    assert module.find_probe_episodes(tmp_path)==[ep]


class SimpleTaskKinematics:
    def __init__(self, nonlinear=False):self.nonlinear=nonlinear
    def pose_jacobian(self,q):
        from scipy.spatial.transform import Rotation
        p=np.asarray(q[:3]).copy();j=np.zeros((6,7));j[:3,:3]=np.eye(3);j[5,3]=1
        if self.nonlinear:p[0]=q[0]**2;j[0,0]=2*q[0]
        return p,Rotation.from_euler('z',q[3]),j


def test_task_reference_uses_world_frame_for_pose_twist_and_jacobian():
    from scipy.spatial.transform import Rotation
    a=SimpleTaskKinematics();q=np.zeros(7);target=q.copy();target[[0,3]]=[.02,.03]
    v=q.copy();v[0]=.1;acc=q.copy();acc[0]=.2;base=Rotation.from_euler('z',np.pi/2).as_matrix()
    t=probe.task_reference_terms(a,q,q,target,v,acc,base)
    np.testing.assert_allclose(t['task_position_error'],[0,.02,0,0,0,.03],atol=1e-12)
    np.testing.assert_allclose(t['commanded_task_velocity'],[0,.1,0,0,0,0],atol=1e-12)
    np.testing.assert_allclose(t['feedforward_task_acceleration'],[0,.2,0,0,0,0],atol=1e-12)
    np.testing.assert_allclose(t['jacobian'][:3,0],base[:,0]);assert t['physics_rate_reference']


def test_task_reference_includes_jacobian_rate_products():
    a=SimpleTaskKinematics(nonlinear=True);q=np.zeros(7);q[0]=.4;v=np.zeros(7);v[0]=.3;acc=np.zeros(7);acc[0]=.2
    t=probe.task_reference_terms(a,q,v,q,v,acc,np.eye(3))
    np.testing.assert_allclose(t['task_position_error'],0,atol=1e-12)
    assert t['feedforward_task_acceleration'][0]==pytest.approx(2*.4*.2+2*.3**2,abs=1e-9)
    assert t['jacobian_dot_velocity'][0]==pytest.approx(2*.3**2,abs=1e-9)


def test_joint_tracking_branch_does_not_consume_expert_reference(monkeypatch):
    monkeypatch.setattr(probe,'_STATE',{'plan':{'tracking':'joint'}})
    assert probe.physics_task_reference(None,None,None,None,None,None) is None


def test_task_tracking_rejects_wrong_live_tool_frame(monkeypatch):
    q=np.zeros(7);expert=SimpleNamespace(_base_pose=lambda:(np.zeros(3),[1,0,0,0]),ee_pose=lambda:(np.ones(3),[1,0,0,0]))
    monkeypatch.setattr(probe,'_STATE',{'plan':{'tracking':'task-space'},'adapter':SimpleTaskKinematics(),'task_reference_checks':[]})
    with pytest.raises(ValueError,match='tool frames differ'):probe.physics_task_reference(expert,q,q,q,q,q)


def test_task_tracking_rejects_wrong_live_jacobian(monkeypatch):
    q=np.zeros(7);expert=SimpleNamespace(_base_pose=lambda:(np.zeros(3),[1,0,0,0]),ee_pose=lambda:(np.zeros(3),[1,0,0,0]),measured_ee_jacobian=lambda:np.zeros((6,7)))
    monkeypatch.setattr(probe,'_STATE',{'plan':{'tracking':'task-space'},'adapter':SimpleTaskKinematics(),'task_reference_checks':[]})
    with pytest.raises(ValueError,match='Jacobians differ'):probe.physics_task_reference(expert,q,q,q,q,q)


def test_task_reference_invalid_robot_vectors_or_base_are_rejected():
    a=SimpleTaskKinematics();q=np.zeros(7);bad=q.copy();bad[0]=np.nan
    with pytest.raises(ValueError,match='joint vectors'):probe.task_reference_terms(a,bad,q,q,q,q,np.eye(3))
    with pytest.raises(ValueError,match='base rotation'):probe.task_reference_terms(a,q,q,q,q,q,np.zeros((3,3)))


def test_task_tracking_requires_explicit_recorded_motion_semantics():
    p,m,a,t=source();p['tracking']='task-space'
    with pytest.raises(ValueError,match='requires recorded-motion'):probe.validate_source(p,m,a,t)
    p['reference']='recorded-motion';probe.validate_source(p,m,a,t)


def test_task_tracking_accepts_consistent_frame_and_counts_physics_updates(monkeypatch):
    a=SimpleTaskKinematics();q=np.zeros(7);expert=SimpleNamespace(_base_pose=lambda:(np.zeros(3),[1,0,0,0]),ee_pose=lambda:(np.zeros(3),[1,0,0,0]),measured_ee_jacobian=lambda:a.pose_jacobian(q)[2])
    state={'plan':{'tracking':'task-space'},'adapter':a,'task_reference_checks':[]};monkeypatch.setattr(probe,'_STATE',state)
    for _ in range(2):
        ref=probe.physics_task_reference(expert,q,q,q,q,q)
        np.testing.assert_allclose(ref['task_position_error'],0)
    assert np.asarray(state['task_reference_checks']).shape==(2,4)


class NativePoseExpertFixture:
    def __init__(self,failure=None):
        self.failure=failure;self.pending={'stale':True};self.calls=[]
    def _base_pose(self):return np.zeros(3),np.array([1,0,0,0])
    def ee_pose(self):return np.zeros(3),np.array([1,0,0,0])
    def ee_target_to_pinch(self,p,q):return np.asarray(p)+np.array([.1,0,0])
    def consume_operational_space_reference(self):
        value=self.pending;self.pending=None;return value
    def operational_space_pinch_action(self,p,q,**kwargs):
        self.calls.append((p.copy(),q.copy(),kwargs))
        if self.failure=='action':return None
        if self.failure!='reference':self.pending={'jacobian':np.zeros((6,7)),'task_position_error':np.zeros(6),'commanded_task_velocity':np.zeros(6),'feedforward_task_acceleration':np.zeros(6)}
        return SimpleNamespace(joint_indices=np.arange(7),joint_positions=np.full(7,.2),joint_velocities=None if self.failure=='velocity' else np.full(7,.01))


def test_native_pose_servo_owns_position_velocity_reference_and_preserves_grip(monkeypatch):
    expert=NativePoseExpertFixture();state={'plan':{'tracking':'native-pose'},'adapter':SimpleTaskKinematics(),'native_pose_calls':[]}
    monkeypatch.setattr(probe,'_STATE',state);target=np.r_[np.zeros(7),.04,.04];v=np.zeros(9);v[3]=2
    q,velocity=probe.native_pose_frame(expert,target,v,np.zeros(7))
    np.testing.assert_allclose(q[:7],.2);np.testing.assert_allclose(q[7:],.04);np.testing.assert_allclose(velocity[:7],.01)
    args=expert.calls[0][2];np.testing.assert_allclose(args['target_linear_velocity'],[0,.2,0]);np.testing.assert_allclose(args['target_angular_velocity'],[0,0,2])
    assert args['dt']==1/60 and args['coupled_operational_pose_task'] and args['use_operational_space_inverse_dynamics']
    assert len(state['native_pose_calls'])==1 and expert.pending is None
    assert probe.physics_task_reference(None,None,None,None,None,None) is state['native_operational_reference']


@pytest.mark.parametrize('failure',['action','velocity','reference'])
def test_native_pose_rejects_missing_paired_native_commands(monkeypatch,failure):
    monkeypatch.setattr(probe,'_STATE',{'plan':{'tracking':'native-pose'},'adapter':SimpleTaskKinematics(),'native_pose_calls':[]})
    with pytest.raises(ValueError):probe.native_pose_frame(NativePoseExpertFixture(failure),np.r_[np.zeros(7),.04,.04],np.zeros(9),np.zeros(7))


def clock_module():
    spec=importlib.util.spec_from_file_location('clock_analysis',Path(__file__).resolve().parents[1]/'scripts/diagnose_hgwind_reference_clock.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_reference_clock_constant_velocity_is_consistent():
    dt=1/180;v=np.ones((7,2));q=np.arange(7)[:,None]*dt*v
    assert clock_module().position_velocity_consistency(q,v,dt)['maximum_abs_rad_s']<1e-12


def test_reference_clock_held_positions_do_not_describe_supplied_nonzero_velocity():
    result=clock_module().position_velocity_consistency(np.zeros((7,2)),np.ones((7,2)),1/180)
    assert result['rms_rad_s']==pytest.approx(1.) and result['maximum_abs_rad_s']==pytest.approx(1.)


def test_reference_clock_invalid_timing_is_rejected():
    with pytest.raises(ValueError):clock_module().position_velocity_consistency(np.zeros((3,2)),np.zeros((3,2)),0)


def test_partial_native_branch_is_verified_before_final_rejected_effort():
    tag={'native_pose_command_count':132}
    assert probe.controller_branch_verified('native-pose',393,394,132,tag,True)
    assert not probe.controller_branch_verified('native-pose',393,394,132,tag,False)
    assert not probe.controller_branch_verified('native-pose',393,395,132,tag,True)


def test_original_expert_can_use_osc_for_only_some_control_phases():
    tag={'original_expert_used':True,'segment_trace_count':209}
    assert probe.controller_branch_verified('native-expert',627,582,209,tag,False)
    assert not probe.controller_branch_verified('native-expert',627,628,209,tag,False)


def test_original_trace_copies_inputs_without_changing_commands(tmp_path,monkeypatch):
    from capstone_vla import hgwind_expert_trace as trace
    state={'plan':{'output':str(tmp_path),'schema':'test','source_reference_sha256':'source','original_depth_bias_m':.002},'rows':[],'assistance':[],'joints':[]}
    monkeypatch.setattr(trace,'_STATE',state)
    target=np.arange(9,dtype=float);original=target.copy();segment={'goal_position':target[:7],'goal_velocity':np.zeros(7),'operational_space_reference':{'jacobian':np.eye(6,7)},'policy_joint_velocity_limit':None,'duration_s':1/60,'episode_time_s':0.}
    trace.record_segment(0,segment,target,np.zeros(9),np.zeros(9),None,'robot',[],{})
    np.testing.assert_array_equal(target,original);target[0]=99
    assert state['rows'][0]['target'][0]==0
    result=trace.summary(None);assert result['original_expert_used'] and not result['saved_commands_own_motors']
    assert (tmp_path/'original_expert_control_references.json').is_file()


def test_original_trace_rejects_unknown_frozen_source():
    from capstone_vla import hgwind_expert_trace as trace
    with pytest.raises(ValueError):trace.patch_collector('unrelated=1')
    with pytest.raises(ValueError):trace.patch_profile_wrapper('unrelated=1')


def test_original_expert_source_validation_does_not_require_saved_action_execution():
    p,m,a,t=source();p.update(tracking='native-expert',reference='online-expert')
    probe.validate_source(p,m,a,t)


def logged_fixture(tmp_path):
    import json
    q=np.zeros((3,9));v=np.zeros((3,9));rows=[]
    for i in range(3):
        target=np.r_[np.full(7,.05),.04,.04].astype(np.float32)
        rows.append({'frame':i,'duration_s':1/60,'episode_time_s':i/60,'joint_positions':q[i].tolist(),'joint_velocities':v[i].tolist(),
            'target':target.tolist(),'goal_position':target[:7].tolist(),'goal_velocity':np.full(7,.1).tolist(),
            'policy_joint_velocity_limit':None if i==0 else np.full(7,1.8).tolist(),
            'operational_space_reference':None if i==0 else {'jacobian':np.eye(6,7).tolist(),'task_position_error':np.zeros(6).tolist(),'commanded_task_velocity':np.ones(6).tolist(),'feedforward_task_acceleration':np.zeros(6).tolist()}})
    p=tmp_path/'controls.json';p.write_text(json.dumps(rows));return p,q,v,rows


def test_logged_control_keeps_original_endpoint_velocity_and_first_joint_fallback(tmp_path):
    p,q,v,rows=logged_fixture(tmp_path);targets,vel,loaded=probe.load_logged_commands(p,q,v)
    np.testing.assert_array_equal(targets,np.asarray([r['target'] for r in rows],np.float32))
    np.testing.assert_allclose(vel[:,:7],.1);assert np.isnan(vel[:,7:]).all()
    assert loaded[0]['operational_space_reference'] is None and loaded[1]['operational_space_reference']['task_position_error']==[0]*6


@pytest.mark.parametrize('fault',['frame','velocity_shape','source_state','osc_velocity'])
def test_logged_control_rejects_mismatched_time_motion_or_command_tuple(tmp_path,fault):
    import json
    p,q,v,rows=logged_fixture(tmp_path)
    if fault=='frame':rows[1]['frame']=2
    elif fault=='velocity_shape':rows[1]['goal_velocity']=[0]*6
    elif fault=='source_state':q[1,0]=1
    else:rows[1]['operational_space_reference']['commanded_task_velocity'][0]=float('nan')
    p.write_text(json.dumps(rows))
    with pytest.raises(ValueError):probe.load_logged_commands(p,q,v)


def test_logged_control_reference_and_policy_limit_reach_physics_boundary(monkeypatch,tmp_path):
    p,q,v,rows=logged_fixture(tmp_path)
    monkeypatch.setattr(probe,'_STATE',{'plan':{'tracking':'logged-command'},'active_logged_row':rows[1]})
    np.testing.assert_allclose(probe.reference_policy_velocity_limit(),1.8)
    ref=probe.physics_task_reference(None,None,None,None,None,None)
    np.testing.assert_allclose(ref['jacobian'],np.eye(6,7));assert 'physics_rate_reference' not in ref
    monkeypatch.setitem(probe._STATE,'active_logged_row',rows[0])
    assert probe.reference_policy_velocity_limit() is None and probe.physics_task_reference(None,None,None,None,None,None) is None
