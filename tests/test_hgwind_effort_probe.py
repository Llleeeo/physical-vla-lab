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
