import copy
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from capstone_vla.cartesian_packets import (TargetBlindSafetyWorld, build_packets,
    execute_request, PacketStreamExecutor, audit_packets, patch_packet_collector)
from capstone_vla.cartesian_target_trace import encode


class GeometryOwner:
    target='/ball'
    def __init__(self):self.refreshes=0;self.drive_configurations=0;self._prev_ee_p=None;self._prev_ee_q=None
    def _ee_velocity(self, position, dt):
        return np.zeros(3) if self._prev_ee_p is None else (position-self._prev_ee_p)/dt
    def _ee_angular_velocity(self, quaternion, dt):
        from scipy.spatial.transform import Rotation
        if self._prev_ee_q is None:return np.zeros(3)
        return (Rotation.from_quat(quaternion[[1,2,3,0]])*Rotation.from_quat(self._prev_ee_q[[1,2,3,0]]).inv()).as_rotvec()/dt
    def _ensure_gripper_drive(self):
        self.drive_configurations+=1;self._gripper_drive_configured=True
    def _update_environment_obstacles(self,world):
        self.refreshes+=1
        world.pose('/fan')
    def _project_keep_clear_joint_velocity(self,*args,**kwargs):raise AssertionError('not expected in toy servo')
    def _environment_safe_translation(self,*args,**kwargs):raise AssertionError('not expected in toy servo')
    def step(self,*args):raise AssertionError('Online skill step must not run')

for name in ['_project_keep_clear_joint_velocity','_environment_safe_translation']:
    method=getattr(GeometryOwner,name)
    method.__module__='expert_data.collision_safety'
    method.__qualname__='CollisionSafety.'+name


class Robot:
    def __init__(self):self.position=np.zeros(3);self.quaternion=np.array([1.,0.,0.,0.]);self.arms=0;self.jaws=[]
    def pinch_pose(self):return self.position.copy(),self.quaternion.copy()
    def operational_space_pinch_action(self,**inputs):
        self.arms+=1
        v=np.r_[np.asarray(inputs['target_position'])-self.position,np.zeros(4)]
        self._pending_operational_space_reference={'jacobian':np.eye(6,7),'commanded_task_velocity':v[:6]}
        return SimpleNamespace(joint_indices=np.arange(7),joint_positions=v.copy(),joint_velocities=v.copy())
    def gripper_action(self,*,width,close):
        assert width==0.
        self.jaws.append(close)
        return SimpleNamespace(joint_indices=np.array([7,8]),joint_positions=np.full(2,0. if close else .04))


def capture():
    owner=GeometryOwner()
    inputs=dict(target_position=[.1,.2,.3],target_quaternion=[1.,0.,0.,0.],dt=1/60,
        target_linear_velocity=[.1,0,0],target_linear_acceleration=[1.,0,0],target_angular_velocity=[0,0,0],
        linear_velocity_reference=None,maximum_relative_linear_speed_mps=None,joint_velocity_reference=None,
        joint_velocity_projector=encode(owner._project_keep_clear_joint_velocity),
        cartesian_velocity_projector=encode(owner._environment_safe_translation))
    return dict(schema='physical-vla-lab.cartesian-target-capture.v1',frames=[0,1,2],
        calls=[dict(call_id=i,frame=i,completed=True,inputs=copy.deepcopy(inputs),returned_action={'do_not_use':True}) for i in range(2)])


def setup(monkeypatch):
    monkeypatch.setitem(sys.modules,'expert_data.skills.base',SimpleNamespace(role_path=lambda x:x))
    robot=Robot();owner=GeometryOwner();world=SimpleNamespace(pose=lambda path:([0,0,0],[1,0,0,0]))
    policy=SimpleNamespace(expert=robot,active_primitives=lambda:[owner],step=lambda *args:pytest.fail('online task step called'))
    packets=build_packets(capture(),[[.04,.04],[0,0],[0,0]])
    return packets,robot,owner,world,policy


def test_packets_own_jaws_and_cartesian_tail_without_skill_decisions(monkeypatch):
    packets,robot,owner,world,policy=setup(monkeypatch);executor=PacketStreamExecutor(packets);segments=[]
    for i in range(3):
        decision=executor.step(policy,world,i,1/60);arm,jaw=decision.actions
        segment=dict(frame=i,goal_position=arm.joint_positions,goal_velocity=arm.joint_velocities,
            operational_space_reference=robot._pending_operational_space_reference)
        segment.update(executor.segment_links(i,segment));segments.append(segment)
    assert robot.jaws==[False,True,True] and owner.refreshes==3 and robot.arms==3 and owner.drive_configurations==1
    payload=executor.payload();audit=audit_packets(payload,packets,segments,3)
    assert audit['packet_execution_verified'] and audit['terminal_cartesian_hold_frames']==1
    assert payload['non_target_safety_geometry_paths']==['/fan']
    assert not payload['online_skill_step_called'] and not payload['expert_joint_fallback_used']
    assert 'do_not_use' not in str(packets)


def test_terminal_hold_zeroes_motion_without_copying_joint_tail():
    source=capture();packets=build_packets(source,np.zeros((3,2)))
    np.testing.assert_array_equal(packets[-1]['inputs']['target_position'],source['calls'][-1]['inputs']['target_position'])
    np.testing.assert_array_equal(packets[-1]['inputs']['target_linear_velocity'],[0,0,0])
    np.testing.assert_array_equal(packets[-1]['inputs']['target_linear_acceleration'],[0,0,0])
    assert packets[-1]['terminal_cartesian_hold'] and not packets[1]['terminal_cartesian_hold']
    assert source['calls'][-1]['inputs']['target_linear_acceleration']==[1.,0,0]


def test_native_feedback_uses_current_robot_state_and_returns_separate_grip():
    packet=build_packets(capture(),np.zeros((3,2)))[0];owner=GeometryOwner()
    callbacks=dict(joint_velocity_projector=owner._project_keep_clear_joint_velocity,cartesian_velocity_projector=owner._environment_safe_translation)
    a,b=Robot(),Robot();b.position[:]=.03
    first=execute_request(a,packet,callbacks);second=execute_request(b,packet,callbacks)
    np.testing.assert_allclose(first[0].joint_velocities[:3]-second[0].joint_velocities[:3],.03)
    np.testing.assert_array_equal(first[1].joint_indices,[7,8])


def test_safety_target_and_descendants_are_blocked_before_read():
    reads=[];world=SimpleNamespace(pose=lambda path:reads.append(path))
    blind=TargetBlindSafetyWorld(world,'/ball')
    for path in ['/ball','/ball/mesh']:
        with pytest.raises(ValueError,match='Target truth'):blind.pose(path)
    assert not reads
    blind.pose('/fan');assert reads==['/fan']
    with pytest.raises(AttributeError,match='Unsupported safety'):blind.contact('/a','/b')


def test_caught_target_access_or_missing_geometry_stops_before_motor(monkeypatch):
    packets,robot,owner,world,policy=setup(monkeypatch)
    def unsafe(blind):
        try:blind.pose('/ball')
        except ValueError:pass
    owner._update_environment_obstacles=unsafe
    with pytest.raises(ValueError,match='attempted target'):PacketStreamExecutor(packets).step(policy,world,0,1/60)
    assert robot.arms==0
    owner._update_environment_obstacles=lambda blind:None
    owner._environment_missing_narrow_phase=['/fan']
    with pytest.raises(ValueError,match='incomplete'):PacketStreamExecutor(packets).step(policy,world,0,1/60)
    assert robot.arms==0


@pytest.mark.parametrize('jaws',[[[0,.04]]*3,[[.01,.01]]*3,[[0,0]]*2])
def test_unsupported_jaw_contract_rejected(jaws):
    with pytest.raises(ValueError):build_packets(capture(),jaws)


def test_missing_call_gaps_and_clock_exhaustion_rejected(monkeypatch):
    source=capture();source['calls'][1]['frame']=2
    with pytest.raises(ValueError,match='one source call'):build_packets(source,np.zeros((3,2)))
    packets,robot,owner,world,policy=setup(monkeypatch);executor=PacketStreamExecutor(packets)
    with pytest.raises(ValueError,match='clock'):executor.step(policy,world,1,1/60)
    with pytest.raises(ValueError,match='cadence'):executor.step(policy,world,0,1/30)


def test_collector_patch_bypasses_decision_call_and_refuses_unknown_source():
    with pytest.raises(ValueError):patch_packet_collector('unrelated')
    anchor='                policy_step_started = time.perf_counter()\n                decision = controller.step(frame, world, dt_control)'
    source='def worker():\n    if True:\n        if True:\n            if True:\n'+anchor
    patched=patch_packet_collector(source)
    assert 'controller.step(' not in patched
    assert 'CAPSTONE_PACKET_DECISION(pol, world, k, dt_control)' in patched
    seen=[]
    namespace=dict(time=SimpleNamespace(perf_counter=lambda:0.),pol='policy',world='world',k=0,dt_control=1/60,
        controller=SimpleNamespace(step=lambda *args:pytest.fail('original decision was called')),
        CAPSTONE_PACKET_DECISION=lambda *args:seen.append(args) or 'packet decision')
    exec(compile(patched+'\n                return decision\n','<packet-branch-test>','exec'),namespace)
    assert namespace['worker']()=='packet decision' and seen==[('policy','world',0,1/60)]


def test_audit_rejects_wrong_gripper_action_and_short_stream(monkeypatch):
    packets,robot,owner,world,policy=setup(monkeypatch);executor=PacketStreamExecutor(packets);segments=[]
    for i in range(3):
        arm,_=executor.step(policy,world,i,1/60).actions
        segments.append(dict(frame=i,goal_position=arm.joint_positions,goal_velocity=arm.joint_velocities,
            operational_space_reference=robot._pending_operational_space_reference))
    payload=executor.payload();payload['calls'][1]['jaw_action']['joint_positions']=[.04,.04]
    assert not audit_packets(payload,packets,segments,3)['packet_execution_verified']
    assert not audit_packets(executor.payload(),packets,segments[:1],1)['packet_execution_verified']


def test_optional_emission_surface_preserves_real_geometry_and_blocks_target():
    mesh=np.arange(9,dtype=float).reshape(1,3,3);reads=[]
    def emission(path):reads.append(path);return mesh
    blind=TargetBlindSafetyWorld(SimpleNamespace(emission_surface=emission),'/ball')
    assert blind.emission_surface('/fan') is mesh
    assert reads==['/fan']
    for path in ['/ball','/ball/mesh']:
        with pytest.raises(ValueError,match='Target truth'):blind.emission_surface(path)
    assert reads==['/fan'] and blind.blocked_paths=={'/ball','/ball/mesh'}


def test_optional_attribute_lookup_follows_world_protocol():
    blind=TargetBlindSafetyWorld(SimpleNamespace(),'/ball')
    assert getattr(blind,'future_optional_geometry',None) is None
    assert blind.emission_surface('/fan') is None
    assert blind.deformable_surface('/fan') is None
    with pytest.raises(AttributeError):blind.contact('/a','/b')


def test_optional_geometry_probe_no_longer_discards_existing_fan_mesh(monkeypatch):
    packets,robot,owner,world,policy=setup(monkeypatch)
    def refresh(blind):
        # Native geometry handles errors by recording the body as missing.
        # An adapter error during its optional emission query must not remove
        # an otherwise complete rigid mesh from the inventory.
        try:
            blind.pose('/fan')
            surface=getattr(blind,'emission_surface',None)
            assert surface('/fan') is None
            owner._environment_missing_narrow_phase=[]
        except Exception:
            owner._environment_missing_narrow_phase=['/fan']
    owner._update_environment_obstacles=refresh
    PacketStreamExecutor(packets).step(policy,world,0,1/60)
    assert robot.arms==1 and not owner._environment_missing_narrow_phase


def test_explicit_close_and_reopen_reach_safety_before_servo(monkeypatch):
    packets,robot,owner,world,policy=setup(monkeypatch)
    packets[2]['gripper_closed']=False
    owner._jaws_closed=True  # Deliberately stale initial context.
    geometry_modes=[];servo_modes=[]
    refresh=owner._update_environment_obstacles
    def check_geometry(blind):
        geometry_modes.append(owner._jaws_closed)
        refresh(blind)
    owner._update_environment_obstacles=check_geometry
    original=robot.operational_space_pinch_action
    def check_servo(**inputs):
        servo_modes.append(owner._jaws_closed)
        return original(**inputs)
    robot.operational_space_pinch_action=check_servo
    executor=PacketStreamExecutor(packets)
    for frame in range(3):executor.step(policy,world,frame,1/60)
    assert geometry_modes==servo_modes==[False,True,False]
    assert robot.jaws==[False,True,False]
    assert [event['safety_gripper_closed'] for event in executor.events]==[False,True,False]


def test_stale_safety_jaw_context_cannot_pass_packet_audit(monkeypatch):
    packets,robot,owner,world,policy=setup(monkeypatch);executor=PacketStreamExecutor(packets);segments=[]
    for frame in range(3):
        arm,_=executor.step(policy,world,frame,1/60).actions
        segments.append(dict(frame=frame,goal_position=arm.joint_positions,goal_velocity=arm.joint_velocities,
            operational_space_reference=robot._pending_operational_space_reference))
    payload=copy.deepcopy(executor.payload())
    assert audit_packets(payload,packets,segments,3)['packet_execution_verified']
    payload['calls'][1]['safety_gripper_closed']=False
    audit=audit_packets(payload,packets,segments,3)
    assert not audit['packet_execution_verified'] and not audit['safety_jaw_context_matches']


def test_close_request_does_not_fabricate_contact_or_skill_phase(monkeypatch):
    packets,robot,owner,world,policy=setup(monkeypatch)
    owner._physical_grasp_confirmed=False;owner.phase='initial'
    executor=PacketStreamExecutor(packets)
    executor.step(policy,world,0,1/60);executor.step(policy,world,1,1/60)
    assert owner._jaws_closed and not owner._physical_grasp_confirmed and owner.phase=='initial'


def test_robot_motion_is_current_before_safety_and_servo(monkeypatch):
    packets,robot,owner,world,policy=setup(monkeypatch);executor=PacketStreamExecutor(packets)
    seen=[];refresh=owner._update_environment_obstacles;servo=robot.operational_space_pinch_action
    def geometry(blind):
        seen.append(('geometry',owner._current_ee_linear_velocity.copy(),owner._current_ee_angular_velocity.copy()))
        refresh(blind)
    def act(**inputs):
        seen.append(('servo',owner._current_ee_linear_velocity.copy(),owner._current_ee_angular_velocity.copy()))
        return servo(**inputs)
    owner._update_environment_obstacles=geometry;robot.operational_space_pinch_action=act
    executor.step(policy,world,0,1/60)
    robot.position[0]=.001;robot.quaternion=np.array([np.cos(.05),0,0,np.sin(.05)])
    executor.step(policy,world,1,1/60)
    for _,linear,angular in seen[:2]:
        np.testing.assert_array_equal(linear,0);np.testing.assert_array_equal(angular,0)
    for _,linear,angular in seen[2:]:
        np.testing.assert_allclose(linear,[.06,0,0],atol=1e-12)
        np.testing.assert_allclose(angular,[0,0,6.],atol=1e-12)


def test_motion_history_advances_once_and_uses_robot_not_target(monkeypatch):
    from capstone_vla.cartesian_packets import audit_robot_safety_motion
    packets,robot,owner,world,policy=setup(monkeypatch);executor=PacketStreamExecutor(packets)
    for frame,position in enumerate([0.,.002,.002]):
        robot.position[0]=position;executor.step(policy,world,frame,1/60)
    np.testing.assert_allclose([e['safety_robot_twist_world']['linear_mps'][0] for e in executor.events],[0.,.12,0.],rtol=0.,atol=1.e-15)
    assert audit_robot_safety_motion(executor.events)
    np.testing.assert_array_equal(owner._prev_ee_p,robot.position)


def test_stale_robot_motion_is_rejected_by_audit(monkeypatch):
    from capstone_vla.cartesian_packets import audit_robot_safety_motion
    packets,robot,owner,world,policy=setup(monkeypatch);executor=PacketStreamExecutor(packets)
    executor.step(policy,world,0,1/60);robot.position[2]=.001;executor.step(policy,world,1,1/60)
    events=copy.deepcopy(executor.events);events[1]['safety_robot_twist_world']['linear_mps']=[0,0,0]
    assert not audit_robot_safety_motion(events)


def test_invalid_robot_pose_stops_before_native_arm_call(monkeypatch):
    packets,robot,owner,world,policy=setup(monkeypatch);robot.position[0]=np.nan
    with pytest.raises(ValueError,match='measured robot pinch'):PacketStreamExecutor(packets).step(policy,world,0,1/60)
    assert robot.arms==0



def gain_source_packets():
    source=capture()
    for i,event in enumerate(source['calls']):
        event['inputs'].update(position_gain=9. if i==0 else 4.,orientation_gain=8. if i==0 else 4.)
    return build_packets(source,np.zeros((3,2)))


def test_fixed_gains_preserve_every_other_request_and_do_not_mutate_source():
    from capstone_vla.cartesian_packets import apply_gain_profile,audit_gain_profile
    source=gain_source_packets();snapshot=copy.deepcopy(source)
    applied=apply_gain_profile(source,'fixed-9-8')
    assert source==snapshot and applied[0]==source[0] and applied[1]!=source[1]
    for before,after in zip(source,applied):
        assert after['inputs']['position_gain']==9. and after['inputs']['orientation_gain']==8.
        restored=copy.deepcopy(after)
        for key in ['position_gain','orientation_gain']:restored['inputs'][key]=before['inputs'][key]
        assert restored==before
    assert audit_gain_profile(source,applied,'fixed-9-8')
    assert apply_gain_profile(source)==source


@pytest.mark.parametrize('field',['target_position','target_linear_acceleration','safety_callbacks','gripper_closed'])
def test_gain_audit_rejects_unrelated_motion_safety_or_gripper_changes(field):
    from capstone_vla.cartesian_packets import apply_gain_profile,audit_gain_profile
    source=gain_source_packets();applied=apply_gain_profile(source,'fixed-9-8')
    if field in applied[1]['inputs']:applied[1]['inputs'][field]=[9.,0.,0.]
    elif field=='gripper_closed':applied[1][field]=not applied[1][field]
    else:applied[1][field]['cartesian_velocity_projector']=None
    assert not audit_gain_profile(source,applied,'fixed-9-8')


def test_gain_probe_rejects_unknown_profile_and_no_effect():
    from capstone_vla.cartesian_packets import apply_gain_profile,audit_gain_profile
    source=gain_source_packets()
    with pytest.raises(ValueError,match='Unknown packet gain'):apply_gain_profile(source,'arbitrary')
    fixed=apply_gain_profile(source,'fixed-9-8')
    assert not audit_gain_profile(fixed,fixed,'fixed-9-8')


def test_fixed_gain_execution_audit_requires_truthful_context(monkeypatch):
    from capstone_vla.cartesian_packets import apply_gain_profile
    _,robot,owner,world,policy=setup(monkeypatch)
    packets=apply_gain_profile(gain_source_packets(),'fixed-9-8')
    executor=PacketStreamExecutor(packets,'fixed-9-8');segments=[]
    for i in range(3):
        arm,_=executor.step(policy,world,i,1/60).actions
        segments.append(dict(frame=i,goal_position=arm.joint_positions,goal_velocity=arm.joint_velocities,
            operational_space_reference=robot._pending_operational_space_reference))
    payload=executor.payload();audit=audit_packets(payload,packets,segments,3)
    assert audit['packet_execution_verified'] and audit['gain_profile_verified']
    assert not audit['source_controller_setting_schedule_retained']
    assert not payload['source_numeric_controller_settings_retained']
    payload['source_numeric_controller_settings_retained']=True
    assert not audit_packets(payload,packets,segments,3)['packet_execution_verified']
    payload=executor.payload();payload['calls'][-1]['inputs']['position_gain']=4.
    assert not audit_packets(payload,packets,segments,3)['packet_execution_verified']



def posture_source_packets():
    packets=gain_source_packets()
    from capstone_vla.cartesian_packets import apply_gain_profile
    packets=apply_gain_profile(packets,'fixed-9-8')
    packets[0]['inputs']['posture_target']=[.1]*7
    packets[1]['inputs']['posture_target']=[.1]*7
    packets[2]['inputs']['posture_target']=None
    return packets


def test_measured_posture_changes_only_reference_without_source_mutation():
    from capstone_vla.cartesian_packets import apply_posture_profile,audit_posture_profile
    source=posture_source_packets();snapshot=copy.deepcopy(source)
    applied=apply_posture_profile(source,'measured')
    assert source==snapshot and applied[-1]==source[-1]
    assert all(p['inputs']['posture_target'] is None for p in applied)
    for before,after in zip(source,applied):
        restored=copy.deepcopy(after);restored['inputs']['posture_target']=before['inputs']['posture_target']
        assert restored==before
    assert audit_posture_profile(source,applied,'measured')
    assert apply_posture_profile(source)==source


@pytest.mark.parametrize('field',['position_gain','target_linear_acceleration','safety_callbacks','gripper_closed'])
def test_posture_audit_rejects_unrelated_changes(field):
    from capstone_vla.cartesian_packets import apply_posture_profile,audit_posture_profile
    source=posture_source_packets();applied=apply_posture_profile(source,'measured')
    if field in applied[0]['inputs']:applied[0]['inputs'][field]=4. if field=='position_gain' else [3.,0.,0.]
    elif field=='gripper_closed':applied[0][field]=not applied[0][field]
    else:applied[0][field]['cartesian_velocity_projector']=None
    assert not audit_posture_profile(source,applied,'measured')


def test_posture_probe_rejects_unknown_profile_and_no_effect():
    from capstone_vla.cartesian_packets import apply_posture_profile,audit_posture_profile
    source=posture_source_packets()
    with pytest.raises(ValueError,match='Unknown packet posture'):apply_posture_profile(source,'home')
    applied=apply_posture_profile(source,'measured')
    assert not audit_posture_profile(applied,applied,'measured')


def test_measured_posture_execution_tracks_current_robot_and_audit_rejects_stale_context(monkeypatch):
    from capstone_vla.cartesian_packets import apply_posture_profile
    _,robot,owner,world,policy=setup(monkeypatch)
    packets=apply_posture_profile(posture_source_packets(),'measured')
    q=np.zeros(7);robot.arm_motion_state=lambda:SimpleNamespace(position=q.copy())
    original=robot.operational_space_pinch_action
    def servo(**inputs):
        assert inputs['posture_target'] is None
        action=original(**inputs)
        robot._last_differential_ik=dict(posture_source='measured_state',posture_target_rad=q.tolist())
        return action
    robot.operational_space_pinch_action=servo
    executor=PacketStreamExecutor(packets,'fixed-9-8','measured');segments=[]
    for i in range(3):
        q[:]=i*.01
        arm,_=executor.step(policy,world,i,1/60).actions
        segments.append(dict(frame=i,goal_position=arm.joint_positions,goal_velocity=arm.joint_velocities,
            operational_space_reference=robot._pending_operational_space_reference))
    payload=executor.payload();audit=audit_packets(payload,packets,segments,3)
    assert audit['posture_profile_verified'] and audit['packet_execution_verified']
    assert payload['calls'][2]['measured_posture_context']['native_posture_target_rad']==[.02]*7
    payload['calls'][2]['measured_posture_context']['native_posture_target_rad']=[0.]*7
    assert not audit_packets(payload,packets,segments,3)['packet_execution_verified']


@pytest.mark.parametrize('defect',['stale','nonfinite'])
def test_native_posture_context_rejects_invalid_resolution(defect):
    from capstone_vla.cartesian_packets import measured_posture_context
    q=np.ones(7)*.2
    robot=SimpleNamespace(arm_motion_state=lambda:SimpleNamespace(position=q.copy()),
        _last_differential_ik=dict(posture_source='measured_state',posture_target_rad=np.zeros(7).tolist()))
    if defect=='nonfinite':q[0]=np.nan
    with pytest.raises(ValueError,match='not current measured'):measured_posture_context(robot)
