"""Live kinematics must depend on the robot, not the reference Jacobian."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from capstone_vla.live_kinematics import live_pinch_jacobian,operational_task_jacobian,validate_logged_task_metric
from capstone_vla import hgwind_effort_probe as probe


def robot(offset=(.1,0,0)):
    jacobian=np.zeros((6,7));jacobian[5,0]=2
    return SimpleNamespace(ee_pose=lambda:(np.array([1.,2.,3.]),np.array([1.,0.,0.,0.])),
        measured_ee_jacobian=lambda:jacobian,
        ee_target_to_pinch=lambda p,q:p+offset)


def test_live_jacobian_shifts_to_current_pinch_without_mutating_tool_jacobian():
    expert=robot();original=expert.measured_ee_jacobian().copy()
    actual=live_pinch_jacobian(expert)
    assert actual[1,0]==pytest.approx(.2)
    np.testing.assert_array_equal(actual[3:],original[3:])
    np.testing.assert_array_equal(expert.measured_ee_jacobian(),original)


def test_live_jacobian_zero_offset_preserves_all_rows():
    expert=robot((0,0,0))
    np.testing.assert_array_equal(live_pinch_jacobian(expert),expert.measured_ee_jacobian())


@pytest.mark.parametrize('fault',['shape','nonfinite','quaternion','pinch'])
def test_invalid_live_robot_geometry_is_rejected(fault):
    expert=robot()
    if fault=='shape':expert.measured_ee_jacobian=lambda:np.zeros((3,7))
    elif fault=='nonfinite':expert.measured_ee_jacobian=lambda:np.full((6,7),np.nan)
    elif fault=='quaternion':expert.ee_pose=lambda:(np.zeros(3),np.zeros(4))
    else:expert.ee_target_to_pinch=lambda p,q:np.full(3,np.nan)
    with pytest.raises(ValueError):live_pinch_jacobian(expert)


def test_saved_jacobian_has_no_effect_on_current_reference_and_other_signals_survive(monkeypatch):
    monkeypatch.setattr(probe,'_STATE',{'plan':{'operational_orientation_length_scale_m':.03},'live_jacobian_records':[]})
    reference={'jacobian':np.eye(6,7).tolist(),'commanded_task_velocity':list(range(6)),
        'feedforward_task_acceleration':[0,0,9.81,0,0,0],'task_position_error':[0]*6,
        'joint_velocity_constraint_matrix':np.eye(7).tolist(),
        'joint_velocity_constraint_lower_bound':[-1]*7}
    original=copy.deepcopy(reference);first=probe.live_reference(robot(),reference,15)
    np.testing.assert_array_equal(first['jacobian'],operational_task_jacobian(live_pinch_jacobian(robot()),.03))
    for key in reference:
        if key!='jacobian':np.testing.assert_array_equal(first[key],reference[key])
    assert reference==original
    reference['jacobian']=np.full((6,7),99.).tolist()
    second=probe.live_reference(robot(),reference,16)
    np.testing.assert_array_equal(first['jacobian'],second['jacobian'])
    assert probe._STATE['live_jacobian_records'][0]['comparison_max_abs']!=probe._STATE['live_jacobian_records'][1]['comparison_max_abs']
    first['feedforward_task_acceleration'][2]=123
    assert reference['feedforward_task_acceleration'][2]==9.81


def test_joint_only_fallback_does_not_query_missing_robot():
    assert probe.live_reference(None,None,0) is None


def test_live_reference_reaches_physics_callback_with_original_velocity_caps(monkeypatch):
    p={'tracking':'live-jacobian','mode':'joint','operational_orientation_length_scale_m':.03};actions=np.zeros((2,9));vel=np.ones((2,9));poses=np.zeros((2,7))
    ref={'jacobian':np.eye(6,7).tolist()}
    rows=[{'operational_space_reference':None,'policy_joint_velocity_limit':None},
          {'operational_space_reference':ref,'policy_joint_velocity_limit':[1.8]*7}]
    state={'plan':p,'velocities':vel,'poses':poses,'requests':[],'logged_rows':rows,'live_jacobian_records':[]}
    monkeypatch.setattr(probe,'_STATE',state)
    names=list(range(9))
    target,velocity=probe.reference_frame(actions,0,names,names,measured_positions=np.zeros(9),expert=None)
    assert probe.physics_task_reference(None,None,None,None,None,None) is None
    assert probe.reference_policy_velocity_limit() is None
    target,velocity=probe.reference_frame(actions,1,names,names,measured_positions=np.zeros(9),expert=robot())
    np.testing.assert_array_equal(target,actions[1]);np.testing.assert_array_equal(velocity,vel[1])
    actual=probe.physics_task_reference(None,None,None,None,None,None)
    np.testing.assert_array_equal(actual['jacobian'],operational_task_jacobian(live_pinch_jacobian(robot()),.03))
    assert 'physics_rate_reference' not in actual  # Original 60 Hz update, held for three physics ticks.
    caps=probe.reference_policy_velocity_limit();np.testing.assert_allclose(caps,1.8)
    caps[0]=99;assert rows[1]['policy_joint_velocity_limit'][0]==1.8
    actual['jacobian'][0,0]=99
    assert probe.physics_task_reference(None,None,None,None,None,None)['jacobian'][0,0]!=99
    assert len(state['live_jacobian_records'])==1


def test_coupled_angular_metric_preserves_equation_with_physical_joint_velocity():
    physical=np.zeros((6,7));physical[:3,:3]=np.eye(3);physical[3:,:3]=2*np.eye(3)
    original=physical.copy();qd=np.arange(7,dtype=float)
    scaled=operational_task_jacobian(physical,.03)
    expected=physical@qd;expected[3:]*=.03
    np.testing.assert_allclose(scaled@qd,expected)
    np.testing.assert_array_equal(scaled[:3],physical[:3])
    np.testing.assert_array_equal(physical,original)
    # A raw Jacobian paired with a weighted twist is the rejected convention.
    assert not np.allclose(physical@qd,expected)


@pytest.mark.parametrize('scale',[0,-1,float('nan'),float('inf')])
def test_coupled_task_rejects_invalid_metric(scale):
    with pytest.raises(ValueError):operational_task_jacobian(np.zeros((6,7)),scale)


def test_all_logged_revolute_axes_must_match_configured_task_metric():
    j=np.zeros((6,7));j[5,:]=.03
    rows=[{'operational_space_reference':{'jacobian':j.tolist()}}, {'operational_space_reference':None}]
    assert validate_logged_task_metric(rows,.03)==1
    with pytest.raises(ValueError,match='metric'):validate_logged_task_metric(rows,1.)
    rows[0]['operational_space_reference']['jacobian'][5][6]=1.
    with pytest.raises(ValueError,match='metric'):validate_logged_task_metric(rows,.03)


def test_live_boundary_applies_metric_before_pairing_original_velocity(monkeypatch):
    state={'plan':{'operational_orientation_length_scale_m':.03},'live_jacobian_records':[]}
    monkeypatch.setattr(probe,'_STATE',state)
    raw=live_pinch_jacobian(robot());qd=np.ones(7)
    weighted=operational_task_jacobian(raw,.03)
    reference={'jacobian':weighted.tolist(),'commanded_task_velocity':(weighted@qd).tolist()}
    actual=probe.live_reference(robot(),reference,0)
    np.testing.assert_allclose(actual['jacobian']@qd,actual['commanded_task_velocity'])
    assert state['live_jacobian_records'][0]['comparison_max_abs']==0



def native_fixture_source():
    # Independent toy FK: a rotating planar tool plus a calibrated offset.
    # The wrapper must extract geometry only, leaving servo work unexecuted.
    return """class Expert:
    def operational_space_pinch_action(self, *, finite_difference_rad=1.e-4):
        try:
            offset=np.asarray(self._pinch_offset_local,float)
            jacobian=np.zeros((6,7))
            for k in range(7):
                plus=q.copy();minus=q.copy()
                plus[k]=min(q[k]+finite_difference_rad,state.upper[k])
                minus[k]=max(q[k]-finite_difference_rad,state.lower[k])
                pp,rp=self._kinematics.compute_forward_kinematics(self._ee_frame,plus)
                pm,rm=self._kinematics.compute_forward_kinematics(self._ee_frame,minus)
                delta=plus[k]-minus[k]
                jacobian[:3,k]=((pp+rp@offset)-(pm+rm@offset))/delta
                jacobian[3:,k]=shortest_quaternion_rotation_vector(utils.matrix_to_quat_wxyz(rm),utils.matrix_to_quat_wxyz(rp))/delta
            self._last_differential_jacobian=jacobian.copy()
            raise RuntimeError('servo must never run')
        except Exception:
            raise
"""


def native_fixture_robot():
    import time
    from scipy.spatial.transform import Rotation
    from capstone_vla.live_kinematics import compile_native_pinch_jacobian
    def quaternion(matrix):
        q=Rotation.from_matrix(matrix).as_quat();return q[[3,0,1,2]]
    def difference(left,right):
        a=Rotation.from_quat(np.asarray(left)[[1,2,3,0]])
        b=Rotation.from_quat(np.asarray(right)[[1,2,3,0]])
        return (b*a.inv()).as_rotvec()
    namespace={'np':np,'time':time,'utils':SimpleNamespace(matrix_to_quat_wxyz=quaternion),'shortest_quaternion_rotation_vector':difference}
    state=SimpleNamespace(position=np.zeros(7),lower=np.full(7,-3.),upper=np.full(7,3.))
    fk=SimpleNamespace(set_robot_base_pose=lambda p,q:None,
        compute_forward_kinematics=lambda frame,q:(np.array([q[0],0.,0.]),Rotation.from_euler('z',q[6]).as_matrix()))
    expert=SimpleNamespace(arm_motion_state=lambda:state,_base_pose=lambda:(np.zeros(3),[1,0,0,0]),
        _pinch_offset_local=np.array([.1,0,0]),_ee_frame='tool',_kinematics=fk,_last_differential_jacobian='unchanged')
    return compile_native_pinch_jacobian(native_fixture_source(),namespace),expert,state


def test_native_geometry_extraction_uses_current_robot_and_skips_servo():
    fn,expert,state=native_fixture_robot();original=state.position.copy()
    actual=fn(expert)
    assert actual[0,0]==pytest.approx(1.)
    assert actual[1,6]==pytest.approx(.1,abs=1e-8)
    assert actual[5,6]==pytest.approx(1.)
    assert expert._last_differential_jacobian=='unchanged'
    np.testing.assert_array_equal(state.position,original)
    state.position[6]=np.pi/2
    changed=fn(expert)
    assert changed[0,6]==pytest.approx(-.1,abs=1e-8)
    assert abs(changed[1,6])<1e-8


def test_native_geometry_keeps_one_sided_joint_limit_difference():
    fn,expert,state=native_fixture_robot();state.position[6]=state.upper[6]
    assert fn(expert)[5,6]==pytest.approx(1.,abs=1e-8)


def test_native_geometry_rejects_unexpected_state_or_frozen_source():
    from capstone_vla.live_kinematics import compile_native_pinch_jacobian
    with pytest.raises(ValueError):compile_native_pinch_jacobian('unrelated=1')
    unsafe=native_fixture_source().replace('offset=np.asarray(self._pinch_offset_local,float)','offset=np.asarray(self.target_object_position,float)')
    with pytest.raises(ValueError,match='non-geometry'):compile_native_pinch_jacobian(unsafe)
    with pytest.raises(ValueError,match='step changed'):compile_native_pinch_jacobian(native_fixture_source().replace('1.e-4','1.e-5'))


def test_logged_task_feedforward_keeps_reference_identity_and_none():
    from capstone_vla.live_kinematics import task_feedforward_reference
    ref={'feedforward_task_acceleration':[0,0,-9.81,0,0,0]}
    assert task_feedforward_reference(ref,'logged') is ref
    assert task_feedforward_reference(None,'zero') is None


def test_zero_task_feedforward_preserves_other_commands_and_input():
    from capstone_vla.live_kinematics import task_feedforward_reference
    ref={'jacobian':np.eye(6,7),'commanded_task_velocity':np.arange(6),
         'feedforward_task_acceleration':np.array([0,0,-9.81,0,0,0]),
         'joint_velocity_constraint_matrix':np.eye(7)}
    original=copy.deepcopy(ref);zeroed=task_feedforward_reference(ref,'zero')
    assert not zeroed['feedforward_task_acceleration'].any()
    for key in ref:
        np.testing.assert_array_equal(ref[key],original[key])
        if key!='feedforward_task_acceleration':np.testing.assert_array_equal(zeroed[key],ref[key])
    # The existing executor extrapolates twist with this very same field.
    np.testing.assert_array_equal(zeroed['commanded_task_velocity']+zeroed['feedforward_task_acceleration']/180,ref['commanded_task_velocity'])


def test_unknown_task_feedforward_mode_is_rejected():
    from capstone_vla.live_kinematics import task_feedforward_reference
    with pytest.raises(ValueError):task_feedforward_reference(None,'invented')


@pytest.mark.parametrize('value',[[0]*3,[float('nan')]*6])
def test_zero_task_feedforward_rejects_invalid_reference_acceleration(value):
    from capstone_vla.live_kinematics import task_feedforward_reference
    with pytest.raises(ValueError):task_feedforward_reference({'feedforward_task_acceleration':value},'zero')


def test_zero_task_feedforward_reaches_physics_callback_with_audit(monkeypatch):
    state={'plan':{'tracking':'live-jacobian','jacobian_method':'native-pinch',
            'operational_orientation_length_scale_m':.03,'task_feedforward':'zero'},
        'native_jacobian_callable':lambda expert:np.eye(6,7),'live_jacobian_records':[]}
    monkeypatch.setattr(probe,'_STATE',state)
    ref={'jacobian':np.eye(6,7).tolist(),'commanded_task_velocity':list(range(6)),
         'feedforward_task_acceleration':[0,0,-9.81,0,0,0],'joint_velocity_limit_rad_s':[1.8]*7}
    actual=probe.live_reference(None,ref,15);state['active_live_reference']=actual
    applied=probe.physics_task_reference(None,None,None,None,None,None)
    assert not applied['feedforward_task_acceleration'].any()
    np.testing.assert_array_equal(applied['commanded_task_velocity'],ref['commanded_task_velocity'])
    np.testing.assert_array_equal(applied['joint_velocity_limit_rad_s'],ref['joint_velocity_limit_rad_s'])
    assert ref['feedforward_task_acceleration'][2]==-9.81
    record=state['task_feedforward_records'][0]
    assert record['original'][2]==-9.81 and not record['applied'].any()
