"""Live kinematics must depend on the robot, not the reference Jacobian."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from capstone_vla.live_kinematics import live_pinch_jacobian
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
    monkeypatch.setattr(probe,'_STATE',{'live_jacobian_records':[]})
    reference={'jacobian':np.eye(6,7).tolist(),'commanded_task_velocity':list(range(6)),
        'feedforward_task_acceleration':[0,0,9.81,0,0,0],'task_position_error':[0]*6,
        'joint_velocity_constraint_matrix':np.eye(7).tolist(),
        'joint_velocity_constraint_lower_bound':[-1]*7}
    original=copy.deepcopy(reference);first=probe.live_reference(robot(),reference,15)
    np.testing.assert_array_equal(first['jacobian'],live_pinch_jacobian(robot()))
    for key in reference:
        if key!='jacobian':np.testing.assert_array_equal(first[key],reference[key])
    assert reference==original
    reference['jacobian']=np.full((6,7),99.).tolist()
    second=probe.live_reference(robot(),reference,16)
    np.testing.assert_array_equal(first['jacobian'],second['jacobian'])
    assert probe._STATE['live_jacobian_records'][0]['comparison_max_abs']!=probe._STATE['live_jacobian_records'][1]['comparison_max_abs']
    first['feedforward_task_acceleration'][2]=123
    assert reference['feedforward_task_acceleration'][2]==9.81


def test_joint_only_startup_does_not_query_missing_robot():
    assert probe.live_reference(None,None,0) is None


def test_live_reference_reaches_physics_callback_with_original_velocity_caps(monkeypatch):
    p={'tracking':'live-jacobian','mode':'joint'};actions=np.zeros((2,9));vel=np.ones((2,9));poses=np.zeros((2,7))
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
    np.testing.assert_array_equal(actual['jacobian'],live_pinch_jacobian(robot()))
    assert 'physics_rate_reference' not in actual  # Original 60 Hz update, held for three physics ticks.
    caps=probe.reference_policy_velocity_limit();np.testing.assert_allclose(caps,1.8)
    caps[0]=99;assert rows[1]['policy_joint_velocity_limit'][0]==1.8
    actual['jacobian'][0,0]=99
    assert probe.physics_task_reference(None,None,None,None,None,None)['jacobian'][0,0]!=99
    assert len(state['live_jacobian_records'])==1
