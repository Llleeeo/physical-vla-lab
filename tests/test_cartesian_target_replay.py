import copy
from types import SimpleNamespace
import numpy as np
import pytest
from capstone_vla.cartesian_target_trace import encode
from capstone_vla.cartesian_target_replay import CartesianTargetReplayer, audit_reexecution


class MeasuredExpert:
    def __init__(self):
        self.position = np.zeros(3)
        self.calls = 0
        self._pending_operational_space_reference = None

    def operational_space_pinch_action(self, target_position, target_quaternion, *, dt=1/60,
            joint_velocity_projector=None, cartesian_velocity_projector=None):
        self.calls += 1
        self.last_target = np.asarray(target_position).copy()
        self.last_callback = joint_velocity_projector
        velocity = np.r_[np.asarray(target_position)-self.position, [0., 0., 0., 0.]]
        self._pending_operational_space_reference = {'jacobian':np.eye(6,7), 'commanded_task_velocity':velocity[:6].copy()}
        return SimpleNamespace(joint_indices=np.arange(7), joint_positions=velocity.copy(), joint_velocities=velocity.copy())


def capture(callback=None):
    inputs=dict(target_position=[.1,.2,.3], target_quaternion=[1.,0.,0.,0.], dt=1/60,
        joint_velocity_projector=encode(callback), cartesian_velocity_projector=None)
    return dict(schema='physical-vla-lab.cartesian-target-capture.v1', frames=[0,1], calls=[dict(
        call_id=0,frame=0,completed=True,inputs=inputs,returned_action={'poison':True},
        pending_operational_reference={'poison':True})])


def execute(expert, replay):
    replay.begin_frame(expert, 0)
    result=expert.operational_space_pinch_action([99,99,99],[0,1,0,0])
    segment=dict(frame=0,goal_position=result.joint_positions,goal_velocity=result.joint_velocities,
        operational_space_reference=expert._pending_operational_space_reference)
    segment.update(replay.segment_links(0,segment))
    replay.begin_frame(expert,1)
    fallback=dict(frame=1,operational_space_reference=None,**replay.segment_links(1,{}))
    return result,[segment,fallback]


def test_saved_inner_commands_are_discarded_live_robot_state_changes_new_result():
    source=capture();a=MeasuredExpert();b=MeasuredExpert();b.position[:]=.04
    ra=CartesianTargetReplayer(source);rb=CartesianTargetReplayer(source)
    first,segments=execute(a,ra);second,_=execute(b,rb)
    np.testing.assert_allclose(first.joint_velocities[:3]-second.joint_velocities[:3],.04)
    np.testing.assert_array_equal(a.last_target,[.1,.2,.3])
    assert a.calls==b.calls==1
    assert not any('poison' in str(x) for x in ra.by_frame.values())
    audit=audit_reexecution(ra.payload(),source,segments,2)
    assert audit['reexecution_verified'] and not audit['standalone_cartesian_policy_verified']


def test_callables_are_native_live_objects_not_deserialized():
    def projector():pass
    projector.__module__='expert_data.collision_safety'
    projector.__qualname__='CollisionSafety._project_keep_clear_joint_velocity'
    expert=MeasuredExpert();replay=CartesianTargetReplayer(capture(projector));replay.begin_frame(expert,0)
    expert.operational_space_pinch_action([9,9,9],[1,0,0,0],joint_velocity_projector=projector)
    assert expert.last_callback is projector
    assert replay.events[0]['native_safety_callbacks_preserved']


def test_unknown_or_missing_safety_callback_stops_before_native_call():
    source=capture();source['calls'][0]['inputs']['joint_velocity_projector']={
        'callable_descriptor_only':True,'module':'evil','qualified_name':'callback'}
    with pytest.raises(ValueError,match='descriptor'):CartesianTargetReplayer(source)
    expert=MeasuredExpert();replay=CartesianTargetReplayer(capture());replay.begin_frame(expert,0)
    with pytest.raises(ValueError,match='callback differs'):
        expert.operational_space_pinch_action([0,0,0],[1,0,0,0],joint_velocity_projector=lambda:None)
    assert expert.calls==0


def test_missing_extra_or_out_of_order_calls_fail_closed():
    expert=MeasuredExpert();replay=CartesianTargetReplayer(capture());replay.begin_frame(expert,0)
    with pytest.raises(ValueError,match='Missing native'):replay.begin_frame(expert,1)
    expert.operational_space_pinch_action([0,0,0],[1,0,0,0])
    with pytest.raises(ValueError,match='Unexpected native'):expert.operational_space_pinch_action([0,0,0],[1,0,0,0])
    replay.begin_frame(expert,1)
    with pytest.raises(ValueError,match='exhausted'):replay.begin_frame(expert,2)


@pytest.mark.parametrize('field,value', [('target_position',[float('nan'),0,0]),('target_quaternion',[0,0,0,0]),('dt',1/30)])
def test_invalid_upper_goal_or_clock_rejected(field,value):
    source=capture();source['calls'][0]['inputs'][field]=value
    with pytest.raises(ValueError):CartesianTargetReplayer(source)


def test_partial_or_changed_reexecution_cannot_pass_audit():
    source=capture();expert=MeasuredExpert();replay=CartesianTargetReplayer(source)
    _,segments=execute(expert,replay);payload=replay.payload()
    changed=copy.deepcopy(payload);changed['calls'][0]['inputs']['target_position'][0]+=.001
    assert not audit_reexecution(changed,source,segments,2)['reexecution_verified']
    assert not audit_reexecution(payload,source,segments[:1],1)['reexecution_verified']
    assert source['calls'][0]['returned_action']=={'poison':True}
    replay.restore()
    expert.operational_space_pinch_action([.9,0,0],[1,0,0,0])
    np.testing.assert_array_equal(expert.last_target,[.9,0,0])
