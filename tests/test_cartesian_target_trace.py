from types import SimpleNamespace
import json
import numpy as np
import pytest
from capstone_vla.cartesian_target_trace import CartesianTargetRecorder,encode,audit_capture,patch_capture_collector


class ToyExpert:
    def __init__(self):self.calls=[];self._pending_operational_space_reference=None
    def operational_space_pinch_action(self,target_position,target_quaternion,*,gain=6.,projector=None):
        self.calls.append((target_position,target_quaternion,gain,projector))
        self._pending_operational_space_reference={'jacobian':np.eye(6,7),'commanded_task_velocity':np.r_[np.asarray(target_position,float),[0,0,0]]}
        self.last_result=SimpleNamespace(joint_indices=np.arange(7),joint_positions=np.zeros(7),joint_velocities=np.ones(7))
        return self.last_result


def test_recorder_preserves_original_objects_defaults_and_return_identity():
    expert=ToyExpert();recorder=CartesianTargetRecorder();original=expert.operational_space_pinch_action
    recorder.begin_frame(expert,0);target=np.array([.1,.2,.3]);quat=np.array([1.,0.,0.,0.])
    result=expert.operational_space_pinch_action(target,quat)
    assert result is expert.last_result
    assert len(expert.calls)==1 and expert.calls[0][0] is target and expert.calls[0][1] is quat
    assert recorder.events[0]['inputs']['gain']==6.
    np.testing.assert_array_equal(result.joint_velocities,1)
    recorder.restore();assert expert.operational_space_pinch_action==original


def test_capture_is_snapshot_and_does_not_consume_pending_control():
    expert=ToyExpert();recorder=CartesianTargetRecorder();recorder.begin_frame(expert,0);target=np.ones(3)
    result=expert.operational_space_pinch_action(target,[1,0,0,0]);target[:]=99
    result.joint_positions[:]=99
    np.testing.assert_array_equal(recorder.events[0]['inputs']['target_position'],1)
    np.testing.assert_array_equal(recorder.events[0]['returned_action']['joint_positions'],0)
    assert expert._pending_operational_space_reference is not None


def test_callable_is_described_without_executing_or_serializing_its_owner():
    seen=[]
    def projector(*args):seen.append(args);raise AssertionError('not called by recorder')
    expert=ToyExpert();recorder=CartesianTargetRecorder();recorder.begin_frame(expert,0)
    expert.operational_space_pinch_action([0,0,0],[1,0,0,0],projector=projector)
    assert not seen and expert.calls[0][3] is projector
    assert recorder.events[0]['inputs']['projector']['callable_descriptor_only']
    json.dumps(recorder.payload(),allow_nan=False)


def test_original_exception_is_rethrown_unchanged_and_logged():
    error=RuntimeError('original failure')
    class Broken:
        def operational_space_pinch_action(self,*args,**kwargs):raise error
    expert=Broken();recorder=CartesianTargetRecorder();recorder.begin_frame(expert,0)
    with pytest.raises(RuntimeError) as caught:expert.operational_space_pinch_action()
    assert caught.value is error
    assert recorder.events[0]['exception_type']=='RuntimeError' and not recorder.events[0]['completed']


def test_multi_call_linking_uses_selected_output_and_reports_missing_alignment():
    expert=ToyExpert();recorder=CartesianTargetRecorder();recorder.begin_frame(expert,0)
    expert.operational_space_pinch_action([0,0,0],[1,0,0,0])
    expert.operational_space_pinch_action([1,0,0],[1,0,0,0])  # Same arm tuple, different request/reference object.
    segment={'frame':0,'goal_position':[0]*7,'goal_velocity':[1]*7,'operational_space_reference':expert._pending_operational_space_reference}
    segment.update(recorder.segment_links(0,segment))
    assert segment['matching_cartesian_call_ids']==[0,1]
    assert segment['selected_cartesian_call_ids']==[1]
    audit=audit_capture(recorder.payload(),[segment],1)
    assert audit['capture_alignment_passed'] and not audit['standalone_cartesian_replay_verified']
    segment['goal_velocity']=[2]*7
    assert not audit_capture(recorder.payload(),[segment],1)['capture_alignment_passed']


def test_none_fallback_frame_is_not_fabricated_into_cartesian_target():
    expert=ToyExpert();recorder=CartesianTargetRecorder();recorder.begin_frame(expert,0)
    expert.operational_space_pinch_action([0,0,0],[1,0,0,0])
    segment={'frame':0,'goal_position':[0]*7,'goal_velocity':[1]*7,'operational_space_reference':expert._pending_operational_space_reference}
    segment.update(recorder.segment_links(0,segment));recorder.begin_frame(expert,1)
    fallback={'frame':1,'operational_space_reference':None,**recorder.segment_links(1,{})}
    assert not fallback['cartesian_call_ids']
    assert audit_capture(recorder.payload(),[segment,fallback],2)['capture_alignment_passed']


def test_capture_rejects_unknown_worker_boundary_and_preserves_statement_order():
    with pytest.raises(ValueError):patch_capture_collector('unrelated=1')
    text='                policy_step_started = time.perf_counter()\n                decision = controller.step(frame, world, dt_control)'
    text='def fake_worker():\n    if True:\n        if True:\n            if True:\n'+text
    patched=patch_capture_collector(text)
    assert patched.index('CAPSTONE_BEGIN')<patched.index('policy_step_started')<patched.index('decision =')


def test_nonfinite_or_unknown_values_are_explicit_not_invalid_json():
    value=encode({'a':np.float64('nan'),'b':object(),'c':np.int64(2)})
    assert value['a']['unsupported_type']=='nonfinite_float' and 'unsupported_type' in value['b']
    json.dumps(value,allow_nan=False)


def test_native_consumption_copy_matches_without_identity(monkeypatch):
    import copy
    expert=ToyExpert();recorder=CartesianTargetRecorder();recorder.begin_frame(expert,0)
    expert.operational_space_pinch_action([0,0,0],[1,0,0,0])
    original=expert._pending_operational_space_reference
    copied=copy.deepcopy(original)
    assert copied is not original
    row={'frame':0,'goal_position':[0]*7,'goal_velocity':[1]*7,'operational_space_reference':copied}
    row.update(recorder.segment_links(0,row))
    assert row['selected_cartesian_call_ids']==[0]
    assert audit_capture(recorder.payload(),[row],1)['capture_alignment_passed']


def test_old_empty_identity_links_are_recoverable_from_exact_recorded_contents():
    import copy
    expert=ToyExpert();recorder=CartesianTargetRecorder();recorder.begin_frame(expert,0)
    expert.operational_space_pinch_action([0,0,0],[1,0,0,0])
    row={'frame':0,'goal_position':[0]*7,'goal_velocity':[1]*7,'operational_space_reference':copy.deepcopy(expert._pending_operational_space_reference),'selected_cartesian_call_ids':[]}
    result=audit_capture(recorder.payload(),[row],1)
    assert result['capture_alignment_passed'] and result['recovered_selection_frames']==[0]
    assert row['selected_cartesian_call_ids']==[]  # Audit does not rewrite evidence.


def test_duplicate_full_references_remain_ambiguous():
    expert=ToyExpert();recorder=CartesianTargetRecorder();recorder.begin_frame(expert,0)
    for _ in range(2):expert.operational_space_pinch_action([0,0,0],[1,0,0,0])
    row={'frame':0,'goal_position':[0]*7,'goal_velocity':[1]*7,'operational_space_reference':expert._pending_operational_space_reference}
    row.update(recorder.segment_links(0,row))
    assert row['reference_matching_cartesian_call_ids']==[0,1] and not row['selected_cartesian_call_ids']
    assert not audit_capture(recorder.payload(),[row],1)['capture_alignment_passed']


def test_full_reference_mismatch_and_nonfinite_cannot_create_a_link():
    import copy
    from capstone_vla.cartesian_target_trace import reference_content_equal
    expert=ToyExpert();recorder=CartesianTargetRecorder();recorder.begin_frame(expert,0)
    expert.operational_space_pinch_action([0,0,0],[1,0,0,0])
    wrong=copy.deepcopy(expert._pending_operational_space_reference);wrong['jacobian'][0,0]+=1e-12
    row={'frame':0,'goal_position':[0]*7,'goal_velocity':[1]*7,'operational_space_reference':wrong}
    assert not recorder.segment_links(0,row)['selected_cartesian_call_ids']
    assert not reference_content_equal([float('nan')],[float('nan')])
    assert not reference_content_equal({'unsupported_type':'nonfinite_float'},{'unsupported_type':'nonfinite_float'})


def test_native_segment_without_frame_links_without_mutating_control_input():
    expert=ToyExpert();recorder=CartesianTargetRecorder();recorder.begin_frame(expert,0)
    expert.operational_space_pinch_action([0,0,0],[1,0,0,0])
    segment=dict(goal_position=[0]*7,goal_velocity=[1]*7,
        operational_space_reference=expert._pending_operational_space_reference)
    links=recorder.segment_links(0,segment)
    assert links['selected_cartesian_call_ids']==[0]
    assert 'frame' not in segment


def test_explicit_conflicting_frame_cannot_be_silently_overwritten():
    expert=ToyExpert();recorder=CartesianTargetRecorder();recorder.begin_frame(expert,0)
    with pytest.raises(ValueError,match='frame differs'):
        recorder.segment_links(0,dict(frame=1))


def test_actual_record_segment_callback_accepts_unframed_native_segment():
    from capstone_vla import hgwind_expert_trace as trace
    expert=ToyExpert();recorder=CartesianTargetRecorder();recorder.begin_frame(expert,0)
    expert.operational_space_pinch_action([0,0,0],[1,0,0,0])
    segment=dict(goal_position=[0]*7,goal_velocity=[1]*7,
        operational_space_reference=expert._pending_operational_space_reference,
        policy_joint_velocity_limit=None,duration_s=1/60,episode_time_s=0.)
    saved=dict(trace._STATE)
    try:
        trace._STATE.clear();trace._STATE.update(rows=[],assistance=[],joints=[],cartesian_recorder=recorder)
        trace.record_segment(0,segment,np.zeros(9),np.zeros(9),np.zeros(9),None,'robot',[],{})
        assert trace._STATE['rows'][0]['selected_cartesian_call_ids']==[0]
        assert trace._STATE['rows'][0]['frame']==0 and 'frame' not in segment
    finally:
        trace._STATE.clear();trace._STATE.update(saved)
