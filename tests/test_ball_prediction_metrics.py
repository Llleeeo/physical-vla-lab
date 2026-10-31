import numpy as np
import pytest
from capstone_vla.ball_prediction_metrics import summarize_stream,summarize_chunks,prefix_stream


def labels(n=5):
    x=np.zeros((n,29));x[:,3]=1;x[:,28]=[0]*(n//2)+[1]*(n-n//2)
    return x


def test_quaternion_sign_equivalence_and_mm_units():
    target=labels();pred=target.copy();pred[:,3]=-1;pred[:,0]+=.003
    result=summarize_stream(pred,target)
    assert result['position_error_mm']['mean']==pytest.approx(3)
    assert result['orientation_error_deg']['mean']==pytest.approx(0)
    assert result['decoder_rejected_frames']==0
    assert result['mode_accuracy']['gripper_closed']['accuracy']==1


def test_degenerate_quaternion_and_active_axis_rejected_not_hidden():
    target=labels();pred=target.copy();pred[0,3:7]=0
    pred[1,23]=1  # enabled but zero orientation axis
    result=summarize_stream(pred,target)
    assert result['orientation_invalid_frames']==1
    assert result['decoder_rejected_frames']==2
    assert result['orientation_error_deg']['count']==4


def test_nonfinite_counts_without_false_mode_accuracy():
    target=labels();pred=target.copy();pred[0,7]=np.nan
    result=summarize_stream(pred,target)
    assert result['nonfinite_frames']==1
    assert result['mode_accuracy']['gripper_closed']['accuracy']==.8
    assert result['predicted_gripper_transition_frames'] is None


def test_clip_and_nullable_flags_are_explicit():
    target=labels();pred=target.copy();pred[:,25]=.2;pred[:,18]=10
    pred[0,19]=1
    result=summarize_stream(pred,target)
    assert result['scalar_projection']['joint_limit_margin_rad']['clipped_frames']==5
    assert result['scalar_projection']['maximum_relative_linear_speed_mps']['clipped_frames']==1
    assert result['vector_error']['target_linear_acceleration']['count']==0


def test_prefix_uses_anchor_chunks_and_partial_tail():
    chunks=np.zeros((5,20,29))
    for i in range(5):
        for j in range(20):chunks[i,j]=i*100+j
    stream,anchors,horizons=prefix_stream(chunks,4)
    np.testing.assert_array_equal(stream[:,0],[0,1,2,3,400])
    np.testing.assert_array_equal(anchors,[0,0,0,0,4])
    np.testing.assert_array_equal(horizons,[0,1,2,3,0])
    with pytest.raises(ValueError):prefix_stream(chunks,0)


def test_metrics_include_end_frame_and_correct_gripper_threshold():
    target=labels();pred=target.copy();pred[-1,28]=.49
    result=summarize_stream(pred,target)
    assert result['mode_accuracy']['gripper_closed']['correct']==4
    chunks=np.broadcast_to(pred[:,None],(5,20,29)).copy()
    metrics=summarize_chunks(chunks,target)
    assert metrics['recorded_observation_prefixes']['1']['frames']==5
    assert metrics['physics_feasibility_verified'] is False
