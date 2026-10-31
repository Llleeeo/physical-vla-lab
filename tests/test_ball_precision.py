import copy
import numpy as np
import pytest
from capstone_vla.ball_precision import metrics,accept


def streams():
    target=np.zeros((269,29));target[:,3]=1;target[144:,28]=1
    target[100:,0]=.05
    prediction=target.copy();prediction[:,0]+=.01
    return target,prediction


def test_continuity_matches_expert_increment_not_zero_motion():
    target,prediction=streams()
    m=metrics(prediction,target,np.zeros(7))
    assert m['precontact_80_119_mean_mm']==pytest.approx(10)
    assert m['precontact_increment_error_mm']==pytest.approx(0)
    prediction[100:,0]-=.05
    assert metrics(prediction,target,np.zeros(7))['precontact_increment_error_mm']>0


def base():
    return dict(approach_position_mean_mm=30.,precontact_80_119_mean_mm=20.,precontact_increment_error_mm=10.,
        full_position_p95_mm=50.,orientation_p95_deg=5.,gripper_error_frames=5,gripper_transitions=3,
        native_rejected_frames=0,nonfinite_frames=0,full_stream_metrics=dict(mode_accuracy={'gripper_closed':{'correct':264}}))


def improved():
    b=base();b.update(approach_position_mean_mm=20.,precontact_80_119_mean_mm=15.,precontact_increment_error_mm=7.)
    return b


def test_two_seeds_must_both_pass_and_best_must_improve():
    assert accept([improved(),improved()],[base(),base()],[base(),base()])
    assert not accept([improved(),base()],[base(),base()],[base(),base()])
    assert not accept([improved(),improved()],[base(),base()],[improved(),improved()])
    with pytest.raises(ValueError):accept([improved()],[base(),base()],[base(),base()])


@pytest.mark.parametrize('field,value',[
    ('precontact_increment_error_mm',9.),('precontact_80_119_mean_mm',19.),
    ('full_position_p95_mm',51.),('orientation_p95_deg',6.),('gripper_error_frames',6),
    ('gripper_transitions',4),('native_rejected_frames',1),('nonfinite_frames',1),
    ('approach_position_mean_mm',float('nan'))])
def test_position_improvement_cannot_hide_regression(field,value):
    c=improved();c[field]=value
    assert not accept([c,improved()],[base(),base()],[base(),base()])


def test_other_modes_are_guarded():
    c=improved();c['full_stream_metrics']['mode_accuracy']['gripper_closed']['correct']=263
    assert not accept([c,improved()],[base(),base()],[base(),base()])
