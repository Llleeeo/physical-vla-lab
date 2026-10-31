import numpy as np
import pytest
from capstone_vla.ball_refinement import generated_metrics,accept_candidate


def test_phase_units_and_no_artificial_selection():
    targets=np.zeros((269,29));targets[:,3]=1;targets[144:,28]=1
    predicted=targets.copy();predicted[:144,0]=.01
    baseline=generated_metrics(predicted,targets,np.zeros(7))
    assert baseline['approach_position_mean_mm']==pytest.approx(10)
    assert baseline['gripper_error_frames']==0 and baseline['gripper_transitions']==1
    assert baseline['native_rejected_frames']==0
    assert not accept_candidate(baseline,baseline,baseline)
    candidate=generated_metrics(targets,targets,np.zeros(7))
    assert accept_candidate(candidate,baseline,baseline)


@pytest.mark.parametrize('field,value',[
    ('gripper_error_frames',1),('gripper_transitions',2),('native_rejected_frames',1),
    ('orientation_p95_deg',2),('nonfinite_frames',1),('approach_position_mean_mm',float('nan'))])
def test_reject_regressions_even_when_position_improves(field,value):
    baseline=dict(approach_end_exclusive=144,approach_position_mean_mm=10.,orientation_p95_deg=1.,
                  gripper_error_frames=0,gripper_transitions=1,native_rejected_frames=0,nonfinite_frames=0)
    candidate={**baseline,'approach_position_mean_mm':5.,field:value}
    assert not accept_candidate(candidate,baseline,baseline)
