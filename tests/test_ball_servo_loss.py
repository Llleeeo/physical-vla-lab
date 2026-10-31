import numpy as np
import pytest
from capstone_vla.ball_servo_loss import command_numpy, future_pinch, command_metrics


def test_free_and_absolute_cap():
    np.testing.assert_allclose(command_numpy([.1,0,0],[.2,0,0],[0,0,0]),[1.1,0,0])
    np.testing.assert_allclose(command_numpy([1,1,0],[0,0,0],[0,0,0]),np.array([1,1,0])*1.7/np.sqrt(2))


def test_relative_ball_order_and_zero_cap():
    # Absolute then relative projection: [1.7,0,0] -> ball at [0,.5,0].
    center=np.array([0,.5,0]);v=np.array([1.7,0,0])-center
    np.testing.assert_allclose(command_numpy([1,0,0],[0,0,0],[0,0,0],center,.2),center+.2*v/np.linalg.norm(v))
    np.testing.assert_array_equal(command_numpy([1,0,0],[0,0,0],[0,0,0],center,0),center)


@pytest.mark.parametrize('reference,limit',[(None,.1),([2,0,0],.1),([0,0,0],-1),([0,0,0],float('nan'))])
def test_reject_invalid_native_context(reference,limit):
    with pytest.raises(ValueError):command_numpy([0,0,0],[0,0,0],[0,0,0],reference,limit)


def test_future_state_alignment_and_padding():
    s=np.zeros((269,30));s[:,14]=np.arange(269)
    pad=np.array([[False]*20,[False]+[True]*19])
    out=future_pinch(s,[0,268],pad)
    np.testing.assert_array_equal(out[0,:,0],np.arange(20));assert np.all(out[1,:,0]==268)
    with pytest.raises(ValueError):future_pinch(s,[0,268],np.zeros((2,20),bool))
    with pytest.raises(ValueError):future_pinch(s,[-1,268],pad)


def valid_actions():
    a=np.zeros((269,29));a[:,3]=1
    return a


def test_metrics_zero_for_exact_actions_and_rejects_not_hidden():
    a=valid_actions();s=np.zeros((269,30))
    m=command_metrics(a,a,s)
    assert m['full_command_mean_mps']==0 and not m['command_rejected_frames']
    bad=a.copy();bad[0,17]=bad[0,19]=1;bad[0,14]=2;bad[0,18]=1
    m=command_metrics(bad,a,s)
    assert m['command_rejected_frames']==[0] and m['full_command_mean_mps'] is None and m['near_command_mean_mps'] is None
