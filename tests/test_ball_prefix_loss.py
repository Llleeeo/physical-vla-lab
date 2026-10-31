import numpy as np
import pytest
from capstone_vla.ball_prefix_loss import temporal_weights


def test_uniform_and_prefix_mass():
    pad=np.zeros((2,20),bool)
    one=temporal_weights(pad,1);four=temporal_weights(pad,4)
    np.testing.assert_array_equal(one,np.ones((2,20)))
    assert four[:,:4].sum()/four.sum()==pytest.approx(.5)
    np.testing.assert_array_equal(four[:,4:],np.ones((2,16)))


def test_padding_never_receives_weight():
    pad=np.zeros((2,20),bool);pad[1,1:]=True
    w=temporal_weights(pad,4)
    assert not w[pad].any() and w.sum()==36 and w[1,0]==4


@pytest.mark.parametrize('factor',[0,-1,float('nan'),float('inf')])
def test_invalid_factor(factor):
    with pytest.raises(ValueError):temporal_weights(np.zeros((2,20),bool),factor)


@pytest.mark.parametrize('pad',[np.ones((1,20),bool),np.zeros((2,19),bool),np.zeros((2,20),float)])
def test_invalid_mask(pad):
    with pytest.raises(ValueError):temporal_weights(pad,4)
