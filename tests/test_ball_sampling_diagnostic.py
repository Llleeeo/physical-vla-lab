from types import SimpleNamespace
import numpy as np
import pytest
from capstone_vla.ball_sampling_diagnostic import decompose_xyz,integration_steps,predict_preserving_noise


def test_empirical_decomposition():
    p=np.array([[[.001,0,0]],[[.003,0,0]]])
    m=decompose_xyz(p,np.zeros((1,3)))
    assert m['empirical_mean_bias_rms_mm']==pytest.approx(2)
    assert m['sampling_spread_rms_mm']==pytest.approx(1)
    assert m['total_error_rms_mm']==pytest.approx(np.sqrt(5))


def test_zero_spread_is_not_zero_error():
    m=decompose_xyz(np.ones((8,4,3))*.01,np.zeros((4,3)))
    assert m['sampling_spread_rms_mm']==pytest.approx(0)
    assert m['empirical_mean_bias_rms_mm']==pytest.approx(np.sqrt(3)*10)


@pytest.mark.parametrize('p,t',[(np.zeros((1,2,3)),np.zeros((2,3))),(np.zeros((2,0,3)),np.zeros((0,3))),(np.ones((2,3,2)),np.zeros((3,2))),(np.full((2,1,3),np.nan),np.zeros((1,3)))])
def test_invalid_inputs(p,t):
    with pytest.raises(ValueError):decompose_xyz(p,t)


class Model:
    def __init__(self,config):self.config=config
    def denoise_step(self,x):x+=1


class Policy:
    def __init__(self,shared=True):
        self.config=SimpleNamespace(num_steps=10)
        self.model=Model(self.config if shared else SimpleNamespace(num_steps=10));self.resets=0
    def reset(self):self.resets+=1
    def predict_action_chunk(self,inputs,noise):
        for _ in range(self.model.config.num_steps):self.model.denoise_step(noise)
        return noise


@pytest.mark.parametrize('shared',[True,False])
def test_mutating_sampler_gets_independent_noise_and_restores(shared):
    policy=Policy(shared);noise=np.zeros(3)
    np.testing.assert_array_equal(predict_preserving_noise(policy,{},noise,10),np.ones(3)*10)
    np.testing.assert_array_equal(predict_preserving_noise(policy,{},noise,30),np.ones(3)*30)
    np.testing.assert_array_equal(noise,np.zeros(3))
    assert policy.resets==2 and policy.config.num_steps==policy.model.config.num_steps==10
    assert 'denoise_step' not in vars(policy.model)


def test_exception_restores_config_and_method():
    policy=Policy()
    with pytest.raises(RuntimeError):
        with integration_steps(policy,30):raise RuntimeError('interrupted')
    assert policy.config.num_steps==10 and 'denoise_step' not in vars(policy.model)


def test_wrong_actual_call_count_fails_and_restores():
    policy=Policy()
    with pytest.raises(ValueError,match='denoise calls'):
        with integration_steps(policy,30):policy.model.denoise_step(np.zeros(1))
    assert policy.config.num_steps==10
