from types import SimpleNamespace
import numpy as np
import pytest
from capstone_vla.ball_camera_protocol import CLAMP_KEYS, RESET_KEY, formal_capture, validated_clamps
from capstone_vla.ball_input_metrics import position_change_mm


def test_position_change_units():
    a=np.array([[0.003,0.004,0.],[0.,0.,0.01]])
    np.testing.assert_allclose(position_change_mm(a,np.zeros_like(a)),[5.,10.])


@pytest.mark.parametrize('fails',[False,True])
def test_render_boundary_reset_and_zero_time(fails):
    settings={};events=[]
    class Settings:
        def set(self,key,value):settings[key]=value;events.append((key,value))
    def step(**kwargs):
        assert kwargs==dict(delta_time=0.,pause_timeline=False,rt_subframes=4)
        assert settings[RESET_KEY] is True
        assert all(settings[k]==2. for k in CLAMP_KEYS)
        if fails:raise RuntimeError('render failed')
    rep=SimpleNamespace(orchestrator=SimpleNamespace(step=step))
    if fails:
        with pytest.raises(RuntimeError):formal_capture(rep,{'temporal_subframes':4},Settings(),{'applied':dict.fromkeys(CLAMP_KEYS,2.)})
    else:formal_capture(rep,{'temporal_subframes':4},Settings(),{'applied':dict.fromkeys(CLAMP_KEYS,2.)})
    assert settings[RESET_KEY] is False
    assert len(events)==4


@pytest.mark.parametrize('value',[float('nan'),float('inf'),0.,-1.])
def test_invalid_calibration(value):
    with pytest.raises(ValueError):validated_clamps({'applied':dict.fromkeys(CLAMP_KEYS,value)})


def test_no_extra_renderer_settings():
    with pytest.raises(ValueError):validated_clamps({'applied':{**dict.fromkeys(CLAMP_KEYS,1.),'physics_dt':.1}})
