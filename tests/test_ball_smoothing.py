import json
import numpy as np
import h5py
import pytest
from capstone_vla.ball_smoothing import smooth_actions


def test_identity_causality_and_channel_isolation():
    raw=np.random.default_rng(4).normal(size=(20,29))
    before=raw.copy()
    np.testing.assert_array_equal(smooth_actions(raw,1),raw)
    out=smooth_actions(raw,.25)
    np.testing.assert_array_equal(out[:,3:],raw[:,3:])
    np.testing.assert_array_equal(out[0],raw[0])
    changed=raw.copy();changed[10:,:3]+=100
    np.testing.assert_array_equal(smooth_actions(changed,.25)[:10],out[:10])
    np.testing.assert_array_equal(raw,before)


def test_filter_attenuates_alternation_but_delays_ramp():
    raw=np.zeros((40,29));raw[:,0]=np.arange(40)%2
    out=smooth_actions(raw,.25)
    assert np.max(abs(np.diff(out[10:,0])))<.2
    raw[:,0]=np.arange(40)*.001
    out=smooth_actions(raw,.25)
    assert .0029 < raw[-1,0]-out[-1,0] < .0031


@pytest.mark.parametrize('alpha',[0,-1,1.1,float('nan')])
def test_invalid_filter(alpha):
    with pytest.raises(ValueError):smooth_actions(np.zeros((5,29)),alpha)


def test_saved_runtime_needs_no_ipc_and_binds_motor_requests(tmp_path,monkeypatch):
    from test_ball_live_policy import runtime_fixture,action
    from capstone_vla.ball_recorded_runtime import RecordedExecutor
    from capstone_vla.hf_archive import sha256
    original,expert,world,policy,robot,sim=runtime_fixture(tmp_path,monkeypatch)
    for path in (tmp_path/'ipc').iterdir():path.unlink()
    (tmp_path/'ipc').rmdir()
    raw=np.tile(action(),(6,1));raw[:,0]=[.1,.2,.1,.2,.1,.2];raw[3:,28]=1
    path=tmp_path/'saved.npz';np.savez(path,actions=raw)
    cfg=dict(original.config,recorded_actions_path=str(path),recorded_actions_sha256=sha256(path),smoothing_alpha=.25)
    executor=RecordedExecutor(cfg)
    for i in range(6):
        sim.current_time=i/60
        executor.live_step(policy,world,i,1/60,None,sim,robot,np.zeros(9),np.zeros(9),np.array([0.,0.,.3]),np.array([1.,0,0,0]))
    executor.finish()
    assert expert.arms==6 and expert.jaws==[False]*3+[True]*3
    assert not (tmp_path/'ipc').exists()
    payload=executor.payload()
    assert payload['recorded_model_actions_used'] and payload['source_actions_used_for_control']
    assert not payload['learned_policy_evaluated'] and not payload['model_inference_used']
    packets=json.loads((tmp_path/'packet_requests.json').read_text())
    with h5py.File(tmp_path/'live_observations.hdf5') as f:
        assert f.attrs['complete']
        np.testing.assert_array_equal(f['action'][:],smooth_actions(raw,.25))
        for i,p in enumerate(packets):np.testing.assert_array_equal(p['inputs']['target_position'],f['action'][i,:3])
    np.savez(path,actions=raw+1)
    with pytest.raises(ValueError,match='changed'):RecordedExecutor(cfg)
