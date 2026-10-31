import json
import h5py
import numpy as np
import pytest
from capstone_vla.ball_gripper_confirmation import PLAN, GripperConfirmation, audit_stream


def sequence(scores):
    raw=np.tile(np.arange(29,dtype=float),(len(scores),1));raw[:,28]=scores
    gate=GripperConfirmation(dict(PLAN))
    return raw,np.stack([gate.apply(a,i) for i,a in enumerate(raw)])


def test_consecutive_confirmation_hysteresis_and_no_input_mutation():
    scores=[.8,.79,.8,.8,.8,.5,.2,.199,.8,.8,.8]
    raw,executed=sequence(scores)
    np.testing.assert_array_equal(executed[:,28],[0,0,0,0,1,1,1,0,0,0,1])
    np.testing.assert_array_equal(raw[:,28],scores)
    np.testing.assert_array_equal(executed[:,:28],raw[:,:28])
    assert GripperConfirmation(PLAN).apply(raw[-1],0)[28]==0
    assert audit_stream(raw,executed,PLAN)['executed_transitions']==[4,7,10]


@pytest.mark.parametrize('channel',[0,28])
def test_audit_rejects_tampered_execution(channel):
    raw,executed=sequence([.9,.9,.9]);executed[1,channel]+=1
    with pytest.raises(AssertionError):audit_stream(raw,executed,PLAN)


def test_rejects_bad_plan_frames_and_values_before_advancing():
    with pytest.raises(ValueError):GripperConfirmation(dict(PLAN,close_threshold=.5))
    gate=GripperConfirmation(PLAN)
    with pytest.raises(ValueError):gate.apply(np.zeros(29),1)
    with pytest.raises(ValueError):gate.apply(np.full(29,np.nan),0)
    assert gate.next_frame==0 and gate.count==0
    gate.apply(np.zeros(29),0)
    with pytest.raises(ValueError):gate.apply(np.zeros(29),0)
    with pytest.raises(ValueError):audit_stream(np.empty((0,29)),np.empty((0,29)),PLAN)


def test_runtime_preserves_raw_predictions_and_executes_confirmed_jaws(tmp_path,monkeypatch):
    from test_ball_live_policy import runtime_fixture,action
    from capstone_vla.ball_live_runtime import LiveExecutor
    from capstone_vla.live_policy_protocol import atomic_npz
    old,expert,world,policy,robot,sim=runtime_fixture(tmp_path,monkeypatch)
    cfg=dict(old.config,prefix=1,gripper_confirmation=dict(PLAN))
    executor=LiveExecutor(cfg)
    scores=[.8,.8,.8,.3,.2,.19]
    raw=[]
    for i,score in enumerate(scores):
        a=action().astype(float);a[28]=score;raw.append(a.copy())
        atomic_npz(tmp_path/'ipc'/f'response_{i:06d}.npz',frame=np.array(i),absolute=np.tile(a,(20,1)),inference_s=np.array(.1))
        sim.current_time=i/60
        executor.live_step(policy,world,i,1/60,None,sim,robot,np.zeros(9),np.zeros(9),np.array([0.,0.,.3]),np.array([1.,0,0,0]))
    executor.finish()
    expected=[False,False,True,True,True,False]
    assert expert.jaws==expected and expert.arms==6
    payload=executor.payload()
    assert payload['gripper_confirmation']==PLAN and payload['policy_postprocessing']
    assert not payload['unmodified_model_actions'] and not payload['source_actions_used_for_control']
    assert not payload['online_skill_step_called']
    packets=json.loads((tmp_path/'packet_requests.json').read_text())
    assert [p['gripper_closed'] for p in packets]==expected
    with h5py.File(tmp_path/'live_observations.hdf5') as f:
        assert f.attrs['complete']
        np.testing.assert_array_equal(f['model_action'][:],raw)
        assert json.loads(f.attrs['gripper_confirmation_json'])==PLAN
        audit_stream(f['model_action'][:],f['action'][:],PLAN)
    with pytest.raises(ValueError,match='combine'):
        LiveExecutor(dict(cfg,vertical_diagnostic={'schema':'unused'}))
