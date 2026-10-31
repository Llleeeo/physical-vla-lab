import json
import h5py
import numpy as np
import pytest
from capstone_vla.ball_vertical_diagnostic import SCHEMA,apply_height,audit_height,validate_plan,validate_provenance


def plan():
    return dict(schema=SCHEMA,channel=2,coordinate_frame='world',unit='m',
                start_frame=2,stop_frame_exclusive=4,target_z_m=[.08,.09])


def test_window_only_changes_z_and_preserves_input_and_gripper():
    raw=np.arange(29,dtype=np.float32)/100;original=raw.copy()
    for frame in range(6):
        result=apply_height(raw,frame,plan(),6)
        expected=original.astype(float)
        if frame in (2,3):expected[2]=[.08,.09][frame-2]
        np.testing.assert_array_equal(result,expected)
        np.testing.assert_array_equal(raw,original)
        audit_height(raw,result,frame,plan(),6)
        result[28]+=1
        with pytest.raises(AssertionError):audit_height(raw,result,frame,plan(),6)
    np.testing.assert_array_equal(apply_height(raw,2,None,6),raw)


@pytest.mark.parametrize('field,value',[
    ('channel',28),('coordinate_frame','robot'),('unit','mm'),('start_frame',-1),
    ('stop_frame_exclusive',7),('target_z_m',[float('nan'),.09]),('target_z_m',[.08])])
def test_invalid_plan_fails_closed(field,value):
    with pytest.raises(ValueError):validate_plan({**plan(),field:value},6)


def test_hybrid_cannot_claim_unassisted_policy_provenance():
    cfg=dict(frames=6,vertical_diagnostic=plan(),source_actions_used_for_control=True,learned_policy_evaluated=False)
    assert validate_provenance(cfg)
    with pytest.raises(ValueError):validate_provenance({**cfg,'learned_policy_evaluated':True})
    with pytest.raises(ValueError):validate_provenance({**cfg,'source_actions_used_for_control':False})


def test_runtime_retains_raw_model_and_executes_only_height_intervention(tmp_path,monkeypatch):
    from test_ball_live_policy import runtime_fixture
    executor,expert,world,policy,robot,sim=runtime_fixture(tmp_path,monkeypatch)
    executor.vertical=plan()
    for i in range(6):
        sim.current_time=i/60
        executor.live_step(policy,world,i,1/60,None,sim,robot,np.zeros(9),np.zeros(9),np.array([0.,0.,.3]),np.array([1.,0,0,0]))
    executor.finish()
    assert expert.arms==6 and expert.jaws==[False]*6
    payload=executor.payload()
    assert payload['source_actions_used_for_control'] and not payload['learned_policy_evaluated']
    assert not payload['standalone_policy_evaluation']
    packets=json.loads((tmp_path/'packet_requests.json').read_text())
    with h5py.File(tmp_path/'live_observations.hdf5') as f:
        assert f.attrs['complete']
        for i in range(6):
            audit_height(f['model_action'][i],f['action'][i],i,plan(),6)
            assert packets[i]['inputs']['target_position'][2]==f['action'][i,2]
        assert f['action'][2,2]==.08 and f['action'][3,2]==.09
        np.testing.assert_array_equal(f['model_action'][:,2],np.full(6,np.float32(.3)))
