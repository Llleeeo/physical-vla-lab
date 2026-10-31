import copy
import json
import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from capstone_vla.ball_approach_guard import SETTINGS,ApproachGuard
from capstone_vla.ball_action_contract import decode_action
from capstone_vla.pilot_dataset import CAMERAS


def plan():
    return dict(settings=copy.deepcopy(SETTINGS),ball_radius_m=.023,
        finger_bounds=[dict(path='/robot/'+str(i),local_corners_m=np.zeros((8,3)).tolist()) for i in range(2)],
        cameras=[dict(name=n,prim_path='/camera/'+n,intrinsics=[[320,0,320],[0,320,240],[0,0,1]]) for n in CAMERAS])


def inputs():
    raw=np.zeros(29);raw[:3]=[.05,0,.01];raw[3]=1;raw[7:13]=-.1;raw[13]=1;raw[14:17]=-.1;raw[17]=1;raw[28]=.6
    pose=np.array([.05,0,.12,1,0,0,0]);points=np.array([[0,0,.04],[.1,0,.04]])
    corners=np.repeat(points[:,None,:],8,axis=1)
    return raw,pose,points,corners,dict(valid=True,position_world_m=[0,0,.023])


def test_unaligned_floor_clears_full_bounds_and_cancels_downward_feedforward_only():
    raw,pose,points,corners,detection=inputs();original=raw.copy()
    result,event=ApproachGuard(plan()).apply(raw,0,pose,points,corners,detection)
    assert event['mode']=='unaligned_height_floor'
    assert result[2]>=.151-1e-12
    np.testing.assert_array_equal(result[[9,12,16]],[0,0,0])
    np.testing.assert_array_equal(result[[0,1,7,8,10,11,14,15,28]],raw[[0,1,7,8,10,11,14,15,28]])
    np.testing.assert_array_equal(raw,original)
    assert decode_action(result,0,np.zeros(7))['gripper_closed']


def test_aligned_pass_preserves_all_model_channels():
    raw,pose,points,corners,detection=inputs();detection['position_world_m'][0]=.05
    result,event=ApproachGuard(plan()).apply(raw,0,pose,points,corners,detection)
    np.testing.assert_array_equal(result,raw);assert not event['changed_channels']


def test_tilted_goal_is_evaluated_before_height_floor():
    raw,pose,points,corners,detection=inputs();corners[:,:,2]+=.12
    raw[3:7]=[0,1,0,0]
    result,event=ApproachGuard(plan()).apply(raw,0,pose,points,corners,detection)
    desired=Rotation.from_quat(raw[[4,5,6,3]])
    proposed=result[:3]+desired.apply(corners.reshape(-1,3)-pose[:3])
    assert proposed[:,2].min()>=event['finger_floor_m']-1e-12


def test_missing_latches_pose_and_requires_consistent_reacquisition():
    raw,pose,points,corners,detection=inputs();guard=ApproachGuard(plan())
    result,event=guard.apply(raw,0,pose,points,corners,dict(valid=False))
    np.testing.assert_array_equal(result[:7],pose);assert result[28]==raw[28]
    assert np.all(result[7:13]==0) and np.all(result[14:17]==0)
    moved=pose.copy();moved[:3]+=.02
    second,event=guard.apply(raw,1,moved,points,corners,dict(valid=False))
    np.testing.assert_array_equal(second[:7],pose)
    third,event=guard.apply(raw,2,moved,points,corners,detection)
    assert event['vision_reason']=='reacquiring'
    fourth,event=guard.apply(raw,3,moved,points,corners,detection)
    assert event['vision_valid'] and event['mode']=='unaligned_height_floor'


def test_descent_reserve_and_visual_jump_and_clock_rejection():
    raw,pose,points,corners,detection=inputs();guard=ApproachGuard(plan())
    _,first=guard.apply(raw,0,pose,points,corners,detection)
    _,second=guard.apply(raw,1,pose,points,corners-.001,detection)
    assert second['descent_reserve_m']>0
    detection['position_world_m'][0]=1
    _,event=guard.apply(raw,2,pose,points,corners,detection)
    assert not event['vision_valid'] and event['vision_reason']=='visual_jump'
    with pytest.raises(ValueError,match='Noncontiguous'):guard.apply(raw,4,pose,points,corners,detection)


def test_runtime_records_raw_filtered_actions_and_robot_only_geometry(tmp_path,monkeypatch):
    from test_ball_live_policy import runtime_fixture,action
    from capstone_vla.ball_live_runtime import LiveExecutor
    import capstone_vla.ball_live_runtime as runtime
    from capstone_vla.live_policy_protocol import atomic_npz
    old,expert,world,policy,robot,sim=runtime_fixture(tmp_path,monkeypatch)
    cfg=dict(old.config,prefix=1,approach_guard=plan());executor=LiveExecutor(cfg)
    raw,pose,points,corners,detection=inputs()
    expert.finger_segment_world=lambda:points.copy()
    monkeypatch.setattr(runtime,'robot_corners',lambda stage,settings:corners.copy())
    monkeypatch.setattr(runtime,'locate_ball',lambda images,poses,intrinsics:dict(valid=False,reason='test_missing'))
    scores=[.49,.51,.49,.51,.49,.51]
    for i,score in enumerate(scores):
        a=action().astype(float);a[28]=score
        atomic_npz(tmp_path/'ipc'/f'response_{i:06d}.npz',frame=np.array(i),absolute=np.tile(a,(20,1)),inference_s=np.array(.1))
        sim.current_time=i/60
        executor.live_step(policy,world,i,1/60,None,sim,robot,np.zeros(9),np.zeros(9),np.array([0.,0.,.3]),np.array([1.,0,0,0]))
    executor.finish()
    assert expert.jaws==[False,True,False,True,False,True]
    assert not executor.payload()['simulator_target_state_used']
    events=json.loads((tmp_path/'approach_guard_events.json').read_text());assert len(events)==6
    with h5py.File(tmp_path/'live_observations.hdf5') as f:
        assert f.attrs['complete'] and f['robot_finger_corners'].shape==(6,2,8,3)
        np.testing.assert_array_equal(f['model_action'][:,28],scores)
        np.testing.assert_array_equal(f['action'][:,28],scores)
        np.testing.assert_array_equal(f['action'][0,:7],f['state'][0,14:21])
    with pytest.raises(ValueError,match='combine'):
        LiveExecutor(dict(cfg,vertical_diagnostic={'not':'supported'}))
