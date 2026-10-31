import json
import sys
from types import SimpleNamespace
import h5py
import numpy as np
import pytest
from capstone_vla.ball_live_protocol import ActionQueue,decoded_request,validate_observation,patch_collector
from capstone_vla.ball_action_contract import SLICES
from capstone_vla.pilot_dataset import CAMERAS


def action():
    a=np.zeros(29,dtype=np.float32);a[:3]=[.1,.2,.3];a[3]=1
    return a


def test_absolute_queue_preserves_four_actions_and_truncates_terminal_chunk():
    queue=ActionQueue(4,6);chunk=np.tile(action(),(20,1));chunk[:,0]=np.arange(20)/10
    queue.accept(0,chunk)
    with pytest.raises(ValueError,match='Stale'):queue.accept(0,chunk)
    for i in range(4):
        value,origin=queue.pop(i)
        assert origin==0 and value[0]==chunk[i,0]
    with pytest.raises(ValueError,match='Stale'):queue.accept(0,chunk)
    queue.accept(4,chunk+0.01)
    assert queue.pop(4)[1]==4 and queue.pop(5)[1]==4 and not queue.values
    with pytest.raises(ValueError,match='clock'):queue.pop(6)


@pytest.mark.parametrize('bad',[np.zeros((20,7)),np.full((20,29),np.nan)])
def test_bad_response_never_enters_queue(bad):
    q=ActionQueue(4,8)
    with pytest.raises(ValueError):q.accept(0,bad)
    assert not q.values and q.next_frame==0


def test_native_semantics_reject_inconsistent_masks_without_repair():
    a=action();a[SLICES['relative_speed_enabled']]=1
    original=a.copy()
    with pytest.raises(ValueError,match='requires'):decoded_request(a,0,np.zeros(7))
    np.testing.assert_array_equal(a,original)
    a[SLICES['velocity_reference_enabled']]=1
    a[SLICES['linear_velocity_reference']]=[2,0,0]
    with pytest.raises(ValueError,match='absolute speed'):decoded_request(a,0,np.zeros(7))
    a[SLICES['linear_velocity_reference']]=[.1,0,0]
    packet=decoded_request(a,0,np.zeros(7))
    np.testing.assert_array_equal(packet['inputs']['target_position'],a[:3])
    assert not packet['terminal_cartesian_hold']


def test_live_schema_refuses_legacy_state_and_extra_camera():
    images={name:np.zeros((2,480,640,3),dtype=np.uint8) for name in CAMERAS}
    validate_observation(np.zeros((2,30)),images)
    with pytest.raises(ValueError):validate_observation(np.zeros((2,6)),images)
    with pytest.raises(ValueError):validate_observation(np.zeros((2,30)),dict(images,target_truth=np.zeros(2)))
    with pytest.raises(ValueError):validate_observation(np.full((2,30),np.nan),images)


def test_patch_replaces_task_step_and_passes_only_robot_sensor_inputs():
    source='''def worker():
    if True:
        if True:
            if True:
                decision = controller.step(frame, world, dt_control)
                if True:
                    dynamic_drive_runtime["segment"] = segment

                    def _hold_jaws():
                        pass
    meta={"skill_terminations": _jsonify(skill_terminations),}
'''
    patched=patch_collector(source)
    assert 'controller.step(' not in patched
    call=next(s for s in patched.splitlines() if 'decision = CAPSTONE_LIVE' in s)
    assert 'ee_p.copy()' in call and 'jp.copy()' in call
    assert 'state_arrays' not in call and 'objects' not in call
    assert 'CAPSTONE_NATIVE_TRACE_SUMMARY(pol)' in patched
    with pytest.raises(ValueError):patch_collector(patched)


def configuration(tmp_path,frames=6):
    return dict(output=str(tmp_path),frames=frames,prefix=4,pinch_offset_local_m=[0,0,.0123],
        camera_config_digest='test',robot_posture_configuration=dict(schema='physical-vla-lab.robot-posture-configuration.v1',
        ik_offset_rad=[0]*7,homotopy_enabled=False,homotopy_rank=0,strength=1.))


def runtime_fixture(tmp_path,monkeypatch):
    from test_cartesian_packets import setup
    from capstone_vla.ball_live_runtime import LiveExecutor
    from capstone_vla.live_policy_protocol import atomic_npz
    _,expert,owner,world,policy=setup(monkeypatch)
    cfg=configuration(tmp_path);(tmp_path/'ipc').mkdir()
    expert.arm_motion_state=lambda:SimpleNamespace(position=np.zeros(7),lower=np.full(7,-3.),upper=np.full(7,3.))
    expert._pinch_offset_local=np.array(cfg['pinch_offset_local_m'])
    original=expert.operational_space_pinch_action
    def native(**inputs):
        expert._last_differential_ik=dict(posture_target_rad=inputs['posture_target'],posture_source='preflight_endpoint')
        return original(**inputs)
    expert.operational_space_pinch_action=native
    robot=SimpleNamespace(get_joint_positions=lambda:np.zeros(9),get_joint_velocities=lambda:np.zeros(9))
    sim=SimpleNamespace(current_time=0.)
    class Camera:
        def __init__(self,*args):self.records=[];self.i=0;self.last_poses=np.tile([0,0,0,1,0,0,0],(2,1))
        def capture(self):
            images={n:np.full((480,640,3),self.i,dtype=np.uint8) for n in CAMERAS};self.i+=1;return images
    monkeypatch.setitem(sys.modules,'capstone_vla.closed_loop',SimpleNamespace(LiveRGB=Camera,publish_gui_physics_state=lambda:None))
    monkeypatch.setitem(sys.modules,'camera.config',SimpleNamespace(config_digest=lambda:'test'))
    for index in (0,4):
        chunk=np.tile(action(),(20,1));chunk[:,0]+=index
        atomic_npz(tmp_path/'ipc'/f'response_{index:06d}.npz',frame=np.array(index),absolute=chunk,inference_s=np.array(.1))
    return LiveExecutor(cfg),expert,world,policy,robot,sim


def test_live_runtime_records_real_history_and_executes_without_expert_steps(tmp_path,monkeypatch):
    executor,expert,world,policy,robot,sim=runtime_fixture(tmp_path,monkeypatch)
    for i in range(6):
        sim.current_time=i/60
        executor.live_step(policy,world,i,1/60,None,sim,robot,np.zeros(9),np.zeros(9),np.array([0.,0.,.3]),np.array([1.,0,0,0]))
    executor.finish()
    assert expert.arms==6 and expert.jaws==[False]*6
    payload=executor.payload();assert payload['learned_policy_evaluated'] and not payload['source_motion_and_gripper_stream_retained']
    assert payload['non_target_safety_geometry_paths']==['/fan'] and not payload['online_skill_step_called']
    with np.load(tmp_path/'ipc/request_000004.npz') as request:
        assert set(request.files)=={'frame','states',*CAMERAS}
        for name in CAMERAS:np.testing.assert_array_equal(request[name][:,0,0,0],[2,4])
        assert request['states'].shape==(2,30)
    with h5py.File(tmp_path/'live_observations.hdf5') as f:
        assert f.attrs['complete'] and f['action'].shape==(6,29)
        np.testing.assert_array_equal(f['action'][:,0],np.array([.1]*4+[4.1]*2,dtype=np.float32))
        assert np.isclose(f['state'][0,16],.3123)


def test_invalid_learned_action_retains_inputs_and_never_calls_motors(tmp_path,monkeypatch):
    from capstone_vla.live_policy_protocol import atomic_npz
    executor,expert,world,policy,robot,sim=runtime_fixture(tmp_path,monkeypatch)
    chunk=np.tile(action(),(20,1));chunk[0,SLICES['relative_speed_enabled']]=1
    atomic_npz(tmp_path/'ipc/response_000000.npz',frame=np.array(0),absolute=chunk,inference_s=np.array(.1))
    with pytest.raises(ValueError,match='requires'):
        executor.live_step(policy,world,0,1/60,None,sim,robot,np.zeros(9),np.zeros(9),np.zeros(3),np.array([1.,0,0,0]))
    executor.finish()
    assert expert.arms==0 and not executor.packets
    with h5py.File(tmp_path/'live_observations.hdf5') as f:
        assert len(f['action'])==1 and not f.attrs['complete']
    assert 'requires' in json.loads((tmp_path/'policy_timing.json').read_text())[0]['error']


def test_observation_audit_detects_rgb_or_state_history_drift():
    from capstone_vla.ball_live_protocol import audit_request_history
    states=np.zeros((2,30),dtype=np.float32)
    images={name:np.zeros((2,480,640,3),dtype=np.uint8) for name in CAMERAS}
    request=dict(frame=np.array(4),states=states.copy(),**{k:v.copy() for k,v in images.items()})
    audit_request_history(request,4,states,images)
    request[CAMERAS[0]][0,0,0,0]=1
    with pytest.raises(AssertionError):audit_request_history(request,4,states,images)
    request[CAMERAS[0]][0,0,0,0]=0;request['states'][0,0]=1
    with pytest.raises(AssertionError):audit_request_history(request,4,states,images)
