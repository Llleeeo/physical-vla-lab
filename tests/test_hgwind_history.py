"""History must preserve attempt identity, observation isolation and partial results."""
import json
from pathlib import Path
from types import SimpleNamespace
import h5py
import numpy as np
import pytest
from capstone_vla import hgwind_history as history
from capstone_vla import hgwind_expert_trace as trace
from capstone_vla import hgwind_startup_trace as force


def test_episode_recorders_rotate_without_nested_observers(tmp_path,monkeypatch):
    monkeypatch.setattr(history,'_PLAN',None);monkeypatch.setattr(history,'_SEEN',[])
    monkeypatch.setattr(force,'_ACTIVE',None);monkeypatch.setattr(trace,'_STATE',{})
    cases={str(i):dict(output=str(tmp_path/str(i)),trace_cartesian_targets=True) for i in [58,59,60]}
    history.configure(dict(history_attempts=[58,59,60],history_cases=cases,output=str(tmp_path)))
    native_calls=[]
    native=SimpleNamespace(apply_force_at_pos=lambda *a:native_calls.append(a))
    namespace={'get_physx_interface':lambda:native}
    exec('def on_update(self): return get_physx_interface()',namespace)
    fan=type('Fan',(),{'on_update':namespace['on_update']})()
    original=namespace['get_physx_interface']
    expert=SimpleNamespace(operational_space_pinch_action=lambda x:x,_pending_operational_space_reference=None)
    original_action=expert.operational_space_pinch_action
    for i in [58,59,60]:
        history.begin_attempt(i)
        assert namespace['get_physx_interface'] is original
        assert expert.operational_space_pinch_action is original_action
        assert trace._STATE['rows']==[] and force._ACTIVE.frame is None
        force._ACTIVE.sim=SimpleNamespace(current_time=.03,get_physics_dt=lambda:1/60)
        force._ACTIVE.install([('/fan','/fan_script.py',fan)])
        fan.on_update().apply_force_at_pos('/ball',[1,0,0],[0,0,0],'Force')
        recorder=trace._STATE['cartesian_recorder'];recorder.begin_frame(expert,0)
        assert expert.operational_space_pinch_action(3)==3
        assert len(recorder.events)==1
        force._ACTIVE.frame=600
    assert len(native_calls)==3
    assert [json.loads(x)['attempt_index'] for x in (tmp_path/'attempt_order.jsonl').read_text().splitlines()]==[58,59,60]
    assert all(len((tmp_path/str(i)/'startup_force_calls.jsonl').read_text().splitlines())==1 for i in [58,59,60])
    with pytest.raises(ValueError,match='Unexpected attempt'):history.begin_attempt(60)
    force._ACTIVE.restore();recorder.restore()


def test_history_rejects_skipped_attempt_before_creating_output(tmp_path,monkeypatch):
    monkeypatch.setattr(history,'_PLAN',None);monkeypatch.setattr(history,'_SEEN',[])
    history.configure(dict(history_attempts=[58,59,60],history_cases={},output=str(tmp_path)))
    with pytest.raises(ValueError,match='order'):history.begin_attempt(59)
    assert list(tmp_path.iterdir())==[]
    with pytest.raises(ValueError):history.configure(dict(history_attempts=[60]))


def put_episode(root,index,name):
    p=root/'1s1p/task'/name;p.mkdir(parents=True)
    (p/'meta.json').write_text(json.dumps({'attempt_index':index}));(p/'replay.hdf5').touch()
    return p


def test_index_retains_safety_abort_prefix_and_rejects_missing_or_duplicate(tmp_path):
    assert history.index_saved_episodes(tmp_path,'task')=={}
    p=put_episode(tmp_path,58,'failed_00000')
    assert history.index_saved_episodes(tmp_path,'task')=={58:p}
    q=put_episode(tmp_path,59,'traj_00000')
    assert list(history.index_saved_episodes(tmp_path,'task'))==[58,59]
    put_episode(tmp_path,59,'failed_00001')
    with pytest.raises(ValueError,match='duplicate'):history.index_saved_episodes(tmp_path,'task')


def test_missing_earlier_episode_is_not_complete_history(tmp_path):
    put_episode(tmp_path,60,'traj_00000')
    with pytest.raises(ValueError,match='earlier'):history.index_saved_episodes(tmp_path,'task')


def create_motion(path,reverse=False,delta=0.):
    with h5py.File(path,'w') as f:
        for key in ['actions','timestamp','is_first','is_last','is_terminal','obs/ee_pos']:f[key]=np.zeros((3,))
        names=['/other','/Root/Dynamics/ball_y9bssvn'];v=np.zeros((3,2,6));v[:,0]=100
        v[1:,1,0]=delta
        if reverse:names=names[::-1];v=v[:,::-1]
        f['evidence/native/inventory/rigid_body/prim_paths']=np.array(names,dtype=h5py.string_dtype())
        f['evidence/native/rigid_body/get_velocities/values']=v


def test_compare_ball_by_identity_and_detect_first_physics_step(tmp_path):
    a,b=tmp_path/'a.h5',tmp_path/'b.h5';create_motion(a);create_motion(b,reverse=True,delta=.004)
    result=history.compare_motion(a,b)
    assert result['all_core_arrays_exact'] is True
    assert result['first_native_velocity_difference']==1
    assert result['first_step_ball_velocity_delta_m_s']==[.004,0.,0.]


def test_patch_rejects_changed_boundary():
    with pytest.raises(ValueError,match='insertion point'):history.patch_history_collector('pass')


@pytest.mark.parametrize('analysis_error',[False,True])
def test_partial_history_renders_saved_failure_without_relaunching_physics(tmp_path,monkeypatch,analysis_error):
    import importlib.util
    import sys
    root=Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('history_runner_test',root/'scripts/run_hgwind_history_diagnostic.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    import analyze_hgwind_effort_probe as analyzer
    out=tmp_path/'run';worker=out/'worker_output';case=out/'attempt_00058'
    episode=worker/'1s1p'/runner.TASK_ID/'failed_00000';episode.mkdir(parents=True)
    meta={'attempt_index':58,'episode_seed':123,'trajectory_variation':{'base_seed':42}}
    (episode/'meta.json').write_text(json.dumps(meta))
    for name in ['replay.hdf5','trajectory.hdf5']:(episode/name).write_bytes(b'fixture')
    case_plan=dict(output=str(case),episode='failed_original',source=str(episode/'replay.hdf5'),source_files=[],expected_episode_seed=123,expected_variation={'base_seed':42})
    (out/'plan.json').write_text(json.dumps({'worker_output':str(worker),'history_cases':{'58':case_plan}}))
    def analyze(_):
        if analysis_error:raise ValueError('malformed trace')
        return dict(independent_grasp_evidence={'status':'fail'},initial_state_match=True,runtime_safety_passed=False)
    monkeypatch.setattr(analyzer,'analyze',analyze)
    monkeypatch.setattr(runner,'compare_motion',lambda *a:{'all_core_arrays_exact':False})
    commands=[]
    monkeypatch.setattr(runner.subprocess,'run',lambda command,**k:(commands.append(command) or SimpleNamespace(returncode=0)))
    monkeypatch.setattr(runner,'inspect_review_videos',lambda *a:{'videos':{}})
    assert runner.finish(out,1)==2
    report=json.loads((out/'report.json').read_text())
    assert report['saved_attempts']==[58] and report['complete'] is False
    assert report['cases'][0]['render_complete'] is True
    assert report['cases'][0]['analysis_complete'] is (not analysis_error)
    assert len(commands)==1 and commands[0][1].endswith('render_hgwind_effort_probe.py')
    assert (episode/'replay.hdf5').read_bytes()==b'fixture'
