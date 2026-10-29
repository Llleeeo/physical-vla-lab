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
@pytest.mark.parametrize('render',[False,True])
def test_partial_history_optional_render_without_relaunching_physics(tmp_path,monkeypatch,analysis_error,render):
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
    assert runner.finish(out,1,render=render)==2
    report=json.loads((out/'report.json').read_text())
    assert report['saved_attempts']==[58] and report['complete'] is False
    assert report['cases'][0]['render_complete'] is render
    assert report['render_requested'] is render
    assert report['cases'][0]['analysis_complete'] is (not analysis_error)
    assert len(commands)==int(render)
    if render:assert commands[0][1].endswith('render_hgwind_effort_probe.py')
    else:assert report['cases'][0]['render_status']=='skipped_by_request'
    assert (episode/'replay.hdf5').read_bytes()==b'fixture'


@pytest.mark.parametrize('failure', [None, 'pending', 'update', 'clock', 'playing'])
def test_stage_load_wait_defers_scripts_and_restores_native_apply(tmp_path, failure):
    from capstone_vla.scene_load_barrier import wait_for_stage
    timeline = SimpleNamespace(is_stopped=lambda:True,get_current_time=lambda:0.)
    manager = SimpleNamespace(_prim_to_scripts={})
    calls=[]
    def native_apply(prim):calls.append(prim)
    manager._apply_scripts=native_apply
    class Barrier:
        _released=False
        _manager=manager
        def hold(self):pass
    barrier=Barrier();manager._load_all_scripts=barrier.hold
    updates=[]
    def update():
        manager._apply_scripts('must-stay-deferred')
        updates.append(1)
        if failure=='update':raise ValueError('attach failed')
        if failure=='clock':timeline.get_current_time=lambda:.01
        if failure=='playing':timeline.is_stopped=lambda:False
    context=SimpleNamespace(get_stage_loading_status=lambda:('',1,1 if failure=='pending' else 0))
    app=SimpleNamespace(update=update)
    path=tmp_path/'stage.json'
    if failure:
        with pytest.raises((RuntimeError,ValueError,TimeoutError)):
            wait_for_stage(context,app,manager,barrier,timeline,path,max_updates=3)
    else:
        report=wait_for_stage(context,app,manager,barrier,timeline,path,max_updates=3)
        assert report['ready'] is True and report['updates']==2
        assert report['initial_timeline_time']==report['final_timeline_time']==0
    assert manager._apply_scripts is native_apply and calls==[]
    result=json.loads(path.read_text())
    assert result['ready'] is (failure is None)
    assert result['deferred_apply_calls']==len(updates)
    if failure:assert 'error' in result
    manager._apply_scripts('restored');assert calls==['restored']


def test_stage_load_wait_rejects_started_scripts_before_any_update(tmp_path):
    from capstone_vla.scene_load_barrier import wait_for_stage
    manager=SimpleNamespace(_prim_to_scripts={'/fan':{'script':object()}})
    barrier=SimpleNamespace(_released=False,_manager=manager)
    with pytest.raises(RuntimeError,match='precede'):
        wait_for_stage(None,None,manager,barrier,SimpleNamespace(is_stopped=lambda:True),tmp_path/'unused')
    assert list(tmp_path.iterdir())==[]


def test_stage_wait_patch_precedes_robot_mutation_and_rejects_missing_anchor():
    from capstone_vla.scene_load_barrier import patch_scene_load_barrier
    source="""def load():
    _st0 = omni.usd.get_context().get_stage()
    if _st0 is None:
        raise RuntimeError()
    _strip_robot_script(_st0, '/robot')
def _inspect_loaded_scripts(_st0, robot_script_removed, allow_no_scripts):
    pass
"""
    patched=patch_scene_load_barrier(source)
    assert patched.index('    CAPSTONE_WAIT_STAGE_LOAD(')<patched.index('    _strip_robot_script(')
    with pytest.raises(ValueError,match='boundary'):patch_scene_load_barrier('pass')



def test_fresh_control_requires_explicit_startup_match_and_no_prior_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(history,'_PLAN',None);monkeypatch.setattr(history,'_SEEN',[])
    monkeypatch.setattr(force,'_ACTIVE',None);monkeypatch.setattr(trace,'_STATE',{})
    case=dict(output=str(tmp_path/'attempt60'),trace_cartesian_targets=False)
    plan=dict(history_attempts=[60],history_cases={'60':case},output=str(tmp_path))
    with pytest.raises(ValueError,match='matched'):history.configure(plan)
    plan.update(fresh_control=True,stage_load_barrier=True)
    history.configure(plan)
    with pytest.raises(ValueError,match='order'):history.begin_attempt(58)
    assert not (tmp_path/'attempt60').exists()
    history.begin_attempt(60)
    assert json.loads((tmp_path/'attempt_order.jsonl').read_text())=={'attempt_index':60,'ordinal':1}
    assert trace._STATE['plan']['output']==case['output']
    with pytest.raises(ValueError,match='order'):history.begin_attempt(60)


def test_fresh_index_does_not_silently_accept_history_prefix(tmp_path):
    episode=put_episode(tmp_path,60,'traj_00000')
    assert history.index_saved_episodes(tmp_path,'task',expected=[60])=={60:episode}
    with pytest.raises(ValueError,match='earlier'):history.index_saved_episodes(tmp_path,'task')
    put_episode(tmp_path,58,'failed_00000')
    with pytest.raises(ValueError,match='Unexpected'):history.index_saved_episodes(tmp_path,'task',expected=[60])


@pytest.mark.parametrize('omission_verified', [None, True, False])
@pytest.mark.parametrize('worker_code,barrier_ready,analysis_error', [(0, True, False), (1, True, False), (1, False, False), (1, True, True)])
def test_fresh_finish_compares_both_controls_without_relaunch(tmp_path,monkeypatch,worker_code,barrier_ready,analysis_error,omission_verified):
    import hashlib,importlib.util
    root=Path(__file__).resolve().parents[1];monkeypatch.syspath_prepend(str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('fresh_runner_test',root/'scripts/run_hgwind_history_diagnostic.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    import analyze_hgwind_effort_probe as analyzer
    out=tmp_path/'fresh';worker=out/'worker_output';case=out/'attempt_00060'
    episode=worker/'1s1p'/runner.TASK_ID/'failed_00000';episode.mkdir(parents=True)
    meta={'attempt_index':60,'episode_seed':123,'trajectory_variation':{'base_seed':42}}
    (episode/'meta.json').write_text(json.dumps(meta))
    for name in ['replay.hdf5','trajectory.hdf5']:(episode/name).write_bytes(b'fixture')
    history_ref=tmp_path/'matched';(history_ref/'attempt_00060').mkdir(parents=True)
    (history_ref/'report.json').write_text('{}')
    (history_ref/'attempt_00060/report.json').write_text(json.dumps({'probe_episode':str(tmp_path/'history60')}))
    old_fresh=tmp_path/'old_fresh';old_fresh.mkdir();(old_fresh/'report.json').write_text(json.dumps({'probe_episode':str(tmp_path/'old60')}))
    monkeypatch.setattr(runner,'FRESH',old_fresh)
    case_plan=dict(output=str(case),source=str(episode/'replay.hdf5'),source_files=[],episode='traj_00001',expected_episode_seed=123,expected_variation=meta['trajectory_variation'])
    plan=dict(worker_output=str(worker),history_attempts=[60],fresh_control=True,stage_load_barrier=True,history_cases={'60':case_plan},matched_history_reference=str(history_ref),matched_history_report_sha256=hashlib.sha256(b'{}').hexdigest())
    if omission_verified is not None:
        from capstone_vla import setup_force_control
        plan['setup_force_control'] = {'fixture':True}
        monkeypatch.setattr(setup_force_control,'audit_intervention',lambda _:{'verified':omission_verified})
    (out/'plan.json').write_text(json.dumps(plan))
    (out/'attempt_order.jsonl').write_text(json.dumps({'attempt_index':60}))
    (out/'stage_load_barrier.json').write_text(json.dumps({'ready':barrier_ready,'held_script_instances':0,'initial_timeline_time':0.,'final_timeline_time':0.}))
    def analyze(_):
        if analysis_error:raise ValueError('unreadable comparison evidence')
        return {'independent_grasp_evidence':{'status':'fail'},'initial_state_match':True,'runtime_safety_passed':False}
    monkeypatch.setattr(analyzer,'analyze',analyze)
    comparisons=[]
    monkeypatch.setattr(runner,'compare_motion',lambda *a:(comparisons.append(a) or {'all_core_arrays_exact':False,'compared_native_rows':7}))
    monkeypatch.setattr(runner.subprocess,'run',lambda *a,**k:pytest.fail('no simulator or renderer should launch'))
    complete = worker_code == 0 and barrier_ready and not analysis_error and omission_verified is not False
    assert runner.finish(out,worker_code)==(0 if complete else 2)
    report=json.loads((out/'report.json').read_text())
    assert report['requested_attempts']==[60] and report['complete'] is complete
    assert report['condition']==('fresh60_matched_startup' if omission_verified is None else 'fresh60_single_setup_force_omission')
    assert report['physical_success'] is False
    assert report['target_grasp_status']==('not_evaluated' if analysis_error else 'fail')
    assert report['stage_load_barrier_verified'] is barrier_ready
    assert report['comparison_evaluable'] is (barrier_ready and not analysis_error and omission_verified is not False)
    if omission_verified is not None:assert report['setup_force_control']['verified'] is omission_verified
    if worker_code == 1 and barrier_ready and not analysis_error and omission_verified is None:
        assert 'evaluable despite worker failure' in report['interpretation']
    assert len(comparisons)==(0 if analysis_error else 3)
    if not analysis_error:assert comparisons[-1][0]==tmp_path/'history60/replay.hdf5'
    assert report['cases'][0]['render_status']=='skipped_by_request'
