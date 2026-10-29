"""Read actual live drive state while preserving native calls and exceptions."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from capstone_vla import gripper_drive_trace as trace


@pytest.fixture
def expert(monkeypatch):
    gain=np.ones((1,2))*1500.;damping=np.ones((1,2))*40.;calls=[]
    def read_gain(**kwargs):
        calls.append(('get_gain',kwargs));return gain.copy(),damping.copy()
    view=SimpleNamespace(is_physics_handle_valid=lambda:True,get_gains=read_gain,
        get_max_efforts=lambda **k:np.array([[70.,70.]]),
        get_joint_max_velocities=lambda **k:np.array([[.065,.065]]))
    token=object();e=SimpleNamespace(_last_gripper_close=False)
    def force(*args,**kwargs):
        calls.append(('native_force',args,kwargs));gain[:]=3500.;return token
    def grip(*args,**kwargs):
        calls.append(('native_grip',args,kwargs));e._last_gripper_close=kwargs.get('close',True);return token
    controller=SimpleNamespace(get_applied_action=lambda:SimpleNamespace(
        joint_positions=np.arange(9)/100.,joint_velocities=None,joint_efforts=None),
        get_gains=lambda:pytest.fail('do not use cached/fallback controller gains'))
    robot=SimpleNamespace(dof_names=['arm'+str(i) for i in range(7)]+list(trace.FINGERS),
        _articulation_view=view,get_joint_positions=lambda:np.arange(9)/100.,
        get_joint_velocities=lambda:np.zeros(9),get_articulation_controller=lambda:controller)
    e._robot=robot;e._prim=SimpleNamespace(GetStage=lambda:None,GetPath=lambda:'/robot')
    e._gripper_dof_indices=lambda:[7];e.ensure_gripper_force=force;e.gripper_action=grip
    monkeypatch.setattr(trace,'usd_fingers',lambda *a:[{'path':'joint1'},{'path':'joint2'}])
    e.test=SimpleNamespace(gain=gain,calls=calls,token=token,view=view,controller=controller)
    return e


def test_native_readbacks_are_selected_by_identity_and_copied(expert):
    row=trace.read_drive(expert)
    assert row['joint_indices']==[7,8] and row['expert_driven_indices']==[7]
    assert row['live_gains'][0]==[[1500.,1500.]]
    assert row['live_max_efforts']==[[70.,70.]]
    expert.test.gain[:]=999
    assert row['live_gains'][0]==[[1500.,1500.]]
    assert all(c[0]!='native_force' for c in expert.test.calls)


def test_invalid_physics_handle_never_substitutes_usd_as_live_gain(expert):
    expert.test.view.is_physics_handle_valid=lambda:False
    row=trace.read_drive(expert)
    assert row['live_physics_handle_valid'] is False and 'live_gains' not in row
    assert 'live_properties' in row['errors']
    assert not any(c[0]=='get_gain' for c in expert.test.calls)


def test_wrappers_preserve_native_arguments_return_and_gain_changes(tmp_path,expert):
    path=tmp_path/'trace.jsonl';recorder=trace.DriveRecorder(path)
    original_force=expert.ensure_gripper_force;original_grip=expert.gripper_action
    recorder.emit('usd_before_reset',{'usd_fingers':[]})
    recorder.observe(expert,0);recorder.observe(expert,154)
    args=(70.,);kwargs={'speed':.065,'reconfigure':True}
    assert expert.ensure_gripper_force(*args,**kwargs) is expert.test.token
    assert expert.gripper_action(.0,close=True) is expert.test.token
    calls=[x for x in expert.test.calls if x[0]=='native_force'];assert len(calls)==1
    assert calls[0][1]==args and calls[0][2]==kwargs
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    assert next(x for x in rows if x['phase']=='before_ensure_force')['snapshot']['live_gains'][0]==[[1500.,1500.]]
    assert next(x for x in rows if x['phase']=='after_ensure_force')['snapshot']['live_gains'][0]==[[3500.,3500.]]
    assert trace.trace_status(path)['verified'] is True
    assert trace.trace_status(path)['first_close_frame']==154
    recorder.restore();assert expert.ensure_gripper_force is original_force and expert.gripper_action is original_grip


def test_observation_errors_and_log_limit_do_not_replace_native_calls(tmp_path,expert):
    recorder=trace.DriveRecorder(tmp_path/'trace.jsonl',max_events=2)
    expert.test.view.get_max_efforts=lambda **k:(_ for _ in ()).throw(ValueError('read unavailable'))
    recorder.observe(expert,0)
    for _ in range(5):assert expert.ensure_gripper_force(70.) is expert.test.token
    assert len([c for c in expert.test.calls if c[0]=='native_force'])==5
    assert trace.trace_status(recorder.path)['verified'] is False
    assert any(json.loads(x)['phase']=='event_limit_reached' for x in recorder.path.read_text().splitlines())


def test_native_exception_identity_is_preserved(tmp_path,expert):
    error=RuntimeError('original force failure')
    def fail(*a,**k):raise error
    expert.ensure_gripper_force=fail
    recorder=trace.DriveRecorder(tmp_path/'trace.jsonl');recorder.observe(expert,0)
    with pytest.raises(RuntimeError) as caught:expert.ensure_gripper_force(70.)
    assert caught.value is error
    assert json.loads(recorder.path.read_text().splitlines()[-1])['phase']=='ensure_force_raised'


def test_nonfinite_native_gain_is_retained_and_flagged(expert):
    expert.test.gain[0,0]=np.nan
    row=trace.read_drive(expert)
    assert row['live_gains'][0][0][0]=='nan'
    assert 'live_gains' in row['errors']


def test_missing_snapshot_is_not_zero_gain(tmp_path):
    status=trace.trace_status(tmp_path/'not_created')
    assert status['verified'] is False and status['reason']=='missing_trace'
    assert trace.plain(np.array([float('nan')]))==['nan']


@pytest.mark.parametrize('ready_history,ready_fresh,observation',[(True,True,True),(False,True,True),(True,False,True),(True,True,False)])
def test_pair_analysis_requires_unchanged_motion_and_valid_observations(tmp_path,monkeypatch,ready_history,ready_fresh,observation):
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('gripper_pair_test',root/'scripts/run_hgwind_gripper_diagnostic.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    jobs={}
    for label in ['history','fresh']:
        run=tmp_path/label;case=run/'attempt_00060';case.mkdir(parents=True)
        (run/'report.json').write_text(json.dumps({'comparison_evaluable':True,'setup_force_control':{'verified':True},'cases':[dict(attempt_index=60,run=str(case),comparison_to_original={'all_core_arrays_exact':ready_history})]}))
        (case/'report.json').write_text(json.dumps({'probe_episode':str(case/'raw')}))
        (case/'gripper_drive_trace.jsonl').write_text('test trace placeholder')
        jobs[label]={'run':str(run),'exit_code':0}
    mod.REFERENCE=tmp_path/'fresh'
    monkeypatch.setattr(mod,'trace_status',lambda _:{'verified':observation})
    monkeypatch.setattr(mod,'compare_motion',lambda *a:{'all_core_arrays_exact':ready_fresh})
    monkeypatch.setattr(mod,'read_snapshots',lambda _:{'frame_0':{'snapshot':{'live_gains':[[[1.,0.]],[[2.,0.]]]}}})
    monkeypatch.setattr(mod.subprocess,'Popen',lambda *a,**k:pytest.fail('CPU analysis cannot launch workers'))
    expected=ready_history and ready_fresh and observation
    assert mod.finish(tmp_path,{'jobs':jobs})==(0 if expected else 2)
    report=json.loads((tmp_path/'report.json').read_text());assert report['comparison_ready'] is expected
    assert report['drive_properties_overridden'] is False


def test_pair_stops_before_fresh_if_history_no_longer_matches(tmp_path,monkeypatch):
    import sys
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('gripper_pair_stop_test',root/'scripts/run_hgwind_gripper_diagnostic.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    monkeypatch.setattr(mod,'ROOT',tmp_path)
    monkeypatch.setattr(sys,'argv',['diagnostic','--run'])
    monkeypatch.setattr(mod.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='CPU ready'))
    executions=[]
    def execute(cmd,log,on_run):
        executions.append(cmd)
        run=tmp_path/'data/derived/hgwind_history_diagnostics/child';run.mkdir(parents=True)
        on_run(str(run));(run/'report.json').write_text(json.dumps({'comparison_evaluable':False,'cases':[]}))
        return {'run':str(run),'exit_code':1}
    monkeypatch.setattr(mod,'execute',execute)
    observed=[]
    monkeypatch.setattr(mod,'finish',lambda out,state:(observed.append(state) or 2))
    assert mod.main()==2 and len(executions)==1
    assert '--fresh-control' not in executions[0]
    assert observed[0]['stop_reason']=='history_or_drive_observation_not_verified'
    assert list(observed[0]['jobs'])==['history']


def test_pair_keeps_report_when_worker_never_produces_output(tmp_path):
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('gripper_pair_missing_test',root/'scripts/run_hgwind_gripper_diagnostic.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    state={'jobs':{'history':{'run':None,'exit_code':1}},'stop_reason':'no_evaluable_job_history'}
    assert mod.finish(tmp_path,state)==2
    report=json.loads((tmp_path/'report.json').read_text())
    assert report['comparison_ready'] is False and report['errors']
    assert report['stop_reason']=='no_evaluable_job_history'
