"""Single force intervention must never silently suppress a different call."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from capstone_vla import hgwind_startup_trace as trace
from capstone_vla.setup_force_control import install, audit_intervention, signature, prepare_intervention, OMITTED


@pytest.fixture
def control(tmp_path, monkeypatch):
    target = tmp_path/'case/startup_force_calls.jsonl'
    target.parent.mkdir()
    recorder = trace.StartupForceRecorder(target)
    recorder.sim = SimpleNamespace(current_time=.03, get_physics_dt=lambda:1/60)
    args = ('/Root/Dynamics/ball_y9bssvn', [0.67, 0., 0.], [1., 2., 3.], 'Force')
    expected = signature(recorder, 'fan_script.py', args, {})
    cfg = dict(schema='physical-vla-lab.setup-force-control.v1', audit=str(tmp_path/'audit.json'),
               expected_signature=expected, baseline_report_sha256='fixed')
    plan = dict(history_attempts=[60],fresh_control=True,stage_load_barrier=True,setup_force_control=cfg,
                history_cases={'60':{'output':str(target.parent)}})
    # Restores patches on exit; no simulator module is needed.
    monkeypatch.setattr(trace,'ForceProxy',trace.ForceProxy)
    monkeypatch.setattr(trace,'startup_frame',trace.startup_frame)
    monkeypatch.setattr(trace,'_ACTIVE',recorder)
    gate = install(plan)
    calls=[];token=object()
    native=SimpleNamespace(apply_force_at_pos=lambda *a,**k:(calls.append((a,k)) or token))
    proxy=trace.ForceProxy(native,recorder,'fan_script.py')
    return SimpleNamespace(plan=plan,gate=gate,recorder=recorder,args=args,
        calls=calls,token=token,native=native,proxy=proxy)


def test_omit_only_pinned_second_call_and_forward_all_control_forces(control):
    c=control
    assert c.proxy.apply_force_at_pos(*c.args) is c.token
    assert c.proxy.apply_force_at_pos(*c.args) is None
    assert len(c.calls)==1
    assert c.calls[0][0][1] is c.args[1]
    assert c.proxy.apply_force_at_pos(*c.args) is c.token
    trace.startup_frame(0)
    for frame in (0,1,7,90):
        c.recorder.frame=frame
        assert c.proxy.apply_force_at_pos(*c.args) is c.token
    assert len(c.calls)==6
    rows=[json.loads(x) for x in c.recorder.path.read_text().splitlines()]
    assert [x['event'] for x in rows if x['status']==OMITTED]==[1]
    assert audit_intervention(c.plan)['verified'] is True
    assert c.gate.state['training_eligible'] is False


@pytest.mark.parametrize('difference',['force','clock','body','signature'])
def test_candidate_mismatch_aborts_instead_of_omitting_a_substitute(control,difference):
    c=control;c.proxy.apply_force_at_pos(*c.args)
    args=list(c.args);kwargs={}
    if difference=='force':args[1]=[.68,0.,0.]
    elif difference=='clock':c.recorder.sim.current_time=.04
    elif difference=='body':args[0]='/other'
    else:kwargs={'unexpected':True}
    with pytest.raises(ValueError):c.proxy.apply_force_at_pos(*args,**kwargs)
    assert len(c.calls)==1 and c.gate.state['omitted_calls']==0
    assert audit_intervention(c.plan)['verified'] is False


def test_missing_repeat_before_frame_zero_is_not_treated_as_valid(control):
    c=control;c.proxy.apply_force_at_pos(*c.args)
    with pytest.raises(ValueError,match='did not occur'):trace.startup_frame(0)
    assert c.recorder.frame is None
    assert audit_intervention(c.plan)['verified'] is False


def test_other_recorder_and_native_exceptions_are_not_suppressed(control,tmp_path):
    c=control;other=trace.StartupForceRecorder(tmp_path/'unrelated.jsonl')
    proxy=trace.ForceProxy(c.native,other,'other_script.py')
    assert proxy.apply_force_at_pos(body='anything') is c.token
    assert c.gate.state['omitted_calls']==0
    c.proxy.apply_force_at_pos(*c.args);c.proxy.apply_force_at_pos(*c.args)
    trace.startup_frame(0)
    error=RuntimeError('native error')
    def fail(*a,**k):raise error
    c.native.apply_force_at_pos=fail
    with pytest.raises(RuntimeError) as caught:c.proxy.apply_force_at_pos(*c.args)
    assert caught.value is error
    assert json.loads(c.recorder.path.read_text().splitlines()[-1])['status']=='raised'


def test_audit_rejects_missing_or_changed_omission_record(control):
    c=control;c.proxy.apply_force_at_pos(*c.args);c.proxy.apply_force_at_pos(*c.args)
    trace.startup_frame(0)
    rows=[json.loads(x) for x in c.recorder.path.read_text().splitlines()]
    rows[1]['force'][0]=999
    c.recorder.path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    assert audit_intervention(c.plan)['verified'] is False
    Path(c.plan['setup_force_control']['audit']).unlink()
    assert audit_intervention(c.plan)['reason']=='audit_missing'


def test_no_history_or_default_flow_can_enable_intervention(tmp_path):
    with pytest.raises(ValueError,match='matched-startup'):
        prepare_intervention({'fresh_control':False},tmp_path)
    with pytest.raises(ValueError,match='one fresh60'):
        install({'history_attempts':[58,59,60],'fresh_control':False})
