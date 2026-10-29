"""Post-USD gain reapplication must stay local, explicit and measurable."""
from copy import deepcopy
import json
from types import SimpleNamespace
import numpy as np
import pytest
from capstone_vla import gripper_gain_control as control


@pytest.fixture
def rig(tmp_path,monkeypatch):
    cfg=dict(schema='test',audit=str(tmp_path/'audit.json'),summary_sha256='digest',
        arguments=deepcopy(control.ARGS),expected_before_gains=[[[10000.,0.]],[[1000.,0.]]],
        requested_gains=[[[10500.,10500.]],[[140.,140.]]],
        expected_efforts=[[70.,70.]],expected_speeds=[[0.05000000074505806]*2])
    kp=np.ones((1,9));kd=np.ones((1,9))*2
    kp[0,7:]=[10000.,0.];kd[0,7:]=[1000.,0.]
    calls=[];knobs={};error=RuntimeError('original error')
    def original(*args,**kwargs):
        calls.append(('original',args,kwargs))
        if knobs.get('native_error'):raise error
        return not knobs.get('native_false',False)
    def snapshot(e):
        return deepcopy(dict(errors={'live':'missing'} if knobs.get('read_error') else {},
            live_physics_handle_valid=not knobs.get('no_handle',False),
            live_gains=[kp[:,7:].tolist(),kd[:,7:].tolist()],
            live_max_efforts=[[70.,70.]],live_max_velocities=cfg['expected_speeds'],
            joint_names=list(control.FINGERS),joint_indices=[7,8],
            expert_state={'_last_gripper_drive':{'verified':True,'stiffness':10500.,'damping':140.}},
            last_applied_action={'joint_positions':[.04,.04]},usd_fingers=['original_mimic'],
            measured_positions_m=[.04,.04],measured_velocities_m_s=[0.,0.]))
    def setter(**kwargs):
        calls.append(('setter',deepcopy(kwargs)))
        assert kwargs['save_to_usd'] is False
        assert kwargs['joint_indices'].tolist()==[7,8]
        if knobs.get('setter_error'):raise ValueError('native setter error')
        if not knobs.get('ignore_setter'):
            kp[:,7:]=kwargs['kps'];kd[:,7:]=kwargs['kds']
        if knobs.get('arm_mutation'):kp[0,0]=999
    view=SimpleNamespace(get_gains=lambda:(kp.copy(),kd.copy()),set_gains=setter)
    expert=SimpleNamespace(ensure_gripper_force=original,_robot=SimpleNamespace(_articulation_view=view))
    monkeypatch.setattr(control,'read_drive',snapshot)
    return SimpleNamespace(cfg=cfg,kp=kp,kd=kd,calls=calls,expert=expert,knobs=knobs,error=error)


def apply(rig):
    gate=control.GainControl(rig.cfg);gate.observe(rig.expert,0)
    assert rig.expert.ensure_gripper_force(70.,speed=.05,reconfigure=True) is True
    return gate


def test_post_native_targeted_setter_with_preserved_arm_and_passive_checks(rig):
    arms=[rig.kp[:,:7].copy(),rig.kd[:,:7].copy()]
    gate=apply(rig)
    assert [x[0] for x in rig.calls]==['original','setter']
    assert rig.calls[0][1:]==((70.,),{'speed':.05,'reconfigure':True})
    assert np.array_equal(rig.kp[:,:7],arms[0]) and np.array_equal(rig.kd[:,:7],arms[1])
    gate.observe(rig.expert,1);gate.observe(rig.expert,154)
    assert len([x for x in rig.calls if x[0]=='setter'])==1
    assert gate.state['applied']==1 and [x['frame'] for x in gate.state['checks']]==[1,154]
    assert gate.state['production_fix_verified'] is False
    with pytest.raises(ValueError,match='Only one'):rig.expert.ensure_gripper_force(70.,speed=.05,reconfigure=True)
    assert len([x for x in rig.calls if x[0]=='original'])==1


@pytest.mark.parametrize('knob,match',[
    ('native_false','initialization'),('read_error','Invalid live'),('no_handle','Invalid live'),
    ('ignore_setter','did not reach'),('arm_mutation','Arm gains'),('setter_error','native setter error')])
def test_invalid_native_state_or_setter_cannot_report_success(rig,knob,match):
    rig.knobs[knob]=True
    with pytest.raises(ValueError,match=match):apply(rig)
    audit=json.loads(__import__('pathlib').Path(rig.cfg['audit']).read_text())
    assert audit['applied']==0 and audit['error']


def test_original_exception_preserved_without_setter(rig):
    rig.knobs['native_error']=True
    with pytest.raises(RuntimeError) as caught:apply(rig)
    assert caught.value is rig.error
    assert not any(x[0]=='setter' for x in rig.calls)


def test_native_gain_overwrite_after_frame0_is_detected(rig):
    gate=apply(rig);rig.kp[0,7]=10000
    with pytest.raises(ValueError,match='changed after'):gate.observe(rig.expert,1)


def test_alternate_force_arguments_rejected_before_native_call(rig):
    gate=control.GainControl(rig.cfg);gate.observe(rig.expert,0)
    with pytest.raises(ValueError,match='arguments differ'):rig.expert.ensure_gripper_force(90.,speed=.05,reconfigure=True)
    assert rig.calls==[]


def test_install_orders_passive_observer_before_control_and_does_not_step(rig,monkeypatch):
    from capstone_vla import hgwind_expert_trace as trace
    observed=[]
    monkeypatch.setattr(trace,'begin_cartesian_frame',lambda e,i:observed.append(i))
    plan=dict(history_attempts=[60],fresh_control=True,setup_force_control={},trace_gripper_drives=True,gripper_gain_control=rig.cfg)
    plan['setup_force_control']={'verified_source':True}
    gate=control.install(plan);trace.begin_cartesian_frame(rig.expert,0)
    assert observed==[0] and gate.expert is rig.expert and rig.calls==[]
    assert rig.expert.ensure_gripper_force(70.,speed=.05,reconfigure=True) is True


def test_control_requires_source_evidence_and_exact_case(tmp_path):
    with pytest.raises(ValueError,match='requires observed'):control.prepare_control({},tmp_path)
    with pytest.raises(ValueError,match='restricted'):control.install({})


def test_missing_audit_is_not_verified(tmp_path):
    assert control.audit_control({'gripper_gain_control':{'audit':str(tmp_path/'missing')}})['verified'] is False


def test_audit_reports_measured_intervention_separately_from_grasp_or_close_time(rig,tmp_path):
    gate=apply(rig);gate.observe(rig.expert,1)
    case=tmp_path/'case';case.mkdir()
    rows=[{'phase':'frame_before_policy','frame':0,'snapshot':{}},
          {'phase':'close_action_built','frame':155,'snapshot':{'live_gains':rig.cfg['requested_gains']}}]
    trace=case/'gripper_drive_trace.jsonl';trace.write_text('\n'.join(json.dumps(x) for x in rows))
    plan={'gripper_gain_control':rig.cfg,'history_cases':{'60':{'output':str(case)}}}
    result=control.audit_control(plan)
    assert result['verified'] and result['first_close_frame']==155 and result['production_fix_verified'] is False
    trace.unlink()
    assert control.audit_control(plan)['verified'] is False
    gate.state['error']='later overwrite';gate.persist()
    assert control.audit_control(plan)['verified'] is False
