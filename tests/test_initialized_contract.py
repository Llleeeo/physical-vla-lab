"""Explicit startup must execute at reset and packet boundaries, not hide in a plan."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import hashlib
import h5py
import numpy as np
import pytest
from capstone_vla import initialized_contract as init
from capstone_vla.ball_action_contract import SCHEMA as ACTION_SCHEMA


def plan_for(root):
    return dict(initialized_contract=init.SCHEMA,history_attempts=[60],attempt_offset=60,
        fresh_control=True,stage_load_barrier=True,trace_gripper_drives=True,
        independent_packets=True,action_contract=ACTION_SCHEMA,trace_cartesian_targets=False,
        packet_gain_profile='fixed-9-8',packet_posture_profile='latched',frames=319,seed=202610280,
        output=str(root),history_cases={'60':{'output':str(root)}},
        gripper_gain_control={'audit':str(root/'gripper_gain_control.json')},
        setup_force_control={'audit':str(root/'setup_force_control_audit.json')},
        initialization_reference_report_sha256='digest')


@pytest.mark.parametrize('key,value',[
    ('history_attempts',[58,59,60]),('attempt_offset',61),('frames',209),('seed',1),
    ('independent_packets',False),('trace_cartesian_targets',True),('initialized_contract','other'),
    ('action_contract','other'),('stage_load_barrier',False),('packet_gain_profile','recorded'),
    ('packet_posture_profile','recorded')])
def test_reject_unmatched_experiment(tmp_path,key,value):
    plan=plan_for(tmp_path);plan[key]=value
    with pytest.raises(ValueError,match='declared fresh60'):init.validate(plan)


def test_audits_cannot_silently_point_to_successful_parent(tmp_path):
    plan=plan_for(tmp_path);init.validate(plan)
    plan['gripper_gain_control']['audit']=str(tmp_path.parent/'old/audit.json')
    with pytest.raises(ValueError,match='prior run'):init.validate(plan)


@pytest.fixture
def runtime(tmp_path,monkeypatch):
    calls=[]
    class Recorder:
        def __init__(self,path):self.path=path;self.expert=None
        def emit(self,*args):calls.append(('emit',args))
        def observe(self,expert,index):
            calls.append(('observe',index))
            if self.expert is expert:return
            self.expert=expert;original=expert.ensure_gripper_force
            def force(*a,**k):
                calls.append(('before_native',));result=original(*a,**k)
                calls.append(('after_native',));return result
            expert.ensure_gripper_force=force
    class Gain:
        def __init__(self,cfg):self.expert=None
        def observe(self,expert,index):
            calls.append(('gain_observe',index))
            if self.expert is expert:return
            self.expert=expert;original=expert.ensure_gripper_force
            def force(*a,**k):
                result=original(*a,**k);calls.append(('gain_apply',));return result
            expert.ensure_gripper_force=force
    monkeypatch.setattr(init,'DriveRecorder',Recorder);monkeypatch.setattr(init,'GainControl',Gain)
    force=SimpleNamespace(_ACTIVE=SimpleNamespace(restore=lambda:calls.append(('retire_pre_attempt',))),
        StartupForceRecorder=lambda path:SimpleNamespace(path=path))
    token=object();expert=SimpleNamespace(ensure_gripper_force=lambda *a,**k:calls.append(('native',a,k)))
    policy=SimpleNamespace(expert=expert)
    def decision(p,w,i,dt):
        calls.append(('packet',i,w))
        if i==0:p.expert.ensure_gripper_force(70.,speed=.05,reconfigure=True)
        return token
    r=init.InitializedRuntime(plan_for(tmp_path),decision,force)
    monkeypatch.setattr(r,'before_reset_usd',lambda:['original_mimic'])
    return SimpleNamespace(runtime=r,policy=policy,calls=calls,force=force,token=token)


def test_original_packet_only_after_observation_and_gain_binding(runtime):
    r=runtime.runtime;r.begin(60);world=object()
    assert r.step(runtime.policy,world,0,1/60) is runtime.token
    assert r.step(runtime.policy,world,1,1/60) is runtime.token
    names=[c[0] for c in runtime.calls]
    assert names.index('observe')<names.index('gain_observe')<names.index('packet')
    assert names.index('before_native')<names.index('native')<names.index('after_native')<names.index('gain_apply')
    assert len([c for c in runtime.calls if c[0]=='native'])==1
    assert [c[2] for c in runtime.calls if c[0]=='packet']==[world,world]
    assert runtime.force._ACTIVE.path==Path(r.plan['output'])/'startup_force_calls.jsonl'
    assert r.state['frames_completed']==[0,1] and r.state['online_skill_decisions'] is False
    with pytest.raises(ValueError,match='one fresh'):r.begin(60)


def test_native_packet_exception_is_recorded_without_retry_or_replacement(runtime):
    r=runtime.runtime;r.begin(60);error=RuntimeError('original packet failure')
    def fail(*a):raise error
    r.decision=fail
    with pytest.raises(RuntimeError) as caught:r.step(runtime.policy,object(),0,1/60)
    assert caught.value is error and r.state['frames_completed']==[]
    assert r.state['frames_entered']==[0] and 'original packet failure' in r.state['error']
    with pytest.raises(ValueError,match='no retry'):r.step(runtime.policy,object(),0,1/60)


def test_clock_and_missing_reset_fail_before_motor_call(runtime):
    r=runtime.runtime
    with pytest.raises(ValueError):r.step(runtime.policy,None,0,1/60)
    assert not any(c[0]=='packet' for c in runtime.calls)


def motion(path,delta=0.):
    with h5py.File(path,'w') as f:
        f['evidence/native/inventory/rigid_body/prim_paths']=np.array(['/Root/Dynamics/ball_y9bssvn'],dtype=h5py.string_dtype())
        for kind,width in [('get_transforms',7),('get_velocities',6)]:
            data=np.zeros((2,1,width));data[1,0,0]=delta
            f['evidence/native/rigid_body/'+kind+'/values']=data


@pytest.mark.parametrize('bad', [None,'gain','force','initial_ball','interface','grasp','execution'])
def test_audit_separates_initialization_interface_and_grasp(tmp_path,monkeypatch,bad):
    plan=plan_for(tmp_path);actual=tmp_path/'actual';actual.mkdir();expected=tmp_path/'expected';expected.mkdir()
    motion(actual/'replay.hdf5',.004 if bad=='initial_ball' else 0.);motion(expected/'replay.hdf5')
    prior=tmp_path/'parent_report.json';prior.write_text(json.dumps({'probe_episode':str(expected)}))
    plan.update(initialization_reference_report=str(prior),initialization_reference_report_sha256=hashlib.sha256(prior.read_bytes()).hexdigest())
    runtime=dict(attempts=[60],frames_entered=list(range(319)),frames_completed=list(range(319)),error=None)
    if bad=='execution':runtime['error']='motor safety abort';runtime['frames_completed']=list(range(200));runtime['frames_entered']=list(range(201))
    (tmp_path/'initialized_contract_runtime.json').write_text(json.dumps(runtime))
    (tmp_path/'stage_load_barrier.json').write_text(json.dumps(dict(ready=True,held_script_instances=0,initial_timeline_time=0.,final_timeline_time=0.)))
    (tmp_path/'report.json').write_text(json.dumps(dict(probe_episode=str(actual),initial_state_match=True,
        completed_frames=319,worker_exit_code=0,runtime_safety_passed=True,
        ball_action_contract_passed=bad not in ['interface','grasp'],
        ball_action_contract_audit={k:bad!='interface' for k in [
            'decoded_actions_match_motor_boundary','fixed_collision_callbacks_always_enabled',
            'live_posture_verified','robot_context_matches_raw_recording']},
        packet_controller_audit={'no_online_skill_decisions_verified':True,'packet_execution_verified_with_normalization_roundoff':True},
        independent_grasp_evidence={'status':'fail' if bad=='grasp' else 'pass'})))
    monkeypatch.setattr(init,'audit_control',lambda p:{'verified':bad!='gain'})
    monkeypatch.setattr(init,'audit_intervention',lambda p:{'verified':bad!='force'})
    result=init.audit(plan)
    assert result['combined_passed'] is (bad is None)
    assert result['initialization_verified'] is (bad not in ['gain','force','initial_ball'])
    assert result['learned_policy_evaluated'] is False
    if bad=='interface':assert result['grasp']=='pass' and result['interface_passed'] is False
    if bad=='grasp':assert result['initialization_verified'] and result['interface_passed']


def test_no_episode_is_not_counted_as_a_failed_grasp(tmp_path):
    result=init.audit(plan_for(tmp_path))
    assert not result['combined_passed'] and result['grasp']=='not_evaluated' and result['errors']


def test_standard_command_rejects_inactive_inherited_hooks(tmp_path):
    spec=importlib.util.spec_from_file_location('contract_entry_test',Path(__file__).resolve().parents[1]/'scripts/run_hgwind_action_contract.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    (tmp_path/'meta.json').write_text(json.dumps({'variation':1.}))
    plan={'source':str(tmp_path/'replay.hdf5'),'setup_force_control':{'audit':'old'}}
    with pytest.raises(ValueError,match='explicit initialized'):module.command(plan,tmp_path)
    plan.update(plan_for(tmp_path));plan.update(worker_output=str(tmp_path/'out'),task='task')
    cmd=module.command(plan,tmp_path)
    assert cmd[2].endswith('run_hgwind_initialized_contract_worker.py')
    assert cmd[cmd.index('--episodes')+1]=='1' and cmd[cmd.index('--max_attempts')+1]=='1'
    assert '--replay_trajectories' not in cmd
