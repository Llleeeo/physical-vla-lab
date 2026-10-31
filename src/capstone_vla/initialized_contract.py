"""Carry measured fresh60 initialization into one frozen 29D surrogate run.

The existing history reset hook is reused for one attempt boundary only.
No online expert task/skill decision or prior episode is executed here.
"""
import json
from pathlib import Path
import h5py
import numpy as np
from .hf_archive import sha256
from .ball_action_contract import SCHEMA as ACTION_SCHEMA
from .gripper_drive_trace import DriveRecorder, usd_fingers
from .gripper_gain_control import GainControl, audit_control
from .setup_force_control import audit_intervention

SCHEMA = 'physical-vla-lab.initialized-ball-contract.v1'


def validate(plan):
    if (plan.get('initialized_contract') != SCHEMA or plan.get('history_attempts') != [60]
            or plan.get('attempt_offset') != 60 or plan.get('fresh_control') is not True
            or plan.get('stage_load_barrier') is not True
            or plan.get('trace_gripper_drives') is not True
            or plan.get('independent_packets') is not True
            or plan.get('action_contract') != ACTION_SCHEMA
            or plan.get('trace_cartesian_targets') is not False
            or plan.get('packet_gain_profile') != 'fixed-9-8'
            or plan.get('packet_posture_profile') != 'latched'
            or plan.get('frames') != 319 or plan.get('seed') != 202610280):
        raise ValueError('Initialized contract requires one declared fresh60 319-frame surrogate')
    root = Path(plan['output']).resolve()
    if Path(plan['history_cases']['60']['output']).resolve() != root:
        raise ValueError('Initialization evidence must belong to the current output')
    for key, filename in [('gripper_gain_control','gripper_gain_control.json'),
                          ('setup_force_control','setup_force_control_audit.json')]:
        if Path(plan[key]['audit']).resolve() != root/filename:
            raise ValueError('Initialization audit still points to a prior run')
    if not plan.get('initialization_reference_report_sha256'):
        raise ValueError('Missing verified initialization reference')


def patch_collector(source):
    from .hgwind_history import patch_history_collector
    return patch_history_collector(source, stage_load_barrier=True)


class InitializedRuntime:
    def __init__(self, plan, decision, force_module):
        validate(plan)
        self.plan = plan;self.decision = decision;self.force = force_module
        self.root = Path(plan['output'])
        self.path = self.root/'initialized_contract_runtime.json'
        if self.path.exists():raise ValueError('Preserve existing initialization runtime audit')
        self.state = dict(schema=SCHEMA,attempts=[],frames_entered=[],frames_completed=[],
            error=None,online_skill_decisions=False,prior_attempts=0)
        self.recorder = None
        gain_class = GainControl
        if plan.get('mimic_drive_control'):
            from .mimic_drive_control import SingleDriveGainControl
            gain_class = SingleDriveGainControl
        self.gain = gain_class(plan['gripper_gain_control'])
        self.persist()

    def persist(self):
        temp=self.path.with_suffix('.tmp')
        temp.write_text(json.dumps(self.state,indent=2)+'\n');temp.replace(self.path)

    def fail(self, message):
        self.state['error']=message;self.persist();raise ValueError(message)

    def before_reset_usd(self):
        import omni.usd
        return usd_fingers(omni.usd.get_context().get_stage(),'/Root/Dynamics/franka')

    def begin(self, index):
        if int(index)!=60 or self.state['attempts']:
            self.fail('Only one fresh attempt60 is permitted; no warmup episodes')
        if self.force._ACTIVE is not None:self.force._ACTIVE.restore()
        self.force._ACTIVE=self.force.StartupForceRecorder(self.root/'startup_force_calls.jsonl')
        self.recorder=DriveRecorder(self.root/'gripper_drive_trace.jsonl')
        try:self.recorder.emit('usd_before_reset',{'usd_fingers':self.before_reset_usd()})
        except Exception as exc:self.recorder.emit('usd_before_reset',{'errors':{'usd':repr(exc)}})
        self.state['attempts'].append(60);self.persist()
        print('INITIALIZED_CONTRACT_BEGIN attempt=60 prior_attempts=0',flush=True)

    def step(self, policy, world, index, dt):
        if (self.recorder is None or index!=len(self.state['frames_completed'])
                or index!=len(self.state['frames_entered']) or not 0<=index<319
                or abs(float(dt)-1/60)>1.e-12):
            self.fail('Initialized packet clock/order differs; no retry or skipped frame')
        self.state['frames_entered'].append(int(index));self.persist()
        try:
            # The observer wraps the original method first. Gain control then
            # re-applies only after the original configuration/observation return.
            self.recorder.observe(policy.expert,index)
            self.gain.observe(policy.expert,index)
            result=self.decision(policy,world,index,dt)
        except Exception as exc:
            self.state['error']=repr(exc);self.persist();raise
        self.state['frames_completed'].append(int(index));self.persist()
        return result


def install(plan):
    from . import hgwind_expert_trace as trace, hgwind_history as history
    from . import hgwind_startup_trace as force
    from .setup_force_control import install as install_force
    validate(plan)
    if plan.get('precision_control'):
        from .action_precision_control import verify_runtime
        verify_runtime(plan)
    if plan.get('mimic_drive_control'):
        from .mimic_drive_control import verify_runtime as verify_mimic_runtime
        verify_mimic_runtime(plan)
    runtime=InitializedRuntime(plan,trace.packet_decision,force)
    force.configure(Path(plan['output'])/'pre_attempt_force_calls.jsonl')
    install_force(plan)
    history.begin_attempt=runtime.begin
    trace.packet_decision=runtime.step
    original=trace.patch_collector
    trace.patch_collector=lambda source:patch_collector(original(source))
    return runtime


def audit(plan):
    """Keep initialization evidence, physical outcome and execution separate."""
    validate(plan)
    root=Path(plan['output'])
    result=dict(schema=SCHEMA,initialization_verified=False,execution_complete=False,
        interface_passed=None,grasp='not_evaluated',combined_passed=False,
        learned_policy_evaluated=False,training_eligible=False,errors=[])
    try:
        runtime=json.loads((root/'initialized_contract_runtime.json').read_text())
        barrier=json.loads((root/'stage_load_barrier.json').read_text())
        result['gain']=audit_control(plan);result['force']=audit_intervention(plan)
        report=json.loads((root/'report.json').read_text())
        source=json.loads(Path(plan['initialization_reference_report']).read_text())
        if sha256(plan['initialization_reference_report'])!=plan['initialization_reference_report_sha256']:
            raise ValueError('Initialization reference report changed')
        target='/Root/Dynamics/ball_y9bssvn';prefix='evidence/native/'
        with h5py.File(Path(report['probe_episode'])/'replay.hdf5','r') as actual, h5py.File(Path(source['probe_episode'])/'replay.hdf5','r') as expected:
            names=prefix+'inventory/rigid_body/prim_paths'
            a_names=actual[names].asstr()[:].tolist();e_names=expected[names].asstr()[:].tolist()
            if a_names.count(target)!=1 or e_names.count(target)!=1:raise ValueError('Ambiguous initialization ball identity')
            initial={}
            for key in ['get_transforms','get_velocities']:
                name=prefix+'rigid_body/'+key+'/values'
                a=actual[name][:2,a_names.index(target)];e=expected[name][:2,e_names.index(target)]
                initial[key]=bool(len(a)==2 and a.shape==e.shape and np.array_equal(a,e,equal_nan=False))
            result['first_two_native_ball_rows_exact']=initial
        entered=runtime['frames_entered'];completed=runtime['frames_completed']
        order_ok=(runtime['attempts']==[60] and entered==list(range(len(entered)))
            and completed==list(range(len(completed))) and len(entered) in (len(completed),len(completed)+1))
        barrier_ok=(barrier.get('ready') is True and barrier.get('held_script_instances')==0
            and barrier.get('initial_timeline_time')==barrier.get('final_timeline_time'))
        result['runtime_error']=runtime['error'];result['packet_order_verified']=order_ok
        result['initialization_verified']=bool(order_ok and barrier_ok
            and result['gain']['verified'] and result['force']['verified']
            and report['initial_state_match'] and all(initial.values()))
        result['execution_complete']=bool(runtime['error'] is None and completed==list(range(319))
            and report['completed_frames']==319 and report['runtime_safety_passed']
            and report['worker_exit_code']==0)
        contract=report.get('ball_action_contract_audit',{})
        packet=report.get('packet_controller_audit',{})
        result['interface_passed']=bool(all(contract.get(k) is True for k in [
            'decoded_actions_match_motor_boundary','fixed_collision_callbacks_always_enabled',
            'live_posture_verified','robot_context_matches_raw_recording'])
            and packet.get('no_online_skill_decisions_verified') is True
            and packet.get('packet_execution_verified_with_normalization_roundoff',
                           packet.get('packet_execution_verified')) is True)
        result['legacy_combined_contract_gate']=report.get('ball_action_contract_passed')
        result['grasp']=report['independent_grasp_evidence']['status']
        result['combined_passed']=bool(result['initialization_verified'] and result['execution_complete']
            and result['interface_passed'] and result['grasp']=='pass')
    except (OSError,ValueError,TypeError,KeyError) as exc:result['errors'].append(repr(exc))
    (root/'initialization_report.json').write_text(json.dumps(result,indent=2)+'\n')
    print('INITIALIZED_CONTRACT initialization_verified='+str(result['initialization_verified']).lower()
        +' interface='+str(result['interface_passed']).lower()+' grasp='+result['grasp']
        +' combined_passed='+str(result['combined_passed']).lower(),flush=True)
    return result
