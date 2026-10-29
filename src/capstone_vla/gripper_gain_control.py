"""Case-specific reapplication of measured warm gripper gains after USD setup.

Diagnostic only: the reproduced warm configuration drives both finger DOFs
while the original mimic constraint remains. This is not a production fix.
"""
import json
from pathlib import Path
import numpy as np
from .hf_archive import sha256
from .gripper_drive_trace import FINGERS, FRAMES, plain, read_drive

SUMMARY = 'results/summaries/2026-10-30-gripper-drive-result.json'
ARGS = {'args': [70.0], 'kwargs': {'speed': 0.05, 'reconfigure': True}}


def prepare_control(plan, root):
    root = Path(root)
    if (plan.get('history_attempts') != [60] or not plan.get('fresh_control')
            or not plan.get('stage_load_barrier') or not plan.get('trace_gripper_drives')
            or not plan.get('setup_force_control')):
        raise ValueError('Gain control requires observed fresh60 with the verified force omission')
    summary = json.loads((root/SUMMARY).read_text())
    if not summary['comparison_ready'] or not summary['independently_rechecked_full_motion_arrays']:
        raise ValueError('Measured drive pair is not verified')
    for name, digest in summary['source_hashes'].items():
        if sha256(root/name) != digest:
            raise ValueError('Measured drive evidence changed: ' + name)
    cfg = dict(schema='physical-vla-lab.gripper-gain-control.v1',
        summary_sha256=sha256(root/SUMMARY), audit=str(Path(plan['output'])/'gripper_gain_control.json'),
        arguments=ARGS, expected_before_gains=[[[10000., 0.]], [[1000., 0.]]],
        requested_gains=[[[10500.,10500.]],[[140.,140.]]],
        expected_efforts=[[70.,70.]], expected_speeds=[[0.05000000074505806]*2],
        training_eligible=False, scope='One native finger-gain reapplication after original frame0 ensure; no cap/target/USD/arm setter or added physics step. Mimic coupling remains.')
    return cfg


class GainControl:
    def __init__(self, config):
        self.config = config
        self.path = Path(config['audit'])
        if self.path.exists():raise ValueError('Preserve existing gain-control audit')
        self.state = dict(schema=config['schema'], summary_sha256=config['summary_sha256'],
            applied=0, frame=None, checks=[], error=None, training_eligible=False,
            mimic_coupling_retained=True, production_fix_verified=False)
        self.expert = None
        self.persist()

    def persist(self):
        temp = self.path.with_suffix('.tmp')
        temp.write_text(json.dumps(plain(self.state),indent=2,allow_nan=False)+'\n')
        temp.replace(self.path)

    def fail(self, message):
        self.state['error'] = message;self.persist();raise ValueError(message)

    def snapshot(self):
        snapshot = read_drive(self.expert)
        if snapshot.get('errors') or snapshot.get('live_physics_handle_valid') is not True:
            self.fail('Invalid live drive observation; no USD fallback')
        return snapshot

    def bind(self, expert):
        if self.expert is expert:return
        if self.expert is not None:self.fail('Gain experiment cannot mix experts')
        self.expert = expert
        original = expert.ensure_gripper_force
        def force(*args, **kwargs):
            if self.state['applied'] or self.state['frame'] != 0:
                self.fail('Only one original frame0 force configuration is permitted')
            actual = plain({'args':args,'kwargs':kwargs})
            if actual != self.config['arguments']:
                self.fail('Original ensure arguments differ from measured pair')
            try:result = original(*args,**kwargs)
            except Exception as exc:
                self.state['error'] = repr(exc);self.persist();raise
            self.state['original_returned'] = plain(result)
            before = self.snapshot();self.state['before'] = before
            cfg = self.config
            if (result is not True or before['live_gains'] != cfg['expected_before_gains']
                    or before['live_max_efforts'] != cfg['expected_efforts']
                    or before['live_max_velocities'] != cfg['expected_speeds']):
                self.fail('Original initialization does not match measured fresh condition')
            bookkeeping = before['expert_state'].get('_last_gripper_drive') or {}
            if (bookkeeping.get('verified') is not True or bookkeeping.get('stiffness') != 10500.
                    or bookkeeping.get('damping') != 140.):
                self.fail('Native requested gains differ from the measured warm values')
            robot = expert._robot;view = robot._articulation_view
            if before['joint_names'] != list(FINGERS) or before['joint_indices'] != [7,8]:
                self.fail('Unexpected Panda finger identity')
            all_before = [np.array(x,copy=True) for x in view.get_gains()]
            if any(x.shape != (1,9) or not np.isfinite(x).all() for x in all_before):
                self.fail('Invalid full-articulation gain readback')
            self.state['setter_started'] = True;self.persist()
            try:
                view.set_gains(kps=np.array(cfg['requested_gains'][0],dtype=np.float32),
                    kds=np.array(cfg['requested_gains'][1],dtype=np.float32),
                    joint_indices=np.array([7,8],dtype=np.int32),save_to_usd=False)
                after = self.snapshot();self.state['after'] = after
                all_after = [np.array(x,copy=True) for x in view.get_gains()]
                if not all(np.array_equal(a[:,:7],b[:,:7]) for a,b in zip(all_before,all_after)):
                    self.fail('Arm gains changed during targeted finger setter')
                for key in ['live_max_efforts','live_max_velocities','last_applied_action',
                            'usd_fingers','measured_positions_m','measured_velocities_m_s']:
                    if before[key] != after[key]:self.fail('Gain setter changed another property: '+key)
                if after['live_gains'] != cfg['requested_gains']:
                    self.fail('Requested gains did not reach native readback')
            except Exception as exc:
                if self.state['error'] is None:self.state['error']=repr(exc);self.persist()
                raise
            self.state.update(applied=1,applied_frame=0,arm_gains_preserved=True,
                caps_targets_usd_state_preserved=True)
            self.persist()
            print('GRIPPER_GAIN_CONTROL_APPLIED frame=0 native_readback=verified mimic=retained',flush=True)
            return result
        expert.ensure_gripper_force = force

    def observe(self, expert, frame):
        self.state['frame'] = int(frame);self.bind(expert)
        if int(frame) == 0:return
        if self.state['applied'] != 1:self.fail('Required initialization gain reapplication was not encountered')
        if int(frame) not in FRAMES | {1}:return
        s = self.snapshot()
        if (s['live_gains'] != self.config['requested_gains']
                or s['live_max_efforts'] != self.config['expected_efforts']
                or s['live_max_velocities'] != self.config['expected_speeds']):
            self.fail('Gains or existing caps changed after reapplication')
        self.state['checks'].append(dict(frame=int(frame),live_gains=s['live_gains'],
            live_max_efforts=s['live_max_efforts'],live_max_velocities=s['live_max_velocities']))
        self.persist()


def install(plan):
    from . import hgwind_expert_trace as trace
    if (plan.get('history_attempts') != [60] or not plan.get('fresh_control')
            or not plan.get('setup_force_control') or not plan.get('trace_gripper_drives')):
        raise ValueError('Gain control is restricted to the declared fresh diagnostic')
    gate = GainControl(plan['gripper_gain_control'])
    original = trace.begin_cartesian_frame
    def frame(expert,index):
        original(expert,index)  # Passive drive observer binds before this wrapper.
        gate.observe(expert,index)
    trace.begin_cartesian_frame = frame
    return gate


def audit_control(plan):
    cfg = plan['gripper_gain_control'];path = Path(cfg['audit'])
    if not path.exists():return dict(verified=False,reason='missing_audit')
    s = json.loads(path.read_text())
    checked = {row['frame'] for row in s['checks']}
    selected = {150,153,154,155,156,160,166,167,170,173}
    # Include actual subsequent passive samples, not just the immediate setter.
    trace_path = Path(plan['history_cases']['60']['output'])/'gripper_drive_trace.jsonl'
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()] if trace_path.exists() else []
    samples = [r['snapshot'] for r in rows if r['phase']=='frame_before_policy' and r['frame'] in selected]
    close = [r for r in rows if r['phase']=='close_action_built']
    verified = bool(s['summary_sha256']==cfg['summary_sha256'] and s['error'] is None
        and s['applied']==1 and s.get('applied_frame')==0
        and s.get('arm_gains_preserved') and s.get('caps_targets_usd_state_preserved')
        and rows and any(r['phase']=='frame_before_policy' and r['frame']==0 for r in rows)
        and 1 in checked and all(r['frame'] in checked for r in rows
            if r['phase']=='frame_before_policy' and r['frame'] in selected)
        and not any(r['phase'] in ['event_limit_reached','ensure_force_raised'] for r in rows)
        and all(r.get('live_gains')==cfg['requested_gains'] for r in samples)
        and all(r['snapshot'].get('live_gains')==cfg['requested_gains'] for r in close))
    return dict(verified=verified,applied=s['applied'],checked_frames=sorted(checked),
        error=s['error'],reason='native_reapplication_measured' if verified else 'incomplete_or_invalid_control_evidence',
        close_observed=bool(close),first_close_frame=close[0]['frame'] if close else None,
        mimic_coupling_retained=True,training_eligible=False,production_fix_verified=False)
