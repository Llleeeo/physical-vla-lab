"""One explicit force-call omission for a development startup experiment.

This deliberately changes applied-force history. It is not a production fan
fix, general duplicate detector, training demonstration or VLA controller.
"""
import json
import math
from pathlib import Path

from .hf_archive import sha256

FIELDS = ('script', 'body', 'force', 'position', 'mode',
          'physics_dt_s', 'simulation_time_s')
OMITTED = 'omitted_by_setup_force_control'
BASELINE = '20261030_003531_182273_fresh60'


def signature(recorder, script, args, kwargs):
    if len(args) != 4 or kwargs or recorder.sim is None:
        raise ValueError('Setup force candidate has an unsupported signature/context')
    row = dict(script=str(script), body=str(args[0]),
        force=[float(args[1][i]) for i in range(3)],
        position=[float(args[2][i]) for i in range(3)], mode=str(args[3]),
        physics_dt_s=float(recorder.sim.get_physics_dt()),
        simulation_time_s=float(recorder.sim.current_time))
    if not all(math.isfinite(x) for x in row['force'] + row['position']
               + [row['physics_dt_s'], row['simulation_time_s']]):
        raise ValueError('Nonfinite setup force candidate')
    return row


def prepare_intervention(plan, root):
    """Pin the observed unmodified control, never derive a successful outcome."""
    root = Path(root)
    if not plan.get('fresh_control') or not plan.get('stage_load_barrier'):
        raise ValueError('Omission requires matched-startup fresh60')
    baseline = root/'data/derived/hgwind_history_diagnostics'/BASELINE
    summary = json.loads((root/'results/summaries/2026-10-30-matched-fresh60.json').read_text())
    for name, digest in summary['source_hashes'].items():
        if sha256(baseline/name) != digest:
            raise ValueError('Pinned fresh-control evidence changed: ' + name)
    report = json.loads((baseline/'report.json').read_text())
    prior = json.loads((baseline/'plan.json').read_text())
    if not (report['comparison_evaluable'] and report['stage_load_barrier_verified']
            and report['cases'][0]['comparison_to_fresh']['all_core_arrays_exact']):
        raise ValueError('Unmodified fresh60 control is not verified')
    for key in ['patch_sha256', 'task_sha256', 'physics_sha256', 'worker_sha256',
                'controller_config_sha256', 'original_depth_bias_m', 'seed']:
        if plan[key] != prior[key]:
            raise ValueError('Intervention changes another dependency: ' + key)
    trace = baseline/'attempt_00060/startup_force_calls.jsonl'
    if sha256(trace) != report['cases'][0]['force_trace']['sha256']:
        raise ValueError('Pinned baseline force trace changed')
    calls = [json.loads(line) for line in trace.read_text().splitlines()]
    if len(calls) < 2 or any(x['frame'] is not None or x['status'] != 'returned'
                              or x['event'] != i for i, x in enumerate(calls[:2])):
        raise ValueError('Expected two returned pre-recording force calls')
    first, second = [{key: row[key] for key in FIELDS} for row in calls[:2]]
    if first != second or second['body'] != '/Root/Dynamics/ball_y9bssvn':
        raise ValueError('Observed first pair is not an identical target-ball call')
    return dict(schema='physical-vla-lab.setup-force-control.v1',
        baseline=str(baseline), baseline_report_sha256=sha256(baseline/'report.json'),
        baseline_force_trace_sha256=sha256(trace), candidate_event=1,
        expected_signature=second,
        audit=str(Path(plan['output'])/'setup_force_control_audit.json'),
        scope='Omit only second observed pre-recording ball-force submission; all other calls forward unchanged',
        modified_forcing_history=True, training_eligible=False)


class OmitOneSetupForce:
    def __init__(self, config, target_trace):
        self.config = config
        self.target = Path(target_trace)
        self.path = Path(config['audit'])
        if self.path.exists():
            raise ValueError('Setup-force intervention audit already exists')
        self.state = dict(schema=config['schema'], baseline_report_sha256=config['baseline_report_sha256'],
            candidate_event=1, expected_signature=config['expected_signature'],
            first_call_returned=False, omitted_calls=0, recording_started=False,
            error=None, modified_forcing_history=True, training_eligible=False)
        self.persist()

    def persist(self):
        temp = self.path.with_suffix('.tmp')
        temp.write_text(json.dumps(self.state, indent=2, allow_nan=False)+'\n')
        temp.replace(self.path)

    def fail(self, message):
        self.state['error'] = message
        self.persist()
        raise ValueError(message)

    def is_target(self, recorder):
        return recorder.path == self.target

    def omit(self, recorder, script, args, kwargs):
        if not self.is_target(recorder) or self.state['omitted_calls'] == 1:
            return False
        if recorder.frame is not None or recorder.events not in (0, 1):
            self.fail('Candidate not encountered before recording; no substitute omission')
        try:
            actual = signature(recorder, script, args, kwargs)
        except (ValueError, TypeError, IndexError) as exc:
            self.fail(str(exc))
        if actual != self.config['expected_signature']:
            self.fail('Setup call differs from pinned fresh-control signature')
        if recorder.events == 0:
            return False
        if not self.state['first_call_returned']:
            self.fail('First native force did not return normally')
        self.state.update(omitted_calls=1, omitted_event=1, observed_signature=actual)
        self.persist()
        return True

    def returned(self, recorder):
        if self.is_target(recorder) and recorder.frame is None and recorder.events == 1:
            self.state['first_call_returned'] = True
            self.persist()

    def recording(self, index):
        if int(index) == 0:
            if self.state['omitted_calls'] != 1:
                self.fail('Required single omission did not occur before frame zero')
            self.state['recording_started'] = True
            self.persist()


def install(plan):
    """Only used by the opt-in worker; passive tracing source stays untouched."""
    from . import hgwind_startup_trace as trace
    if (plan.get('history_attempts') != [60] or not plan.get('fresh_control')
            or not plan.get('stage_load_barrier')):
        raise ValueError('Force intervention only supports one fresh60 attempt')
    gate = OmitOneSetupForce(plan['setup_force_control'],
        Path(plan['history_cases']['60']['output'])/'startup_force_calls.jsonl')
    original_proxy, original_frame = trace.ForceProxy, trace.startup_frame

    class ControlledProxy(original_proxy):
        def apply_force_at_pos(self, *args, **kwargs):
            if gate.omit(self.recorder, self.script, args, kwargs):
                self.recorder.record(self.script, args, kwargs, OMITTED)
                return None  # Frozen fan uses this native call as a statement.
            result = super().apply_force_at_pos(*args, **kwargs)
            gate.returned(self.recorder)
            return result

    def frame(index):
        gate.recording(index)
        original_frame(index)

    trace.ForceProxy, trace.startup_frame = ControlledProxy, frame
    return gate


def audit_intervention(plan):
    """Require one recorded omission and its exact pinned runtime signature."""
    cfg = plan['setup_force_control']
    path = Path(cfg['audit'])
    result = dict(requested=True, verified=False, reason='audit_missing')
    if not path.exists():
        return result
    state = json.loads(path.read_text())
    trace = Path(plan['history_cases']['60']['output'])/'startup_force_calls.jsonl'
    rows = [json.loads(x) for x in trace.read_text().splitlines()] if trace.exists() else []
    omitted = [row for row in rows if row['status'] == OMITTED]
    valid = (state.get('error') is None and state.get('first_call_returned') is True
        and state.get('recording_started') is True and state.get('omitted_calls') == 1
        and state.get('baseline_report_sha256') == cfg['baseline_report_sha256']
        and state.get('observed_signature') == cfg['expected_signature']
        and len(rows) >= 2 and rows[0]['event'] == 0 and rows[0]['frame'] is None
        and rows[0]['status'] == 'returned'
        and {key: rows[0][key] for key in FIELDS} == cfg['expected_signature']
        and len(omitted) == 1 and omitted[0]['event'] == 1 and omitted[0]['frame'] is None
        and {key: omitted[0][key] for key in FIELDS} == cfg['expected_signature'])
    return dict(requested=True, verified=bool(valid),
        reason='single_omission_verified' if valid else 'intervention_not_verified',
        audit_sha256=sha256(path), modified_forcing_history=True,
        native_forcing_is_unmodified=False, training_eligible=False)
