"""Single matched numeric-precision diagnostic; not a learned action contract."""
import copy
import json
from pathlib import Path
import h5py
import numpy as np
from .hf_archive import sha256
from .cartesian_target_replay import require
from .ball_action_contract import SLICES, VARIABLE_INPUTS, encode_stream, decode_stream, native_inputs_match
from .cartesian_packets import build_packets
from .posture_anchor import compute_anchor

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'physical-vla-lab.action-precision-control.v1'
BASELINE = ROOT/'data/derived/hgwind_initialized_contract/20261031_105421_302664'
SUMMARY = ROOT/'results/summaries/2026-10-31-initialized-contract-result.json'


def restore_precision(packets, reference32):
    """Restore original numeric labels, with the float32 baseline's sign choices."""
    reference32 = np.asarray(reference32)
    require(reference32.dtype == np.float32 and reference32.shape == (len(packets),29)
        and np.isfinite(reference32).all(), 'Need finite original float32 labels')
    require(np.array_equal(encode_stream(packets),reference32),
        'Original packets differ from the pinned float32 stream')
    actions = reference32.astype(np.float64)
    for i, packet in enumerate(packets):
        require(packet['frame'] == i, 'Noncontiguous original packet')
        for name in VARIABLE_INPUTS:
            value = packet['inputs'][name]
            if value is None:
                continue
            value = np.asarray(value, dtype=np.float64)
            if name == 'target_quaternion' and np.dot(value, reference32[i,SLICES[name]]) < 0:
                value = -value
            actions[i,SLICES[name]] = value
    require(np.isfinite(actions).all() and np.array_equal(actions.astype(np.float32),reference32),
        'Restored labels do not round back to exact baseline; reject confounded control')
    return actions


def matched_plan(plan, baseline):
    for key in ['seed','attempt_offset','frames','source','source_reference_sha256','task_sha256',
            'physics_sha256','patch_sha256','worker_sha256','controller_config_sha256','urdf_sha256',
            'packet_gain_profile','packet_posture_profile','robot_posture_configuration',
            'cartesian_target_capture_sha256','teacher_replay_sha256','original_depth_bias_m',
            'stage_load_barrier','trace_gripper_drives','history_attempts','initialization_reference_report_sha256']:
        require(plan[key] == baseline[key], 'Precision control changes '+key)
    for key in ['setup_force_control','gripper_gain_control']:
        a=copy.deepcopy(plan[key]);b=copy.deepcopy(baseline[key])
        a.pop('audit');b.pop('audit')
        require(a==b, 'Precision control changes '+key)
    # These files handle only optional preparation, output metadata and admission.
    allowed = {'scripts/run_hgwind_initialized_contract.py','scripts/run_hgwind_action_contract.py',
               'src/capstone_vla/initialized_contract.py','scripts/analyze_hgwind_effort_probe.py'}
    for name, digest in baseline['adapter_sha256'].items():
        path=Path(name) if Path(name).is_absolute() else ROOT/name
        if str(path.relative_to(ROOT)) not in allowed:
            require(sha256(path)==digest, 'Baseline execution dependency changed: '+str(path))


def build(plan, reference32):
    capture=json.loads(Path(plan['cartesian_target_capture']).read_text())
    with h5py.File(plan['teacher_replay'],'r') as f:
        anchor=compute_anchor(f['obs/joint_pos'][0,:7], plan['robot_posture_configuration'])
        packets=build_packets(capture,f['actions'][:,7:],'fixed-9-8','latched',anchor)
    restored=restore_precision(packets,reference32)
    # Validate the same live decoder without launching the simulator.
    decoded=decode_stream(restored,anchor)
    require(len(decoded)==319, 'Wrong diagnostic length')
    return restored


def prepare(plan, reference32, diagnostics):
    summary=json.loads(SUMMARY.read_text())
    for name,digest in summary['source_hashes'].items():
        require(sha256(ROOT/name)==digest, 'Measured baseline evidence changed: '+name)
    baseline=json.loads((BASELINE/'plan.json').read_text())
    matched_plan(plan,baseline)
    original=np.load(BASELINE/'actions.npy',allow_pickle=False)
    require(np.array_equal(reference32,original), 'Regenerated float32 stream differs from measured baseline')
    actions=build(plan,original)
    plan['precision_control']=dict(schema=SCHEMA,baseline=str(BASELINE),
        baseline_plan_sha256=sha256(BASELINE/'plan.json'),
        baseline_actions_sha256=sha256(BASELINE/'actions.npy'),
        baseline_report_sha256=sha256(BASELINE/'report.json'),
        source_dtype='float64',comparison_dtype='float32',same_decoder=True,
        float32_roundtrip_exact=True,learned_policy=False)
    plan['reference_fidelity']='Diagnostic original float64 labels through the unchanged29D decoder; same initialized fresh60, fixed gains, latched posture, callbacks and quaternion hemispheres as pinned float32 failure. Not a model-compatible precision claim.'
    plan['adapter_sha256'][str(Path(__file__).resolve())]=sha256(__file__)
    import hashlib
    diagnostics.update(action_dtype='float64',action_array_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
        precision_control=SCHEMA,float32_roundtrip_exact=True,matched_baseline=BASELINE.name,
        model_training_eligible=False)
    return actions


def verify_runtime(plan):
    control=plan['precision_control']
    require(control['schema']==SCHEMA and control['source_dtype']=='float64'
        and control['comparison_dtype']=='float32' and control['same_decoder'] is True
        and control['learned_policy'] is False, 'Invalid precision diagnostic declaration')
    baseline=Path(control['baseline'])
    require(baseline.resolve()==BASELINE.resolve(), 'Unpinned precision baseline')
    for name,key in [('plan.json','baseline_plan_sha256'),('actions.npy','baseline_actions_sha256'),
                     ('report.json','baseline_report_sha256')]:
        require(sha256(baseline/name)==control[key], 'Precision baseline changed: '+name)
    matched_plan(plan,json.loads((baseline/'plan.json').read_text()))
    require(sha256(plan['action_vectors'])==plan['action_vectors_sha256'], 'Diagnostic actions changed')
    actions=np.load(plan['action_vectors'],allow_pickle=False)
    original=np.load(baseline/'actions.npy',allow_pickle=False)
    require(actions.dtype==np.float64 and np.array_equal(actions,build(plan,original)),
        'Diagnostic labels must be regenerated original float64 values, not widened float32')
    return actions


def audit(plan, initialization):
    root=Path(plan['output'])
    result=dict(schema=SCHEMA,initialization_verified=initialization['initialization_verified'],
        issued_prefix_inputs_match=False,full_execution=False,grasp=initialization['grasp'],
        combined_passed=False,learned_policy_evaluated=False,training_eligible=False,errors=[])
    try:
        actions=verify_runtime(plan)
        payload=json.loads((root/'cartesian_packet_execution.json').read_text())
        packets=decode_stream(actions,payload['posture_initialization']['anchor_rad'])
        events=payload['calls']
        prefix=bool(events) and len(events)<=len(packets) and all(
            e['frame']==p['frame'] and native_inputs_match(e['inputs'],dict(p['inputs'],**p['safety_callbacks']))
            and e['gripper_closed'] is p['gripper_closed']
            and e['safety_gripper_closed'] is p['gripper_closed']
            and np.array_equal(e['jaw_action']['joint_indices'],[7,8])
            and np.array_equal(np.asarray(e['jaw_action']['joint_positions'],np.float32),
                np.full(2,0 if p['gripper_closed'] else .04,np.float32))
            for e,p in zip(events,packets))
        report=json.loads((root/'report.json').read_text())
        result.update(issued_prefix_inputs_match=prefix,issued_commands=len(events),
            completed_intervals=report['completed_frames'],requested_intervals=plan['frames'],
            full_execution=initialization['execution_complete'],
            combined_passed=initialization['combined_passed'],
            independent_grasp_evidence=report['independent_grasp_evidence'],
            baseline_completed_intervals=179,baseline_grasp='fail',
            interpretation='One exposed precision diagnostic; success does not certify float32/model robustness.')
    except (OSError,ValueError,KeyError,TypeError) as exc:
        result['errors'].append(repr(exc))
    (root/'precision_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PRECISION_CONTROL initialization='+str(result['initialization_verified']).lower()
        +' completed='+str(result.get('completed_intervals','unknown'))+'/319'
        +' prefix_inputs_match='+str(result['issued_prefix_inputs_match']).lower()
        +' grasp='+result['grasp']+' combined_passed='+str(result['combined_passed']).lower(),flush=True)
    return result
