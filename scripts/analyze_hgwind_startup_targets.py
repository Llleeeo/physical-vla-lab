#!/usr/bin/env python3
"""Read-only CPU audit of exposed ball startup targets and nominal translation.

--write saves a fresh report under data/derived. No model inference, controller
execution, source modification or simulation is performed. Nominal requests
exclude collision projection, Jacobian/IK and the effort/physics loop.
"""
import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from capstone_vla.ball_action_contract import decode_action
from capstone_vla.ball_prediction_metrics import distribution
from capstone_vla.hf_archive import sha256

DATA = ROOT / 'data/derived/hgwind_traj18_development_20261031'
CONTRACT = ROOT / 'data/derived/hgwind_action_contract_v1/20261029_220618_809794_traj_00018'
PREDICTIONS = {
    '200': '20261031_115841_081174/offline_predictions/20261031_120646_484791',
    '2000': '20261031_121355_529372/offline_predictions/20261031_121834_695770',
}
PHASES = {'startup': (0, 42), 'approach': (42, 144),
          'closure_to_source_contact': (144, 162), 'source_contact_onward': (162, 269)}


def audit():
    plan_file = CONTRACT / 'plan.json'
    plan = json.loads(plan_file.read_text())
    manifest_file = DATA / 'dataset.json'
    manifest = json.loads(manifest_file.read_text())
    for path, digest in manifest['source_hashes'].items():
        if sha256(path) != digest:
            raise ValueError('Source changed: ' + path)
    if sha256(DATA / 'arrays.npz') != manifest['payload_sha256']:
        raise ValueError('Development array payload changed')
    capture_file = Path(plan['cartesian_target_capture'])
    if sha256(capture_file) != plan['cartesian_target_capture_sha256']:
        raise ValueError('Captured targets changed')
    capture = json.loads(capture_file.read_text())
    with np.load(DATA / 'arrays.npz', allow_pickle=False) as f:
        actions, states = f['actions'].copy(), f['states'].copy()
    if actions.shape != (269, 29) or states.shape != (269, 30):
        raise ValueError('Wrong episode shape')
    if not (np.isfinite(actions).all() and np.isfinite(states).all()):
        raise ValueError('Nonfinite development arrays')
    n = len(capture['calls'])
    if [c['frame'] for c in capture['calls']] != list(range(n)):
        raise ValueError('Captured calls are not consecutive pre-action frames')
    raw = np.array([c['inputs']['target_position'] for c in capture['calls']])
    np.testing.assert_array_equal(raw.astype(np.float32), actions[:n, :3])
    replay = Path(plan['teacher_replay'])
    telemetry_file = replay.with_name('analyse.jsonl')
    rows = [json.loads(line) for line in telemetry_file.read_text().splitlines()]
    telemetry = {r['data']['name']: np.asarray(r['data']['values'])
                 for r in rows if r['type'] == 'control_telemetry'}
    with h5py.File(replay, 'r') as f:
        paths = [x.decode() if isinstance(x, bytes) else str(x)
                 for x in f['scene/object_bindings/object_paths'][:]]
        target_path = json.loads(f['evidence/contacts/control_target_paths'][0])[0]
        binding = paths.index(target_path)
        idx = int(f['scene/object_bindings/observation_indices'][binding])
        ball_pos = f['obs/object_pos'][:, idx]
        ball_vel = f['obs/object_vel'][:, idx]
    # Bind the otherwise unpinned telemetry to independently captured motor inputs.
    live_goal = telemetry['live_grasp_target_pos']
    expected = live_goal[:n].copy()
    # This exposed local profile uses a separately logged 2mm downward bias.
    teacher_plan = json.loads(capture_file.with_name('plan.json').read_text())
    depth = float(teacher_plan['original_depth_bias_m'])
    # Bias is applied by the original helper only while jaws are open.
    expected[actions[:n, 28] < .5, 2] -= depth
    startup_goal_difference = float(np.max(np.abs(expected[:42] - raw[:42])))
    if startup_goal_difference > 1.e-6:
        raise ValueError('Startup telemetry disagrees with captured target + depth')
    frozen = Path(plan['worker']).parents[2] / 'Scripts/expert_data'
    bundle_manifest = Path(plan['bundle_manifest'])
    if sha256(bundle_manifest) != plan['bundle_manifest_sha256']:
        raise ValueError('Frozen source manifest changed')
    frozen_hashes = {r['path']: r['sha256'] for r in json.loads(bundle_manifest.read_text())['files']}
    for name in ['kinematic_servo.py', 'expert.py', 'skills/catch.py']:
        source = frozen / name
        if sha256(source) != frozen_hashes[str(source.relative_to(bundle_manifest.parent))]:
            raise ValueError('Frozen control source changed: ' + name)
    module_file = frozen / 'kinematic_servo.py'
    spec = importlib.util.spec_from_file_location('ball_startup_frozen_servo', module_file)
    servo = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = servo
    spec.loader.exec_module(servo)

    def nominal(vector, frame):
        u = decode_action(vector, int(frame))['inputs']
        config = servo.KinematicServoConfig(
            position_gain=u['position_gain'], maximum_linear_speed_mps=u['maximum_linear_speed_mps'])
        return servo.linear_task_velocity(
            np.asarray(u['target_position']) - states[frame, 14:17],
            u['target_linear_velocity'], config=config,
            linear_velocity_reference_mps=u['linear_velocity_reference'],
            maximum_relative_linear_speed_mps=u['maximum_relative_linear_speed_mps'])

    reference = np.stack([nominal(v, i) for i, v in enumerate(actions)])
    models = {}; files = [plan_file, manifest_file, DATA / 'arrays.npz', capture_file,
                          replay, telemetry_file, bundle_manifest, module_file, frozen / 'expert.py',
                          frozen / 'skills/catch.py', Path(__file__).resolve()]
    for name, suffix in PREDICTIONS.items():
        folder = DATA / 'training_runs' / suffix
        report_file = folder / 'report.json'
        archive = folder / 'predictions.hdf5'
        report = json.loads(report_file.read_text())
        if sha256(archive) != report['prediction_archive_sha256']:
            raise ValueError('Prediction archive changed')
        for path, digest in report['source_hashes'].items():
            if sha256(path) != digest:
                raise ValueError('Prediction input changed: ' + path)
        with h5py.File(archive, 'r') as f:
            if not bool(f.attrs['complete']):
                raise ValueError('Incomplete prediction')
            np.testing.assert_array_equal(f['target_absolute29'][:], actions)
            pred = f['finetuned/absolute_chunks'][:, 0]
        values = np.full((269, 3), np.nan); errors = []
        for i, v in enumerate(pred):
            try:
                values[i] = nominal(v, i)
            except ValueError as exc:
                errors.append({'frame': i, 'reason': str(exc)})
        phases = {}
        for phase, (start, end) in PHASES.items():
            e = np.linalg.norm(values[start:end] - reference[start:end], axis=1)
            valid = np.isfinite(e)
            phases[phase] = {'start': start, 'end_exclusive': end,
                             'rejected': int((~valid).sum()),
                             'nominal_velocity_error_mps_valid_only': distribution(e[valid]),
                             'upper_position_error_mm': distribution(
                                 np.linalg.norm(pred[start:end, :3].astype(float)
                                                - actions[start:end, :3], axis=1) * 1000)}
        models[name] = {'nominal_request_rejected_frames': errors, 'phases': phases}
        files += [report_file, archive]
    jumps = np.linalg.norm(np.diff(raw, axis=0), axis=1) * 1000
    jump_frames = np.flatnonzero(jumps > 100) + 1
    # Descriptive sign reversals; do not label every small support chatter a bounce.
    reversals = np.flatnonzero((ball_vel[:41, 2] < -.15) & (ball_vel[1:42, 2] > .15)) + 1
    examples = []
    for i in [0, 1, 7, 8, 17, 18, 24, 25, 30, 31, 35, 40]:
        center = ball_pos[i]
        axis = telemetry['center_approach_axis_world'][i]
        candidate = center - axis * float(telemetry['pregrasp_safe_standoff_m'][i])
        examples.append({'frame': i, 'ball_z_m': float(center[2]),
            'ball_vz_mps': float(ball_vel[i, 2]), 'upper_z_m': float(raw[i, 2]),
            'pinch_z_m': float(states[i, 16]),
            'pregrasp_safe_standoff_m': float(telemetry['pregrasp_safe_standoff_m'][i]),
            'dynamic_safe_distance_m': float(telemetry['dynamic_safe_distance_m'][i]),
            'pose_ingress_scale': float(telemetry['target_pose_ingress_scale'][i]),
            'live_goal_to_standoff_candidate_mm': float(np.linalg.norm(live_goal[i] - candidate) * 1000),
            'nominal_request_mps': reference[i].tolist(),
            'captured_post_ik_request_mps': capture['calls'][i]['pending_operational_reference']['commanded_task_velocity'][:3]})
    return {'schema': 'physical-vla-lab.ball-startup-audit.v1',
        'scope': 'exposed local/traj_00018; recorded robot states, no model inference or physics',
        'depth_bias_m': depth, 'startup_capture_telemetry_max_component_difference_m': startup_goal_difference,
        'upper_jumps_over_100mm': [{'frame': int(i), 'mm': float(jumps[i-1])} for i in jump_frames],
        'startup_vertical_reversal_frames': reversals.tolist(), 'examples': examples, 'models': models,
        'sources': {str(p.relative_to(ROOT)): sha256(p) for p in files},
        'nominal_comparison_includes': 'position feedback + predicted feedforward + absolute/relative speed bounds',
        'nominal_comparison_excludes': 'collision projection, orientation, Jacobian/IK, effort feasibility, dynamics and live policy feedback',
        'conclusion': 'Target jumps are present in teacher before model conversion; initial falling/rebounds and dynamic pregrasp braking clearance contribute. Several upward target peaks numerically match the recorded standoff construction. Bounded nominal requests still differ. No target deletion/smoothing or threshold change is justified by this audit alone.',
        'simulation_rollout': False, 'learned_grasp_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    report = audit()
    print('STARTUP_AUDIT_VERIFIED jumps=' + str(len(report['upper_jumps_over_100mm']))
          + ' reversals=' + str(report['startup_vertical_reversal_frames']) + ' CUDA=false physics=false')
    for name, result in report['models'].items():
        print(name, json.dumps({'nominal_rejected': len(result['nominal_request_rejected_frames']),
                              'startup': result['phases']['startup']}))
    if args.write:
        out = ROOT / 'data/derived/hgwind_startup_target_audit' / datetime.now(
            ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')
        out.mkdir(parents=True, exist_ok=False)
        (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print('REPORT', out / 'report.json')


if __name__ == '__main__':
    main()
