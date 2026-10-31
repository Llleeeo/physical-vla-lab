#!/usr/bin/env python3
"""CPU-only provenance and pre-collision translation audit of exposed traj18.

Requires local frozen evidence. Does not relabel data, run models or simulate.
Nominal translation requests are not final robot velocities or grasp outcomes.
"""
import argparse
import ast
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from capstone_vla.ball_action_contract import decode_action, encode_stream
from capstone_vla.ball_noise_replanning import assemble_prefix, SEEDS
from capstone_vla.cartesian_packets import build_packets
from capstone_vla.hf_archive import sha256

DATA = ROOT/'data/derived/hgwind_traj18_development_20261031'
FROZEN = ROOT/'data/derived/hgwind_production_20261027/worker_bundle/data/derived/hgwind_current_baseline_20261027/repo/Scripts/expert_data'
SURROGATE = ROOT/'data/derived/hgwind_action_contract_v1/20261029_220618_809794_traj_00018'
NOISE = DATA/'training_runs/20261031_133555_985430/noise_replanning_diagnostics/20261031_150546_311797'


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    manifest = json.loads((DATA/'dataset.json').read_text())
    plan = json.loads((SURROGATE/'plan.json').read_text())
    source, capture_path = Path(plan['source']), Path(plan['cartesian_target_capture'])
    telemetry_path = source.with_name('analyse.jsonl')
    paths = [Path(__file__), DATA/'dataset.json', DATA/'arrays.npz', SURROGATE/'plan.json',
             SURROGATE/'report.json', SURROGATE/'actions.npy', source, capture_path,
             telemetry_path, FROZEN/'skills/catch.py', FROZEN/'expert.py',
             FROZEN/'kinematic_servo.py', NOISE/'report.json', NOISE/'predictions.npz']
    require(sha256(DATA/'arrays.npz') == manifest['payload_sha256'], 'Dataset changed')
    for path in (source, capture_path, SURROGATE/'actions.npy', SURROGATE/'plan.json'):
        require(sha256(path) == manifest['source_hashes'][str(path)], 'Provenance mismatch: '+path.name)
    expected = next(x['sha256'] for x in plan['source_files'] if x['path']=='analyse.jsonl')
    require(sha256(telemetry_path)==expected, 'Original telemetry changed')
    teacher_plan_path = capture_path.with_name('plan.json')
    teacher_plan = json.loads(teacher_plan_path.read_text())
    bundle_manifest = Path(teacher_plan['bundle_manifest'])
    manifest_entries = {row['path']:row['sha256'] for row in json.loads(bundle_manifest.read_text())['files']}
    for name in ('skills/catch.py', 'expert.py', 'kinematic_servo.py'):
        native = FROZEN/name
        require(sha256(native)==manifest_entries[str(native.relative_to(bundle_manifest.parent))], 'Frozen native source changed')
    paths += [teacher_plan_path,bundle_manifest]
    capture = json.loads(capture_path.read_text())
    a = np.load(DATA/'arrays.npz', allow_pickle=False)
    state, target, ts = a['states'], a['actions'], a['timestamp']
    calls = capture['calls']; count = len(calls)
    require(len(target)==269 and count==254 and capture['frames']==list(range(269)), 'Unexpected capture')
    require([c['frame'] for c in calls]==list(range(count)) and all(c['completed'] for c in calls), 'Call/frame mapping')
    with h5py.File(source) as f:
        packets = build_packets(capture, f['actions'][:,7:], 'fixed-9-8', 'latched', state[0,21:28])
        require(np.array_equal(encode_stream(packets), target), 'Rebuilt labels differ')
        require(np.array_equal(f['timestamp'][:], ts), 'Source timestamps differ')
        object_z = f['obs/object_pos'][:10,0,2].tolist()
    require(np.array_equal(target, np.load(SURROGATE/'actions.npy')), 'Passing contract labels differ')
    require(np.allclose(ts,np.arange(269)/60,atol=1e-7,rtol=0), 'Clock mismatch')
    telemetry = {x['data']['name']:np.asarray(x['data']['values']) for x in map(json.loads,telemetry_path.read_text().splitlines()) if x['type']=='control_telemetry'}
    captured = np.asarray([c['inputs']['target_position'] for c in calls])
    require(np.array_equal(captured.astype(np.float32), target[:count,:3]), 'XYZ conversion mismatch')
    # Original telemetry is before the reviewed <=2mm downward goal adapter.
    goal = telemetry['live_grasp_target_pos'][:count]
    difference = captured-goal
    require(np.max(np.abs(difference[:10,:2]))<3e-7 and np.max(np.abs(difference[:10,2]+.002))<3e-7, 'Startup depth profile differs')
    # Execute the original pure helper in isolation; do not import simulator skill.
    tree=ast.parse((FROZEN/'skills/catch.py').read_text())
    helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_live_pregrasp_standoff_goal')
    env={'np':np};exec(compile(ast.Module(body=[helper],type_ignores=[]),'<frozen standoff helper>','exec'),env)
    axis=telemetry['center_approach_axis_world'][0]
    # Axial invariant does not need unknown pre-governor transverse goal/center.
    axial_shift=float(np.dot(goal[0]-state[0,14:17],axis))
    require(abs(axial_shift)<3e-5, 'Startup target did not retain measured axial height')
    sample=env['_live_pregrasp_standoff_goal'](goal[1], goal[1], state[0,14:17], axis,
        float(telemetry['pregrasp_safe_standoff_m'][0]),target_motion_live_now=False)
    require(abs(np.dot(sample-state[0,14:17],axis))<1e-12, 'Frozen helper invariant')
    spec=importlib.util.spec_from_file_location('audited_frozen_servo',FROZEN/'kinematic_servo.py')
    servo=importlib.util.module_from_spec(spec);sys.modules[spec.name]=servo;spec.loader.exec_module(servo)
    def nominal(stream):
        rows=[]
        for frame, action in enumerate(stream):
            i=decode_action(action,frame,state[0,21:28])['inputs']
            rows.append(servo.linear_task_velocity(np.asarray(i['target_position'])-state[frame,14:17],
                i['target_linear_velocity'],config=servo.KinematicServoConfig(position_gain=i['position_gain'],maximum_linear_speed_mps=i['maximum_linear_speed_mps']),
                linear_velocity_reference_mps=i['linear_velocity_reference'],maximum_relative_linear_speed_mps=i['maximum_relative_linear_speed_mps']))
        return np.asarray(rows)
    reference = nominal(target)
    nr=json.loads((NOISE/'report.json').read_text())
    require(sha256(NOISE/'predictions.npz')==nr['predictions_sha256'], 'Predictions changed')
    pred=np.load(NOISE/'predictions.npz',allow_pickle=False)
    require(np.array_equal(pred['seeds'],SEEDS) and np.array_equal(pred['frames'],np.arange(0,269,4)), 'Prediction alignment')
    rows=[]
    for si,seed in enumerate(SEEDS):
        stream=assemble_prefix(pred['chunks'][0,si]); cmd=nominal(stream)
        error=np.linalg.norm(cmd-reference,axis=1)
        rows.append({'seed':int(seed), **{name:float(error[lo:hi].mean()) for name,lo,hi in [
            ('startup_0_39_mean_mps',0,40),('near_80_119_mean_mps',80,120),('full_mean_mps',0,269)]}})
    result=dict(status='CPU_UPPER_TARGET_AUDIT_PASS',source_episode='local/traj_00018',
        labels_rebuilt_exact=True,frames=269,captured_calls=count,terminal_hold_frames=269-count,
        source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in paths},
        startup=dict(target_xyz_m=target[:10,:3].tolist(),measured_pinch_xyz_m=state[:10,14:17].tolist(),object_origin_z_m=object_z,
            target_z_change_0_to_1_m=float(captured[1,2]-captured[0,2]),pinch_z_change_0_to_1_m=float(state[1,16]-state[0,16]),
            nominal_translation_mps=reference[:10].tolist(),native_admitted_translation_mps=[c['pending_operational_reference']['commanded_task_velocity'][:3] for c in calls[:10]],
            pre_depth_target_axial_difference_frame0_m=axial_shift,
            telemetry={k:telemetry[k][:10].tolist() for k in ['skill_phase','phenomenon_phase','pregrasp_corridor_ingress_scale','target_pose_ingress_scale','approach_servo_barrier_active']},
            gate_interpretation='Source helper and recorded axial invariant support dynamic-ingress hold then release; internal pre-governor arguments/liveness flag were not captured, so no full branch replay is claimed.'),
        nominal_translation_comparison=dict(scope='All 8 retained-parent independent-noise streams; same expert measured state per frame; before geometry projection, IK and dynamics.',
            per_seed=rows,mean_over_seeds={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='seed'}),
        limitations=['Single exposed development demonstration; no held-out inference.',
            'Nominal request error is not measured arm velocity or post-safety command error.',
            'No evidence for deleting/smoothing labels; no policy/controller change or grasp improvement claimed.'])
    out=args.output or DATA/'upper_target_audits'/datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')/'report.json'
    out.parent.mkdir(parents=True,exist_ok=True);require(not out.exists(),'Output exists')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result['status']);print(json.dumps(result['nominal_translation_comparison']['mean_over_seeds']));print('REPORT',out)


if __name__=='__main__':
    main()
