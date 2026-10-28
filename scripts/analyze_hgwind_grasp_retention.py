#!/usr/bin/env python3
"""Read-only alignment of three approved grasps and two cold-worker candidates."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'data/derived/hgwind_current_baseline_20261027/repo/runs'
TASK = 'Trajectory/1s1p/grasp_wind_driven_motion_1s1p_hgprgew'
CHECK = ROOT / 'data/derived/hgwind_mimic_v3_20261027/checks/20261027_220749_166103'


def sources():
    episodes = [(f'approved_success_{i}', RUNS / '20261027_185825_637620' / TASK / f'traj_{i:05d}') for i in range(3)]
    for entry in json.loads((CHECK / 'report.json').read_text())['runs']:
        if entry['profile'] != 'mimic-safe':
            continue
        run = Path(entry['run'])
        row = json.loads((run / 'analysis/batch_failure_audit.json').read_text())['rows'][0]
        episodes.append((f'candidate_offset_{entry["offset"]}', run / TASK / row['episode']))
    return episodes


def measure(label, episode):
    meta = json.loads((episode / 'meta.json').read_text())
    telemetry = {}
    for line in (episode / 'analyse.jsonl').open():
        record = json.loads(line)
        if record['type'] == 'control_telemetry':
            telemetry[record['data']['name']] = np.asarray(record['data']['values'])
    with h5py.File(episode / 'replay.hdf5') as f:
        t = f['timestamp'][:]
        phase = telemetry['skill_phase'].reshape(len(t), -1)[:, 0]
        starts = np.flatnonzero((phase == 12) & np.r_[True, phase[:-1] != 12])
        if not len(starts):
            raise RuntimeError('No LIFT in ' + str(episode))
        start = int(starts[0])
        following = np.flatnonzero((np.arange(len(t)) > start) & (phase != 12))
        end = int(following[0]) if len(following) else len(t) - 1
        oi = meta['object_paths'].index(meta['policy']['roles']['A'])
        ball, ee = f['obs/object_pos'][:, oi], f['obs/ee_pos'][:]
        quat = f['obs/ee_quat'][:]
        rotation = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
        pinch = ee + rotation.apply(meta['validation']['pinch_frame']['offset_local_m'])
        relative = rotation.inv().apply(ball - pinch)
        slip = np.linalg.norm(relative - relative[start], axis=1)
        both = telemetry['contact_L'] & telemetry['contact_R']
        first_contact = int(np.flatnonzero(both)[0])
        # Do not include the initial 90mm spawn/drop in a grasp lift metric.
        peak_after_contact = float((ball[first_contact:, 2].max() - ball[first_contact, 2]) * 1000)
        speed = telemetry['capture_relative_speed']
        row = {'label': label, 'episode': str(episode), 'success': meta['success'],
               'first_lift_frame': start, 'first_lift_end_frame': end,
               'first_lift_end_phase_code': int(phase[end]),
               'first_lift_relative_speed_mps': float(speed[start]),
               'first_lift_ball_relative_to_pinch_local_mm': (relative[start] * 1000).tolist(),
               'relative_center_displacement_at_first_lift_end_mm': float(slip[end] * 1000),
               'first_bilateral_frame': first_contact,
               'peak_ball_lift_after_first_bilateral_mm': peak_after_contact,
               'all_first_lift_close_requested': bool(telemetry['close_request'][start:end].all()),
               'finger_joint_positions_entry_mm': (f['obs/joint_pos'][start, 7:] * 1000).tolist(),
               'finger_joint_targets_entry_mm': (f['actions'][start, 7:] * 1000).tolist(),
               'input_sha256': {name: hashlib.sha256((episode / name).read_bytes()).hexdigest()
                                for name in ('meta.json', 'analyse.jsonl', 'replay.hdf5')}}
        trace = {'time': t - t[start], 'ball_height': (ball[:, 2] - ball[start, 2]) * 1000,
                 'ee_height': (ee[:, 2] - ee[start, 2]) * 1000, 'slip': slip * 1000,
                 'speed': speed, 'end': t[end] - t[start]}
        return row, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'data/derived/hgwind_capture_gate_v4_20261027/retention_analysis')
    args = parser.parse_args()
    rows, traces = zip(*(measure(label, episode) for label, episode in sources()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(len(rows), 3, figsize=(13, 2.1 * len(rows)), constrained_layout=True)
    limits = []
    for keys in (('ball_height', 'ee_height'), ('slip',), ('speed',)):
        visible = np.concatenate([trace[key][(trace['time'] >= -.12) & (trace['time'] <= 1.5)]
                                  for trace in traces for key in keys])
        low, high = float(visible.min()), float(visible.max())
        padding = max((high-low)*.06, 1e-5)
        limits.append((low-padding, high+padding))
    for panels, row, trace in zip(axes, rows, traces):
        panels[0].plot(trace['time'], trace['ball_height'], label='ball')
        panels[0].plot(trace['time'], trace['ee_height'], label='EE')
        panels[0].set_ylabel(row['label'] + '\nHeight change (mm)')
        panels[1].plot(trace['time'], trace['slip'])
        panels[1].set_ylabel('Center shift in hand (mm)')
        panels[2].plot(trace['time'], trace['speed'])
        panels[2].set_ylabel('Relative speed (m/s)')
        for axis, ylimits in zip(panels, limits):
            axis.set_ylim(*ylimits)
            axis.axvline(0, color='black', linestyle=':', linewidth=1)
            axis.axvline(trace['end'], color='gray', linestyle='--', linewidth=1)
            axis.set_xlim(-.12, min(1.5, trace['time'][-1]))
            axis.grid(alpha=.2)
    axes[0, 0].legend()
    for axis in axes[-1]:
        axis.set_xlabel('Time from first LIFT entry (s)')
    figure.suptitle('Approved successes vs physical-actuation candidates; observational comparison\nBlack: first LIFT. Gray: end of first LIFT (HOLD or return to TRACK).')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image = args.out_dir / 'retention_comparison.png'
    figure.savefig(image, dpi=130)
    result = {'scope': 'Read-only development comparison; different seeds, dwell and actuation confound causal attribution.',
              'rows': rows, 'figure': str(image),
              'verified_code_path': 'Current _capture_contact_transition_ready accepts capture_speed_safe but its physical-contact return does not use it.',
              'candidate': 'Opt-in gate at frictional LIFT entry using the existing payload-aware relative-speed limit, with no fixed dwell.',
              'unknowns': ['Speed correlation alone does not establish cause.', 'Center displacement in the hand is not material-point tangential slip or ball spin.', 'Not all normal contact impulse is available friction force or a grasp-force capacity certificate.', 'Success samples are selected outcomes, not a success-rate estimate.']}
    path = args.out_dir / 'report.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    for row in rows:
        print('RETENTION ' + json.dumps({k: row[k] for k in ('label', 'success', 'first_lift_relative_speed_mps', 'relative_center_displacement_at_first_lift_end_mm', 'peak_ball_lift_after_first_bilateral_mm')}))
    print('RETENTION_REPORT ' + str(path))
    print('FIGURE ' + str(image))


if __name__ == '__main__':
    main()
