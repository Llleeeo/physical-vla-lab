#!/usr/bin/env python3
"""Two fixed-condition, fresh-worker LIFT-speed admission diagnostics.

Reuses verified v3 mimic-safe outcomes as off-controls. Default is CPU only.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/derived/hgwind_capture_gate_v4_20261027'
PRIOR = ROOT / 'data/derived/hgwind_mimic_v3_20261027/checks/20261027_220749_166103/report.json'
TASK = 'Trajectory/1s1p/grasp_wind_driven_motion_1s1p_hgprgew'
SCENE_SHA = 'df11a22a0656b6e2a4ae5313789e4cdd2fb0853858ae96e5ce63bca199147846'
CORE_HASHES = {'collector_sha256': 'data/derived/hgwind_current_baseline_20261027/repo/Scripts/expert_data/collect_worker.py',
               'control_adapter_sha256': 'src/capstone_vla/hgwind_control.py',
               'mimic_actuation_adapter_sha256': 'src/capstone_vla/hgwind_mimic_actuation.py'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(offset, run=False):
    result = [sys.executable, str(ROOT / 'scripts/collect_hgwind_baseline.py'),
              '--scene', 'edited', '--seed', '202610280', '--attempt-offset', str(offset),
              '--attempts', '1', '--control-profile', 'mimic-safe', '--capture-speed-gate',
              '--targeting-mode', 'auto', '--experiment-label', 'capture_gate_v4_offset' + str(offset),
              '--baseline-report', str(PRIOR)]
    return result + (['--run'] if run else [])


def load_control():
    report = json.loads(PRIOR.read_text())
    if not report['completed'] or report['scene_sha256'] != SCENE_SHA:
        raise RuntimeError('Original completed v3 report/scene mismatch')
    controls = {}
    for entry in report['runs']:
        if entry['profile'] != 'mimic-safe':
            continue
        run = Path(entry['run'])
        provenance = json.loads((run / 'run.json').read_text())
        if (provenance['exit_code'] != 0 or provenance['seed'] != 202610280
                or provenance['attempt_offset'] != entry['offset']
                or provenance['attempt_budget'] != 1 or provenance['task_sha256'] != SCENE_SHA
                or provenance.get('capture_speed_gate_enabled', False)):
            raise RuntimeError('Old off-control identity or configuration mismatch')
        for key, relative in CORE_HASHES.items():
            if provenance[key] != digest(ROOT / relative):
                raise RuntimeError('Core actuation/motion/collector changed: ' + relative)
        audit = json.loads((run / 'analysis/batch_failure_audit.json').read_text())
        if audit['saved_episodes'] != 1:
            raise RuntimeError('Old control count mismatch')
        row = audit['rows'][0]
        episode = run / TASK / row['episode']
        for name, expected in row['input_sha256'].items():
            if digest(episode / name) != expected:
                raise RuntimeError('Old raw evidence hash mismatch: ' + name)
        controls[entry['offset']] = {'run': run, 'row': row, 'episode': episode}
    if set(controls) != {0, 9}:
        raise RuntimeError('Both prior fixed off-controls are required')
    return controls


def compare(control, run):
    provenance = json.loads((run / 'run.json').read_text())
    audit = json.loads((run / 'analysis/batch_failure_audit.json').read_text())
    if audit['saved_episodes'] != 1 or not provenance['capture_speed_gate_enabled']:
        raise RuntimeError('Missing candidate episode or gate provenance')
    row = audit['rows'][0]
    episode = run / TASK / row['episode']
    meta = json.loads((episode / 'meta.json').read_text())
    gate = meta.get('capstone_capture_speed_gate')
    if (not gate or not gate['enabled'] or gate['required_dwell_s'] != 0
            or not gate.get('revokes_confirmation_when_deferred')):
        raise RuntimeError('Candidate lacks live gate receipt')
    if meta['capstone_gripper_actuation']['actuated_indices'] != [7]:
        raise RuntimeError('Candidate lost independent-jaw actuation')
    if row['attempt_index'] != control['row']['attempt_index'] or row['episode_seed'] != control['row']['episode_seed']:
        raise RuntimeError('Candidate seed/attempt identity mismatch')
    differences = {}
    with h5py.File(control['episode'] / 'replay.hdf5') as old, h5py.File(episode / 'replay.hdf5') as new:
        for key in ('joint_pos', 'joint_vel', 'object_pos', 'object_quat', 'object_vel', 'object_ang_vel'):
            differences[key] = float(np.max(np.abs(old['obs/' + key][0] - new['obs/' + key][0])))
    return {'baseline_run': str(control['run']), 'candidate_run': str(run),
            'old': control['row'], 'new': row, 'gate_receipt': gate,
            'initial_state_max_abs_differences': differences,
            'recorded_initial_states_match_at_1e_7': all(v <= 1e-7 for v in differences.values()),
            'candidate_success': row['category'] == 'success'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--offset', type=int, choices=(0, 9), help='optional one-condition check to avoid rerunning both')
    args = parser.parse_args()
    controls = load_control()
    if args.offset is not None:
        controls = {args.offset: controls[args.offset]}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    for offset in sorted(controls):
        checked = subprocess.run(command(offset), cwd=ROOT, env=env, capture_output=True, text=True)
        if checked.returncode:
            raise RuntimeError('CPU preflight failed: ' + checked.stdout + checked.stderr)
    print('CPU_PREFLIGHT_OK new_attempts=' + str(len(controls)) + ' reused_controls=' + str(len(controls))
          + ' seed=202610280 offsets=' + ','.join(map(str, sorted(controls))), flush=True)
    print('SCOPE only_existing_relative_speed_at_LIFT=true fixed_dwell=0 scene_force_scoring=unchanged', flush=True)
    if not args.run:
        print('PREPARED_ONLY GPU_started=false', flush=True)
        return
    out = BASE / 'checks' / datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    print('OUTPUT ' + str(out), flush=True)
    report = {'completed': False, 'baseline_report': str(PRIOR), 'pairs': {},
              'entry_note': 'Launcher/worker add an explicitly recorded opt-in hook; collector, motion and mimic actuator code hashes match prior off-controls.',
              'causal_limits': 'Recorded initial-state checks do not certify hidden solver state. Two fixed development conditions cannot estimate collection success rate.'}
    for offset, control in sorted(controls.items()):
        log = out / ('offset_' + str(offset) + '.log')
        run = None
        print('CONTROL_BEGIN offset=' + str(offset), flush=True)
        with log.open('w') as handle:
            process = subprocess.Popen(command(offset, run=True), cwd=ROOT, env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                handle.write(line); handle.flush()
                if line.startswith('OUTPUT '):
                    run = Path(line[7:].strip())
                if line.startswith(('OUTPUT ', 'CAPSTONE_', 'CAPTURE_SPEED_GATE', 'COLLECT ATTEMPT', 'COLLECTION_EXIT', 'Traceback', 'RuntimeError')):
                    print(line.rstrip(), flush=True)
            code = process.wait()
        if code or run is None:
            print('\n'.join(log.read_text().splitlines()[-24:]), flush=True)
            raise RuntimeError('Candidate failed to execute; see ' + str(log))
        subprocess.run([sys.executable, str(ROOT / 'scripts/analyze_hgwind_batch.py'),
                        '--run-dir', str(run)], cwd=ROOT, env=env, check=True)
        pair = compare(control, run)
        report['pairs'][str(offset)] = pair
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print('CONTROL_OUTCOME offset=' + str(offset) + ' ' + pair['new']['category'], flush=True)
    report['completed'] = True
    report['evaluated_offsets'] = sorted(controls)
    report['successes'] = sum(p['candidate_success'] for p in report['pairs'].values())
    report['conclusion'] = 'Read physical outcomes and live gate receipt; completing selected controls alone is not evidence of improved grasping.'
    path = out / 'report.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print('CAPTURE_GATE_CHECKS_COMPLETED report=' + str(path), flush=True)
    for offset, pair in report['pairs'].items():
        print('PAIR offset=' + offset + ' old=' + pair['old']['category'] + ' new=' + pair['new']['category']
              + ' initial_match=' + str(pair['recorded_initial_states_match_at_1e_7']), flush=True)


if __name__ == '__main__':
    main()
