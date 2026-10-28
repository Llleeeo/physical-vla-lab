#!/usr/bin/env python3
"""Four cold-worker development controls for independent-jaw actuation.

Default performs CPU preflights only. --run starts four sequential simulator
workers, preserves every outcome, and writes a compact paired diagnostic.
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
BASE = ROOT / 'data/derived/hgwind_mimic_v3_20261027'
TASK_ID = 'grasp_wind_driven_motion_1s1p_hgprgew'
SCENE = ROOT / 'data/derived/hgwind_scene_edit_20261027/Tasks/1s1p' / (TASK_ID + '.usd')
SCENE_SHA = 'df11a22a0656b6e2a4ae5313789e4cdd2fb0853858ae96e5ce63bca199147846'
BASELINE = ROOT / 'data/derived/hgwind_current_baseline_20261027/repo/runs/transition_v2_20261027_211245_534833/analysis/batch_failure_audit.json'
CASES = [('previous_fail', 0), ('previous_success', 9)]
OBSERVATIONS = ('joint_pos', 'joint_vel', 'object_pos', 'object_quat', 'object_vel', 'object_ang_vel')


def command(condition, offset, profile, run=False):
    cmd = [sys.executable, str(ROOT / 'scripts/collect_hgwind_baseline.py'),
           '--scene', 'edited', '--targeting-mode', 'auto', '--seed', '202610280',
           '--attempt-offset', str(offset), '--attempts', '1', '--control-profile', profile,
           '--experiment-label', 'mimic_v3_' + condition + '_' + ('new' if profile == 'mimic-safe' else 'old'),
           '--baseline-report', str(BASELINE)]
    if run:
        cmd.append('--run')
    return cmd


def load_reusable_controls(path):
    """Reuse only completed controls with unchanged inputs and relevant code."""
    path = Path(path).resolve()
    if path.is_dir():
        path = path / 'progress.json'
    progress = json.loads(path.read_text())
    if progress['scene_sha256'] != SCENE_SHA:
        raise RuntimeError('Resume scene hash mismatch')
    controls = {}
    expected = {(name, profile): offset for name, offset in CASES
                for profile in ('smooth-only', 'mimic-safe')}
    for entry in progress['runs']:
        if entry['exit_code'] != 0:
            continue
        key = (entry['condition'], entry['profile'])
        if key not in expected or entry['offset'] != expected[key]:
            raise RuntimeError('Resume control identity mismatch')
        run = Path(entry['run'])
        provenance = json.loads((run / 'run.json').read_text())
        required = {'seed': 202610280, 'attempt_budget': 1,
                    'attempt_offset': expected[key], 'expert_targeting_mode_requested': 'auto',
                    'expert_control_profile': key[1], 'task_sha256': SCENE_SHA,
                    'task_sha256_after': SCENE_SHA, 'exit_code': 0}
        if any(provenance.get(k) != v for k, v in required.items()):
            raise RuntimeError('Resume provenance mismatch: ' + str(run))
        for relative in ('src/capstone_vla/hgwind_control.py', 'scripts/sim/collect_hgwind_worker.py',
                         'scripts/sim/dispatch_hgwind_collection.py', 'scripts/collect_hgwind_baseline.py'):
            if (run / 'control_snapshot' / relative).read_bytes() != (ROOT / relative).read_bytes():
                raise RuntimeError('Resume control source changed: ' + relative)
        if provenance['collector_sha256'] != hashlib.sha256(
                (ROOT / 'data/derived/hgwind_current_baseline_20261027/repo/Scripts/expert_data/collect_worker.py').read_bytes()).hexdigest():
            raise RuntimeError('Resume upstream collector changed')
        # A fixed adapter must never silently reuse a result from its old version.
        if key[1] == 'mimic-safe' and provenance.get('mimic_actuation_adapter_sha256') != hashlib.sha256(
                (ROOT / 'src/capstone_vla/hgwind_mimic_actuation.py').read_bytes()).hexdigest():
            raise RuntimeError('Resume candidate adapter changed; start a fresh diagnostic')
        audit = json.loads((run / 'analysis/batch_failure_audit.json').read_text())
        if audit['saved_episodes'] != 1 or audit['run_exit_code'] != 0:
            raise RuntimeError('Resume physical episode incomplete')
        row = audit['rows'][0]
        if row['attempt_index'] != expected[key]:
            raise RuntimeError('Resume attempt index mismatch')
        episode = run / 'Trajectory/1s1p' / TASK_ID / row['episode']
        for name, digest in row['input_sha256'].items():
            if hashlib.sha256((episode / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('Resume raw evidence changed: ' + name)
        metadata = json.loads((episode / 'meta.json').read_text())
        controls[key] = {'run': run, 'row': row,
                         'receipt': metadata.get('capstone_gripper_actuation'),
                         'source_entry': entry}
    return path, controls


def initial_state(run):
    root = run / 'Trajectory/1s1p' / TASK_ID
    episodes = sorted(root.glob('*/replay.hdf5'))
    if len(episodes) != 1:
        raise RuntimeError('Expected exactly one saved physical episode: ' + str(run))
    with h5py.File(episodes[0]) as handle:
        return {k: handle['obs/' + k][0] for k in OBSERVATIONS}


def summarize_pair(old, new):
    a, b = initial_state(old['run']), initial_state(new['run'])
    differences = {k: float(np.max(np.abs(a[k] - b[k]))) for k in OBSERVATIONS}
    keys = ('category', 'bilateral_frames', 'longest_bilateral_s', 'post_contact_final_lift_m',
            'maximum_raw_joint_speed_ratio', 'terminal_reason', 'no_assistance_observed')
    result = {'initial_recorded_state_max_abs_differences': differences,
              'recorded_initial_states_match_at_1e_7': all(v <= 1e-7 for v in differences.values()),
              'old': {k: old['row'][k] for k in keys}, 'new': {k: new['row'][k] for k in keys},
              'same_episode_seed': old['row']['episode_seed'] == new['row']['episode_seed'],
              'candidate_actuation_receipt': new['receipt']}
    result['candidate_success'] = new['row']['category'] == 'success'
    result['both_success'] = result['candidate_success'] and old['row']['category'] == 'success'
    result['failure_to_success'] = result['candidate_success'] and old['row']['category'] != 'success'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--resume', type=Path, help='prior checks folder or progress.json; reuse verified completed controls')
    args = parser.parse_args()
    if hashlib.sha256(SCENE.read_bytes()).hexdigest() != SCENE_SHA:
        raise RuntimeError('Edited scene changed; review this fixed-condition diagnostic first')
    if not BASELINE.is_file():
        raise FileNotFoundError(BASELINE)
    resume_path, reusable = load_reusable_controls(args.resume) if args.resume else (None, {})
    for condition, profile in reusable:
        print('REUSE_VERIFIED ' + condition + ' ' + profile, flush=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    for condition, offset in CASES:
        for profile in ('smooth-only', 'mimic-safe'):
            checked = subprocess.run(command(condition, offset, profile), cwd=ROOT, env=env,
                                     capture_output=True, text=True)
            if checked.returncode:
                raise RuntimeError('CPU preflight failed: ' + checked.stdout + checked.stderr)
    print('CPU_PREFLIGHT_OK cases=4 attempts_per_case=1 fresh_worker=true physical=true seed=202610280 offsets=0,9', flush=True)
    print('SCOPE old=smooth-only new=mimic-safe motion_settings=identical limits=unchanged original_accepted_3=retained', flush=True)
    if not args.run:
        print('PREPARED_ONLY GPU_started=false; add --run to execute four sequential controls', flush=True)
        return
    out = BASE / 'checks' / datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    print('OUTPUT ' + str(out), flush=True)
    results = {}
    receipt_path = out / 'progress.json'
    progress = {'scene_sha256': SCENE_SHA, 'runs': [], 'completed': False,
                'resumed_from': str(resume_path) if resume_path else None,
                'purpose': 'Privileged simulator expert actuation diagnosis; not autonomous VLA or generalization evaluation',
                'condition_note': 'Offsets0/9 failed/succeeded in the earlier warm batch; fresh-worker outcomes must be remeasured.',
                'causal_limit': 'Separate cold workers and recorded initial-state checks reduce reset-history confounding; hidden solver state is not directly certified.'}
    for condition, offset in CASES:
        results[condition] = {}
        for profile in ('smooth-only', 'mimic-safe'):
            if (condition, profile) in reusable:
                saved = reusable[(condition, profile)]
                results[condition][profile] = saved
                progress['runs'].append(dict(saved['source_entry'], reused=True))
                receipt_path.write_text(json.dumps(progress, indent=2) + '\n')
                print('CONTROL_REUSED ' + condition + ' ' + profile, flush=True)
                continue
            name = condition + '_' + profile
            log = out / (name + '.log')
            run = None
            print('CONTROL_BEGIN ' + name, flush=True)
            with log.open('w') as handle:
                process = subprocess.Popen(command(condition, offset, profile, run=True), cwd=ROOT,
                                           env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, bufsize=1)
                for line in process.stdout:
                    handle.write(line)
                    handle.flush()
                    if line.startswith('OUTPUT '):
                        run = Path(line[7:].strip())
                    if line.startswith(('OUTPUT ', 'COLLECT ATTEMPT', 'CAPSTONE_', 'COLLECTION_EXIT', 'Traceback', 'RuntimeError')):
                        print(line.rstrip(), flush=True)
                code = process.wait()
            progress['runs'].append({'condition': condition, 'profile': profile, 'offset': offset,
                                     'run': str(run) if run else None, 'log': str(log), 'exit_code': code})
            receipt_path.write_text(json.dumps(progress, indent=2) + '\n')
            if code or run is None:
                print('FAILURE_LOG_TAIL\n' + '\n'.join(log.read_text().splitlines()[-24:]), flush=True)
                raise RuntimeError('Control failed to execute; no efficacy conclusion. See ' + str(log))
            subprocess.run([sys.executable, str(ROOT / 'scripts/analyze_hgwind_batch.py'),
                            '--run-dir', str(run)], cwd=ROOT, env=env, check=True)
            audit = json.loads((run / 'analysis/batch_failure_audit.json').read_text())
            if audit['saved_episodes'] != 1 or audit['rows'][0]['attempt_index'] != offset:
                raise RuntimeError('Control episode identity/count mismatch')
            row = audit['rows'][0]
            metadata = json.loads((run / 'Trajectory/1s1p' / TASK_ID / row['episode'] / 'meta.json').read_text())
            receipt = metadata.get('capstone_gripper_actuation')
            if profile == 'mimic-safe':
                if not receipt or receipt['physical_indices'] != [7, 8] or receipt['actuated_indices'] != [7] or receipt['passive_indices'] != [8] or not receipt['live_readback']['verified'] or receipt['force_configuration_calls'] < 1:
                    raise RuntimeError('Missing/invalid live mimic actuation proof; inspect ' + str(run))
                if metadata['arm_execution']['driven_finger_dof_indices'] != [7]:
                    raise RuntimeError('Collector still commands the passive jaw')
            results[condition][profile] = {'run': run, 'row': row, 'receipt': receipt}
            print('CONTROL_OUTCOME ' + name + ' ' + row['category'], flush=True)
    report = dict(progress, completed=True,
                  pairs={name: summarize_pair(value['smooth-only'], value['mimic-safe'])
                         for name, value in results.items()},
                  conclusion='Bounded development controls completed. Read physical outcomes and initial-state checks before deciding whether to expand; this sample does not estimate a population success rate.')
    report_path = out / 'report.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    receipt_path.write_text(json.dumps(dict(progress, completed=True), indent=2) + '\n')
    print('MIMIC_ACTUATION_CHECKS_COMPLETED report=' + str(report_path), flush=True)
    for name, value in report['pairs'].items():
        print('PAIR ' + name + ' ' + json.dumps(value), flush=True)


if __name__ == '__main__':
    main()
