#!/usr/bin/env python3
"""Portable entry for the frozen, visually reviewed HGwind expert configuration."""
import argparse
from contextlib import contextmanager
import signal
from collections import Counter
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'data/derived/hgwind_current_baseline_20261027/repo'
TASK_ID = 'grasp_wind_driven_motion_1s1p_hgprgew'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def verify_bundle():
    manifest = json.loads((ROOT / 'bundle_manifest.json').read_text())
    for row in manifest['files']:
        path = ROOT / row['path']
        if digest(path) != row['sha256']:
            raise RuntimeError('Frozen bundle file changed: ' + row['path'])
    for row in manifest['links']:
        path = ROOT / row['path']
        if not path.is_symlink() or os.readlink(path) != row['target']:
            raise RuntimeError('Frozen bundle link changed: ' + row['path'])
    return manifest


def validate_allocation(job, seed, budget, target, smoke):
    allocations = {'local': 202610280, 'hongtao': 202710280,
                   'smoke_local': 20261027, 'smoke_hongtao': 20261027}
    if job not in allocations or seed != allocations[job]:
        raise ValueError('Job/seed differs from the registered host allocation')
    if smoke != job.startswith('smoke_') or budget < 1 or target < 1 or target > budget:
        raise ValueError('Invalid smoke/target/attempt allocation')
    if not smoke and (target != 50 or budget != 500):
        raise ValueError('This first production cohort requires 50 successes / 500 attempts per host')
    return 9 if smoke else 0


def episode_summary(out):
    rows = []
    task = out / 'Trajectory/1s1p' / TASK_ID
    for path in sorted(task.glob('*/meta.json')):
        if not path.parent.name.startswith(('traj_', 'failed_')):
            continue
        metadata = json.loads(path.read_text())
        eligible = (metadata.get('success') is True
                    and metadata.get('trajectory_validity', {}).get('passed') is True
                    and metadata.get('grasp_mode') == 'physical_finger_contact_only')
        if not (path.parent / 'replay.hdf5').is_file():
            raise RuntimeError('Incomplete saved episode: ' + str(path.parent))
        configuration = metadata.get('capstone_expert_control', {})
        if configuration.get('profile') != 'smooth_lift_v1':
            raise RuntimeError('Wrong expert profile in saved episode: ' + str(path.parent))
        rows.append({'episode': path.parent.name, 'attempt_index': metadata['attempt_index'],
                     'episode_seed': metadata.get('episode_seed'), 'success': eligible,
                     'frames': metadata['num_steps'], 'grasp_mode': metadata.get('grasp_mode')})
    indices = [row['attempt_index'] for row in rows]
    if len(set(indices)) != len(indices):
        raise RuntimeError('Duplicate attempt indices in production shard')
    return rows


def main():
    global ROOT, REPO
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lab', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--depth-m', type=float, default=0.)
    parser.add_argument('--job', choices=('local', 'hongtao', 'smoke_local', 'smoke_hongtao'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--target-successes', type=int, default=50)
    parser.add_argument('--max-attempts', type=int, default=500)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    ROOT = args.bundle.resolve(strict=True)
    REPO = ROOT / 'data/derived/hgwind_current_baseline_20261027/repo'
    if not 0 <= args.depth_m <= .002:
        parser.error('depth-m must lie in [0, .002]')
    offset = validate_allocation(args.job, args.seed, args.max_attempts, args.target_successes, args.smoke)
    verify_bundle()
    config = json.loads((ROOT / 'production_config.json').read_text())
    task = ROOT / config['task_relative']
    if digest(task) != config['task_sha256']:
        raise RuntimeError('Task version mismatch')
    lab = args.lab.resolve(strict=True)
    sim = (lab / '_isaac_sim').resolve(strict=True)
    if (lab / 'VERSION').read_text().strip() != '2.2.1' or not (sim / 'VERSION').read_text().startswith('4.5.0'):
        raise RuntimeError('Pinned Isaac Lab / Sim version mismatch')
    # All writable caches and outputs live in this independent bundle, including on hongtao/33.
    os.environ.update(CAPSTONE_ISAACLAB_PATH=str(lab), ISAACLAB_PATH=str(lab), ISAAC_SIM_ROOT=str(sim),
                      ISAACSIM_PATH=str(sim), ROBOPHYS_CAPSTONE_OFFLINE_TASK='1',
                      ROBOPHYS_TASK_CONTEXT=json.dumps(config['task_context']),
                      CAPSTONE_HGWIND_CONTROL_PROFILE='smooth-lift', PYTHONDONTWRITEBYTECODE='1',
                      OMNI_KIT_ACCEPT_EULA='YES')
    sys.path.insert(0, str(REPO / 'Scripts'))
    from core.task_resolver import resolve_task
    from core.task_plan import build_execution_plan
    from remote.dispatch import resolve_task_dependencies
    from pxr import Usd
    plan = build_execution_plan(resolve_task(str(task)).spec).as_dict()
    if len(plan['steps']) != 1 or plan['steps'][0]['skill'] != 'Grasp':
        raise RuntimeError('Unexpected task execution plan')
    stage = Usd.Stage.Open(str(task))
    if not stage or not stage.GetPrimAtPath('/Root/Dynamics/ball_y9bssvn').IsValid():
        raise RuntimeError('Expected spherical target missing')
    _, dependencies = resolve_task_dependencies(ROOT, [task])
    out = args.output.resolve()
    cache = out.parent / 'runtime_cache' / args.job
    command = [sys.executable, str(ROOT / 'scripts/sim/dispatch_hgwind_collection.py'),
               '--task', str(task), '--no-cloud', '--physics-check',
               '--episodes', str(args.target_successes), '--max_attempts', str(args.max_attempts),
               '--seed', str(args.seed), '--attempt_offset', str(offset), '--run_id', 'hgwind_v1_' + args.job,
               '--out_root', str(out / 'Trajectory'), '--grasp_mode', 'physical',
               '--catch-targeting-mode', 'auto', '--control_hz', '60', '--episode_s', '12',
               '--no-expert-config', '--keep_failed', '--preserve-failed-on-start',
               '--no-code-snapshot', '--no-visualization', '--headless', '--device', 'cuda:0',
               '--fast_exit_after_save', '--experience', 'isaacsim.exp.base.python.kit',
               '--kit_args=--portable-root ' + str(cache)
               + ' --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/enabled=false'
               + ' --/renderer/multiGpu/maxGpuCount=1 --/app/extensions/fsWatcherEnabled=false']
    command += ['--target-all', '1'] if args.smoke else ['--target_successes', str(args.target_successes)]
    if args.depth_m:
        command[1] = str(Path(__file__).with_name('run_hgwind_depth_dispatch.py'))
        os.environ.update(CAPSTONE_FROZEN_BUNDLE=str(ROOT), CAPSTONE_HGWIND_DEPTH_M=str(args.depth_m))
    print('PRODUCTION_DEPTH_BIAS requested_m=' + str(args.depth_m) + ' validated_improvement=false', flush=True)
    print(f'PRODUCTION_PREFLIGHT_OK job={args.job} target={args.target_successes} max_attempts={args.max_attempts} scene={config["task_sha256"]} assets={dependencies["files"]}', flush=True)
    print('COMMAND ' + shlex.join(command), flush=True)
    if not args.run:
        return subprocess.run(command + ['--dry-run'], cwd=REPO, env=os.environ).returncode
    if shutil.disk_usage(ROOT).free < 12 * 1024**3:
        raise RuntimeError('Need at least 12 GiB free before collection')
    # A single owner prevents two commands sharing the physical GPU/cache/output shard.
    lock = (ROOT / 'collection.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if out.exists():
        raise RuntimeError('Existing shard retained; do not silently repeat or overwrite: ' + str(out))
    out.mkdir(parents=True)
    atomic_json(out / 'task_context.json', config['task_context'])
    provenance = {'schema': 'capstone_hgwind_production_v1', 'job': args.job, 'seed': args.seed,
                  'attempt_offset': offset, 'attempt_budget': args.max_attempts,
                  'target_successes': args.target_successes, 'smoke_only': args.smoke,
                  'command': command, 'task_sha256': config['task_sha256'],
                  'scene_selection': {'task_path': str(task), 'sha256': config['task_sha256']},
                  'source_reviewed_run': config['source_reviewed_run'],
                  'expert_control_profile': 'smooth-lift', 'bundle_manifest_sha256': digest(ROOT / 'bundle_manifest.json'),
                  'lab': str(lab), 'sim': str(sim), 'dependencies': dependencies,
                  'purpose': 'expert development dataset collection; not VLA or held-out physics-generalization result',
                  'started_at': datetime.now().astimezone().isoformat(),
                  'overnight_entry_sha256': digest(Path(__file__)),
                  'experimental_capture_gate': False, 'experimental_mimic_adapter': False,
                  'configuration_note': 'Exact frozen reviewed control; output/cache paths and temporary keep-awake only'}
    provenance['depth_bias_m'] = args.depth_m
    if args.depth_m:
        provenance['configuration_note'] = 'Reviewed base control plus unvalidated collision-projected downward approach-goal bias; no truth/scene/force/success change'
        provenance['depth_worker_sha256'] = digest(Path(__file__).with_name('run_hgwind_depth_worker.py'))
    atomic_json(out / 'run.json', provenance)
    cache.mkdir(parents=True, exist_ok=True)
    for name in ('config', 'xdg', 'tmp'):
        (cache / name).mkdir(exist_ok=True)
    os.environ.update(OMNI_CONFIG_PATH=str(cache / 'config'), XDG_CACHE_HOME=str(cache / 'xdg'),
                      TMPDIR=str(cache / 'tmp'))
    code = subprocess.run(command, cwd=REPO, env=os.environ).returncode
    rows = episode_summary(out)
    successes = sum(row['success'] for row in rows)
    complete = code == 0 and (len(rows) == 1 and successes == 1 if args.smoke else successes >= args.target_successes)
    provenance.update(exit_code=code, completed=complete, successes=successes,
                      failures=len(rows)-successes, saved_attempts=len(rows),
                      ended_at=datetime.now().astimezone().isoformat(), episodes=rows)
    atomic_json(out / 'run.json', provenance)
    atomic_json(out / 'cohort_summary.json', provenance)
    # Freeze a transport receipt only after all writers have stopped.
    files = [{'path': p.relative_to(out).as_posix(), 'bytes': p.stat().st_size, 'sha256': digest(p)}
             for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'transport_manifest.json']
    atomic_json(out / 'transport_manifest.json', {'job': args.job, 'files': files})
    verify_bundle()
    print(f'PRODUCTION_FINISHED job={args.job} success={successes} failed={len(rows)-successes} complete={complete} output={out}', flush=True)
    return 0 if complete else (code or 2)


POWER_KEY = ['org.gnome.settings-daemon.plugins.power', 'sleep-inactive-ac-type']


def power_value():
    return subprocess.check_output(['gsettings', 'get', *POWER_KEY], text=True).strip()


@contextmanager
def keep_awake():
    previous = power_value()
    changed = previous != "'nothing'"
    if changed:
        subprocess.run(['dbus-run-session', '--', 'gsettings', 'set', *POWER_KEY, 'nothing'], check=True,
                       stdout=subprocess.DEVNULL)
        if power_value() != "'nothing'":
            raise RuntimeError('Could not disable automatic AC suspend for overnight collection')
    print('PRODUCTION_KEEP_AWAKE active=true previous=' + previous, flush=True)
    try:
        yield
    finally:
        if changed and power_value() == "'nothing'":
            subprocess.run(['dbus-run-session', '--', 'gsettings', 'set', *POWER_KEY, previous], check=True,
                           stdout=subprocess.DEVNULL)
        print('PRODUCTION_KEEP_AWAKE restored=true', flush=True)


if __name__ == '__main__':
    if '--power-check' in sys.argv:
        with keep_awake():
            print('PRODUCTION_POWER_CHECK_OK current=' + power_value(), flush=True)
    elif '--run' in sys.argv:
        with keep_awake():
            raise SystemExit(main())
    else:
        raise SystemExit(main())
