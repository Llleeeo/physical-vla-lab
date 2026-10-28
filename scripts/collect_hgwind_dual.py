#!/usr/bin/env python3
"""Run the frozen HGwind expert on two hosts and verify delivery to this machine."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'configs/hgwind_dual.json').read_text())
BASE = ROOT / 'data/derived/hgwind_production_20261027'
REPO_REL = Path('data/derived/hgwind_current_baseline_20261027/repo')
SSH = ['ssh', '-F', str(ROOT / 'configs/hongtao_ssh.conf'), 'capstone-hongtao']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def worker_command(host, *, smoke=False, run=False):
    remote = host == 'hongtao'
    bundle = Path(CONFIG['remote_bundle']) if remote else ROOT / CONFIG['bundle']
    python = CONFIG['remote_python'] if remote else str(ROOT / CONFIG['local_python'])
    job = ('smoke_' if smoke else '') + host
    command = [python, str(bundle / 'scripts/run_hgwind_production_worker.py'),
               '--lab', CONFIG['remote_lab'] if remote else CONFIG['local_lab'],
               '--job', job, '--seed', str(20261027 if smoke else CONFIG['seeds'][host]),
               '--target-successes', str(1 if smoke else CONFIG['successes_per_host']),
               '--max-attempts', str(1 if smoke else CONFIG['attempt_cap_per_host'])]
    if smoke:
        command.append('--smoke')
    if run:
        command.append('--run')
    return (SSH + ['exec env PYTHONDONTWRITEBYTECODE=1 ' + shlex.join(command)]) if remote else command


def execute(host, *, smoke=False, run=False):
    kind = 'smoke' if smoke else ('collect' if run else 'preflight')
    log_dir = BASE / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / (kind + '_' + host + '_' + str(time.time_ns()) + '.log')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    print(f'START {host} mode={kind} log={log}', flush=True)
    with log.open('w') as stream:
        process = subprocess.Popen(worker_command(host, smoke=smoke, run=run), cwd=ROOT, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            if line.startswith(('PRODUCTION_', 'COLLECTION_', 'EXPERT_', 'CAPSTONE_', 'CPU_', 'COLLECT ATTEMPT', 'Traceback', 'RuntimeError', 'ValueError')):
                print(f'[{host}] {line.rstrip()}', flush=True)
        code = process.wait()
    print(f'EXIT {host} code={code} log={log}', flush=True)
    return {'host': host, 'exit_code': code, 'log': str(log)}


def verify_transport(directory):
    directory = Path(directory)
    receipt = json.loads((directory / 'transport_manifest.json').read_text())
    for row in receipt['files']:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Unsafe transport path')
        path = directory / relative
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise RuntimeError('Transport hash mismatch: ' + str(path))
    expected = {row['path'] for row in receipt['files']} | {'transport_manifest.json'}
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    if actual != expected:
        raise RuntimeError('Delivery contains unlisted or missing files')
    return receipt


def synchronize(host, smoke=False):
    job = ('smoke_' if smoke else '') + host
    source = Path(CONFIG['remote_bundle'] if host == 'hongtao' else ROOT / CONFIG['bundle']) / REPO_REL / 'runs' / job
    destination_root = BASE / 'smoke_results' if smoke else ROOT / CONFIG['local_final']
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / host
    if destination.exists():
        verify_transport(destination)
        print(f'ALREADY_VERIFIED {host} {destination}', flush=True)
        return json.loads((destination / 'cohort_summary.json').read_text())
    incoming = destination_root / ('.incoming_' + host)
    incoming.mkdir(exist_ok=True)
    if host == 'hongtao':
        # No deletion flags: neither the old disk data nor remote output is removed.
        remote_shell = shlex.join(['ssh', '-F', str(ROOT / 'configs/hongtao_ssh.conf')])
        command = ['rsync', '-rt', '--partial', '--delay-updates', '-e', remote_shell,
                   'capstone-hongtao:' + str(source) + '/', str(incoming) + '/']
        subprocess.run(command, cwd=ROOT, check=True)
    else:
        shutil.copytree(source, incoming, dirs_exist_ok=True)
    receipt = verify_transport(incoming)
    incoming.rename(destination)
    print(f'DELIVERY_VERIFIED host={host} files={len(receipt["files"])} destination={destination}', flush=True)
    return json.loads((destination / 'cohort_summary.json').read_text())


def validate_smoke():
    for host in ('local', 'hongtao'):
        path = BASE / 'smoke_results' / host
        verify_transport(path)
        summary = json.loads((path / 'cohort_summary.json').read_text())
        if summary['exit_code'] != 0 or summary['saved_attempts'] != 1 or summary['successes'] != 1 or summary['expert_control_profile'] != 'smooth-lift':
            raise RuntimeError('Missing verified physical smoke for ' + host)
        current = digest(ROOT / CONFIG['bundle'] / 'bundle_manifest.json')
        if summary['bundle_manifest_sha256'] != current:
            raise RuntimeError('Smoke used a different frozen bundle')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--smoke', action='store_true', help='one saved physical attempt on each host')
    modes.add_argument('--run', action='store_true', help='50 successes / 200 attempt cap on each host')
    modes.add_argument('--sync', action='store_true', help='retry delivery of finished production shards only')
    args = parser.parse_args()
    if (args.run or args.sync) and (BASE / 'retired_batch.json').exists():
        raise RuntimeError('This stopped cohort was retired and its raw trajectories removed. Use scripts/run_hgwind_transition_v2.py for the separate improvement experiment.')
    if args.run:
        validate_smoke()
    if args.sync:
        summaries = [synchronize(host) for host in ('local', 'hongtao')]
        results = []
    else:
        summaries = []
        results = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {host: pool.submit(execute, host, smoke=args.smoke, run=args.run or args.smoke)
                       for host in ('local', 'hongtao')}
            for host, future in futures.items():
                result = future.result()
                results.append(result)
                if args.run or args.smoke:
                    try:
                        summaries.append(synchronize(host, args.smoke))
                    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
                        print(f'DELIVERY_PENDING host={host} reason={exc}; remote data retained', flush=True)
                        result['delivery_error'] = str(exc)
    report = {'mode': 'smoke' if args.smoke else ('production' if args.run or args.sync else 'CPU preflight'),
              'results': results, 'summaries': summaries,
              'successes': sum(s['successes'] for s in summaries),
              'failures': sum(s['failures'] for s in summaries),
              'expected_successes': 100 if args.run or args.sync else None,
              'remote_output_retained': True,
              'scope': 'expert training/development pool; held-out evaluation not yet defined'}
    if not (args.run or args.smoke or args.sync):
        passed = all(r['exit_code'] == 0 for r in results)
    else:
        passed = len(summaries) == 2 and all(s['completed'] for s in summaries)
    report['completed'] = passed
    BASE.mkdir(parents=True, exist_ok=True)
    filename = 'smoke_report.json' if args.smoke else ('production_report.json' if args.run or args.sync else 'preflight_report.json')
    (BASE / filename).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f'DUAL_FINISHED completed={passed} successes={report["successes"]} failures={report["failures"]} report={BASE / filename}', flush=True)
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
