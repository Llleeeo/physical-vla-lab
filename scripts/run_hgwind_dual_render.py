#!/usr/bin/env python3
"""Two-host camera-v2 rendering with verified private-HF return to the main machine."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from capstone_vla.hf_archive import download_package, sha256
from capstone_vla.hgwind_cohort import require
from capstone_vla.hgwind_render_delivery import install_visual_download

OUT = ROOT / 'data/derived/hgwind_40_camera_v2_20261028'
MANIFEST = OUT / 'render_manifest.json'
SSH = ['ssh', '-F', str(ROOT / 'configs/hongtao_ssh.conf'), 'capstone-hongtao']
REMOTE = '/media/RoboPhys/DataCollection33/Capstone_HGwind_Worker_Yuan_Haoming/hgwind_render_v2_20261028'
REMOTE_OUT = REMOTE + '/data/derived/hgwind_40_camera_v2_20261028'
REMOTE_PYTHON = '/media/RoboPhys/DataCollection33/Capstone_HGwind_Worker_Yuan_Haoming/cpu_env/bin/python'


def ssh_json(path):
    code = 'from pathlib import Path; print(Path(' + repr(path) + ').read_text())'
    reply = subprocess.run(SSH + [shlex.join([REMOTE_PYTHON, '-c', code])],
                           text=True, capture_output=True, check=True, timeout=30)
    return json.loads(reply.stdout)


def sync_cloud(manifest):
    receipt = ssh_json(REMOTE_OUT + '/cloud_receipts.json')
    require(receipt['task_sha256'] == manifest['task_sha256']
            and receipt['cohort'] == manifest['archive_cohort'], 'Wrong remote camera revision')
    config = json.loads((ROOT / 'configs/hf_private_archive.json').read_text())
    rows = {r['episode']: r for r in manifest['rows'] if r['host'] == 'hongtao'}
    records = []
    for episode, entry in sorted(receipt['uploads'].items()):
        require(episode in rows, 'Unknown cloud episode')
        archive = entry['archive']
        expected = '/'.join(('episodes', archive['task_id'], manifest['archive_cohort'],
                             'hongtao', episode, 'rendered'))
        require(archive['prefix'] == entry['prefix'] == expected, 'Wrong cloud prefix')
        incoming = OUT / 'cloud_downloads' / (episode + '_' + entry['revision'][:12])
        print('HF_DOWNLOAD_BEGIN ' + episode, flush=True)
        if not incoming.exists():
            download_package(entry['prefix'], config, incoming, revision=entry['revision'])
        qa = install_visual_download(incoming, rows[episode], archive)
        records.append(dict(episode=episode, revision=entry['revision'], frames=qa['frames']))
        print('LOCAL_RENDER_INSTALLED ' + episode + ' hashes=verified', flush=True)
    (OUT / 'cloud_delivery.json').write_text(json.dumps(dict(rows=records), indent=2) + '\n')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--sync', action='store_true', help='Retry completed cloud deliveries without rendering')
    parser.add_argument('--limit-per-host', type=int, default=0, help='1 samples one trajectory on each GPU')
    args = parser.parse_args()
    require(args.limit_per_host >= 0, 'Negative limit')
    lock = (OUT / 'dual_render.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads(MANIFEST.read_text())
    if args.sync:
        sync_cloud(manifest)
        return 0
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    logs = OUT / 'dual_runs' / run_id
    logs.mkdir(parents=True)
    commands = {
        'local': [sys.executable, str(ROOT / 'scripts/render_hgwind_cohort.py'),
                  '--manifest', str(MANIFEST), '--host', 'local', '--limit', str(args.limit_per_host)],
        'hongtao': SSH + [shlex.join([REMOTE_PYTHON, REMOTE + '/scripts/hgwind_remote_render_job.py',
                        '--limit', str(args.limit_per_host), '--run-id', run_id] + (['--run'] if args.run else []))]}
    if args.run:
        commands['local'].append('--run')
        require((OUT / 'dual_preflight.json').exists(), 'Run CPU dual preflight first')
        prior = json.loads((OUT / 'dual_preflight.json').read_text())
        require(prior['manifest_sha256'] == sha256(MANIFEST) and prior['contracts_equal'],
                'Preflight does not match the current manifest')
    processes = {}
    def execute(host):
        print('BEGIN ' + host + ' log=' + str(logs / (host + '.log')), flush=True)
        with (logs / (host + '.log')).open('w') as stream:
            p = subprocess.Popen(commands[host], cwd=ROOT,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TERM='xterm'),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                start_new_session=True)
            processes[host] = p
            for line in p.stdout:
                stream.write(line)
                stream.flush()
                if line.startswith(('COHORT_', 'RENDER PLAN', 'SELECTED ', '[render]', 'HF_', 'RAW_SOURCE',
                                    'OUTPUT ', 'Traceback', 'RuntimeError', 'ValueError')):
                    print('[' + host + '] ' + line.rstrip(), flush=True)
            code = p.wait()
        print('EXIT ' + host + ' code=' + str(code), flush=True)
        return dict(host=host, exit_code=code)
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        futures = {host: pool.submit(execute, host) for host in commands}
        results = [f.result() for f in futures.values()]
    except BaseException:
        # Cancel only this run's identified remote process group.
        try:
            remote = ssh_json(REMOTE_OUT + '/remote_job.json')
            if remote['run_id'] == run_id and remote['status'] == 'running':
                code = ('import os,signal; from pathlib import Path; p=' + str(remote['pid'])
                        + '; a=(Path("/proc")/str(p)/"cmdline").read_bytes(); '
                        + 'assert b"hgwind_remote_render_job.py" in a; '
                        + 'assert os.getpgid(p)==' + str(remote['pgid']) + '; os.killpg(p,signal.SIGTERM)')
                subprocess.run(SSH + [shlex.join([REMOTE_PYTHON, '-c', code])], timeout=15)
        except Exception:
            print('REMOTE_STOP_UNCONFIRMED inspect remote_job.json before restarting', flush=True)
        for p in processes.values():
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        raise
    finally:
        pool.shutdown(wait=True)
    report = dict(results=results, manifest_sha256=sha256(MANIFEST), gpu_render_requested=args.run)
    if not all(r['exit_code'] == 0 for r in results):
        (logs / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        raise RuntimeError('One host failed; saved data retained. Inspect logs: ' + str(logs))
    local_contract = json.loads((OUT / 'render_contract.json').read_text())
    remote_contract = ssh_json(REMOTE_OUT + '/render_contract.json')
    require(local_contract == remote_contract, 'Two-host scene/camera contract mismatch')
    report['contracts_equal'] = True
    if not args.run:
        (OUT / 'dual_preflight.json').write_text(json.dumps(report, indent=2) + '\n')
        print('DUAL_CPU_PREFLIGHT_PASS allocation=local21,hongtao19 camera_revision=v2 gpu_started=false', flush=True)
        return 0
    report['cloud_deliveries'] = sync_cloud(manifest)
    command = [sys.executable, str(ROOT / 'scripts/render_hgwind_cohort.py'),
               '--manifest', str(MANIFEST), '--audit-only']
    subprocess.run(command, cwd=ROOT, check=True)
    report['visual_audit'] = json.loads((OUT / 'visual_audit.json').read_text())
    (logs / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    ready = report['visual_audit']['ready']
    print('DUAL_RENDER_AND_CLOUD_RETURN ready=' + str(ready) + '/40 report=' + str(logs / 'report.json'), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
