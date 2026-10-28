#!/usr/bin/env python3
"""Remote render then private-HF archive. All new bulk files stay on the 33 disk."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from capstone_vla.hf_archive import pack_episode, upload_package, sha256
from capstone_vla.hgwind_cohort import require

OUT = ROOT / 'data/derived/hgwind_40_camera_v2_20261028'
MANIFEST = OUT / 'render_manifest.json'
LAB = '/media/RoboPhys/DataCollection33/RoboPhys/isaaclab'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--run-id', default='preflight')
    args = parser.parse_args()
    lock = (OUT / 'remote_job.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if os.getpgrp() != os.getpid():
        os.setsid()
    registry = OUT / 'remote_job.json'
    registry.write_text(json.dumps(dict(pid=os.getpid(), pgid=os.getpgrp(), run_id=args.run_id,
                                       status='running' if args.run else 'preflight')) + '\n')
    child = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        command = [sys.executable, str(ROOT / 'scripts/render_hgwind_cohort.py'),
                   '--manifest', str(MANIFEST), '--lab', LAB, '--host', 'hongtao',
                   '--limit', str(args.limit)]
        if args.run:
            command.append('--run')
        child = subprocess.Popen(command, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TERM='xterm'))
        code = child.wait()
        if code or not args.run:
            return code
        # Activate only this project's existing remote credentials.
        private = Path('/home/RoboPhys/.config/capstone_hf_archive')
        require(private.joinpath('token').is_file(), 'Personal HF login missing')
        os.environ.update(HF_HOME=str(private), HF_TOKEN_PATH=str(private / 'token'),
                          HF_HUB_CACHE=str(OUT / 'hf_cache/hub'), HF_XET_CACHE=str(OUT / 'hf_cache/xet'))
        os.environ.pop('HF_TOKEN', None)
        os.environ.pop('HUGGING_FACE_HUB_TOKEN', None)
        from huggingface_hub import HfApi
        config = json.loads((ROOT / 'configs/hf_private_archive.json').read_text())
        require(HfApi().whoami().get('name') == config['owner'], 'Wrong project HF account')
        manifest = json.loads(MANIFEST.read_text())
        report = json.loads((OUT / 'visual_audit_hongtao.json').read_text())
        require(report['rows'], 'No completed rendered episodes')
        receipt_path = OUT / 'cloud_receipts.json'
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else dict(
            schema='capstone.hgwind_render_cloud.v1', cohort=manifest['archive_cohort'],
            task_sha256=manifest['task_sha256'], uploads={})
        require(receipt['task_sha256'] == manifest['task_sha256'], 'Cloud receipt scene changed')
        for row in report['rows']:
            require(row['suspicious_blank_count'] == 0, 'Blank sampled render requires inspection')
            package, archive = pack_episode(row['path'], cohort=manifest['archive_cohort'],
                                             host='hongtao', stage='rendered', output=OUT / 'packages')
            prior = receipt['uploads'].get(row['episode'])
            if prior:
                require(prior['archive'] == archive, 'Previously uploaded render changed')
                continue
            print('HF_UPLOAD_BEGIN ' + row['episode'] + ' bytes=' + str(archive['payload']['bytes']), flush=True)
            result = upload_package(package, config)
            receipt['uploads'][row['episode']] = dict(**result, archive=archive)
            tmp = receipt_path.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(receipt, indent=2) + '\n')
            tmp.replace(receipt_path)
            print('HF_UPLOAD_VERIFIED ' + row['episode'] + ' revision=' + result['revision'], flush=True)
        return 0
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        registry.write_text(json.dumps(dict(pid=os.getpid(), pgid=os.getpgrp(), run_id=args.run_id,
                                           status='finished')) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
