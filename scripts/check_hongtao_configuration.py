#!/usr/bin/env python3
"""Recheck the prepared hongtao snapshot with CPU only; never submits work."""
import json
from pathlib import Path
import shlex
import subprocess
ROOT=Path(__file__).resolve().parents[1]
config=json.loads((ROOT/'configs/hongtao_render.json').read_text())
if config['automatic_submission_enabled'] or config['assigned_trajectories']:
    raise RuntimeError('This checker requires task submission disabled')
root=str(Path(config['workspace']).parent)
check="""from pathlib import Path
import hashlib,json
r=Path(%r)
for row in json.loads((r/'files_manifest.json').read_text()):
 p=r/row['path']
 if hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:raise RuntimeError('Snapshot changed: '+str(p))
print('SNAPSHOT_HASHES_OK')
""" % root
ssh=['ssh','-F',str(ROOT/'configs/hongtao_ssh.conf'),'capstone-hongtao']
subprocess.run(ssh+['python3 -c '+shlex.quote(check)],check=True)
command='PYTHONDONTWRITEBYTECODE=1 '+shlex.quote(config['cpu_python'])+' '+shlex.quote(root+'/preflight.py')
subprocess.run(ssh+[command],check=True)
print('HONGTAO_CONFIGURATION_CHECK_PASS render_jobs_submitted=0')
