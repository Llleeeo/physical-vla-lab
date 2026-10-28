#!/usr/bin/env python3
"""Prepare or run the capstone's local Isaac Franka interface checks."""
import argparse
from datetime import datetime
import json
import hashlib
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config',type=Path,default=ROOT/'configs/franka_action_interface.json')
parser.add_argument('--output',type=Path)
parser.add_argument('--smoke',action='store_true')
parser.add_argument('--gui',action='store_true')
parser.add_argument('--device',default='cuda:0')
parser.add_argument('--run',action='store_true',help='start Isaac; default validates and prints only')
args=parser.parse_args()
cfg=json.loads(args.config.read_text())
lab=Path(cfg['isaaclab_root'])
for path in [Path(cfg['robot_usd']),Path(cfg['urdf']),lab/'isaaclab.sh']:
    if not path.is_file():parser.error(f'missing local dependency: {path}')
stamp=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S')
out=(args.output or ROOT/'data/derived/franka_interface_checks'/stamp).resolve()
if out.exists() and any(out.iterdir()):parser.error(f'output already contains files: {out}')
command=[str(lab/'isaaclab.sh'),'-p',str(ROOT/'scripts/sim/check_franka_interface.py'),
         '--config',str(args.config.resolve()),'--output',str(out),'--device',args.device,
         '--experience','isaacsim.exp.base.python.kit',
         '--kit_args=--portable-root ' + str(ROOT/'data/derived/.cache/isaacsim/linux')
         + ' --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/enabled=false'
         + ' --/renderer/multiGpu/maxGpuCount=1 --/app/extensions/fsWatcherEnabled=false']
if args.smoke:command.append('--smoke')
if not args.gui:command.append('--headless')
print('OUTPUT',out,flush=True)
print('COMMAND',shlex.join(command),flush=True)
if not args.run:
    print('DRY_RUN_ONLY: add --run to execute');raise SystemExit(0)
out.mkdir(parents=True,exist_ok=True)
env=dict(os.environ);env['TERM']='xterm';env.setdefault('OMNI_KIT_ACCEPT_EULA','YES')
cache_root=ROOT/'data/derived/.cache/isaacsim/linux'
config_root=cache_root/'config'
if (config_root/'omniverse.toml').is_file():env['OMNI_CONFIG_PATH']=str(config_root)
env.setdefault('XDG_CACHE_HOME',str(cache_root/'xdg'))
env['PYTHONPATH']=':'.join([str(ROOT/'src'),*[str(p) for p in (lab/'source').iterdir() if p.is_dir()],env.get('PYTHONPATH','')])
with (out/'run.log').open('w') as log:
    proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                          text=True,errors='replace',start_new_session=True)
    try:
        for line in proc.stdout:
            log.write(line);log.flush()
            if 'INTERFACE_' in line:print(line.rstrip(),flush=True)
        code=proc.wait()
    except KeyboardInterrupt:
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        raise SystemExit(130)
fingerprint_paths=[args.config.resolve(),ROOT/'src/capstone_vla/franka.py',
                   ROOT/'scripts/sim/check_franka_interface.py',Path(__file__).resolve()]
provenance={'schema':'capstone_franka_interface_run_provenance_v1','command':command,
            'process_exit_code':code,'device':args.device,
            'isaaclab_version':(lab/'VERSION').read_text().strip(),
            'isaacsim_version':(lab/'_isaac_sim/VERSION').read_text().strip(),
            'code_and_config_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in fingerprint_paths}}
(out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
if code:
    print('FAILED; log:',out/'run.log')
    print(''.join((out/'run.log').read_text(errors='replace').splitlines(keepends=True)[-25:]))
raise SystemExit(code)
