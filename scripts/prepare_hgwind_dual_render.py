#!/usr/bin/env python3
"""Deploy only portable rendering adapters and the accepted camera-only scene to hongtao."""
import copy
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REMOTE_BASE = '/media/RoboPhys/DataCollection33/Capstone_HGwind_Worker_Yuan_Haoming'
REMOTE = REMOTE_BASE + '/hgwind_render_v2_20261028'
REMOTE_PYTHON = REMOTE_BASE + '/cpu_env/bin/python'
SSH = ['ssh', '-F', str(ROOT / 'configs/hongtao_ssh.conf'), 'capstone-hongtao']
OUT_REL = Path('data/derived/hgwind_40_camera_v2_20261028')


def remote_python(code):
    subprocess.run(SSH + [shlex.join([REMOTE_PYTHON, '-c', code])], check=True)


def main():
    out = ROOT / OUT_REL
    manifest_path = out / 'render_manifest.json'
    m = json.loads(manifest_path.read_text())
    contract = json.loads((out / 'render_contract.json').read_text())
    assert contract['task_sha256'] == m['task_sha256']
    m['expected_render_contract'] = contract
    manifest_path.write_text(json.dumps(m, indent=2) + '\n')
    remote = copy.deepcopy(m)
    remote.update(bundle=REMOTE_BASE + '/production_v1',
                  runtime_root=REMOTE_BASE + '/production_v1/data/derived/hgwind_current_baseline_20261027/repo/runs/' + m['cohort'] + '/runtime_cache/hongtao',
                  dependency_root=REMOTE_BASE,
                  source_root=REMOTE_BASE + '/production_v1/data/derived/hgwind_current_baseline_20261027/repo/runs/' + m['cohort'],
                  task_usd=str(Path(REMOTE) / Path(m['task_usd']).relative_to(ROOT)),
                  rows=[r for r in remote['rows'] if r['host'] == 'hongtao'])
    remote['task_context'] = remote['source_root'] + '/hongtao/task_context.json'
    for row in remote['rows']:
        row['source'] = remote['source_root'] + '/hongtao/Trajectory/1s1p/grasp_wind_driven_motion_1s1p_hgprgew/' + row['episode']
        row['staged'] = str(Path(REMOTE) / OUT_REL / 'render_work/hongtao/Trajectory/1s1p/grasp_wind_driven_motion_1s1p_hgprgew' / row['episode'])
    remote['episode_count'] = len(remote['rows'])
    remote['total_frames'] = sum(r['frames'] for r in remote['rows'])
    (out / 'hongtao_manifest.json').write_text(json.dumps(remote, indent=2) + '\n')
    files = ['scripts/render_hgwind_cohort.py', 'scripts/hgwind_remote_render_job.py',
             'src/capstone_vla/__init__.py', 'src/capstone_vla/hgwind_cohort.py',
             'src/capstone_vla/hf_archive.py', 'configs/hf_private_archive.json',
             str(Path(m['task_usd']).relative_to(ROOT)),
             str(OUT_REL / 'hongtao_manifest.json')]
    remote_python('from pathlib import Path; Path(' + repr(REMOTE) + ').mkdir(parents=True, exist_ok=True)')
    subprocess.run(['rsync', '-rt', '--relative', '-e', shlex.join(SSH[:3]),
                    *files, 'capstone-hongtao:' + REMOTE + '/'], cwd=ROOT, check=True)
    code = '''
import json,sys,shutil
from pathlib import Path
root=Path(REMOTE_ROOT)
sys.path.insert(0,str(root/'src'))
from capstone_vla.hgwind_cohort import stage_episode,require
from capstone_vla.hf_archive import sha256
out=root/'data/derived/hgwind_40_camera_v2_20261028'
m=json.loads((out/'hongtao_manifest.json').read_text())
task=Path(m['task_usd']);scene=task.parents[2]
variant=Path(m['bundle'])/'data/derived/hgwind_current_baseline_20261027/repo/SceneVariants/df11a22a0656b6e2'
for p in variant.iterdir():
 if p.is_dir() and p.name!='Tasks':
  link=scene/p.name
  if link.exists():require(link.resolve()==p.resolve(),'Asset link changed')
  else:link.symlink_to(p.resolve(),target_is_directory=True)
require(sha256(task)==m['task_sha256'],'Transferred scene hash mismatch')
require(sha256(Path(m['bundle'])/'bundle_manifest.json')==m['bundle_manifest_sha256'],'Frozen bundle mismatch')
for row in m['rows']:stage_episode(row['source'],row['staged'],row['files'])
target=out/'render_manifest.json'
if target.exists():require(json.loads(target.read_text())==m,'Remote manifest already differs')
else:target.write_text(json.dumps(m,indent=2)+'\\n')
free=shutil.disk_usage(root).free
# Both rendered files and immutable TAR packages are retained.
estimated=m['total_frames']*5_000_000*2+15*1024**3
require(free>estimated,'Insufficient 33-disk space for renders plus archive copies')
print('REMOTE_STAGED episodes='+str(len(m['rows']))+' frames='+str(m['total_frames'])+' free_GiB='+format(free/1024**3,'.1f'))
'''
    remote_python(code.replace('REMOTE_ROOT', repr(REMOTE)))
    print('DUAL_DEPLOYED CPU preflight required; GPU not started', flush=True)


if __name__ == '__main__':
    main()
