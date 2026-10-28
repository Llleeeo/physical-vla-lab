#!/usr/bin/env python3
"""Download and verify the frozen Task2 cohort; never renders or trains."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import shutil
import sys
import h5py
import numpy as np
from huggingface_hub import hf_hub_download
ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'data/derived/task2_expansion_inventory_20261025'
RAW=ROOT/'data/raw/task2_expansion_20261025'
sys.path.insert(0,str(ROOT/'data/derived/hf_pilot_render_repo/Scripts'))
from dataset.replay_io import read_replay_scene,validate_replay_scene_consistency

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def inspect_replay(path,row):
    if path.stat().st_size!=row['replay_bytes'] or digest(path)!=row['replay_sha256']:
        raise ValueError('replay bytes/hash do not match cloud metadata')
    with h5py.File(path,'r') as f:
        a=f['actions'][:];q=f['obs/joint_pos'][:];ts=f['timestamp'][:];n=len(a)
        if a.shape!=(n,9) or q.shape!=(n,9) or n!=row['num_steps'] or ts.shape!=(n,):
            raise ValueError('unsupported joint/sample layout')
        if not np.isfinite(a).all() or not np.isfinite(q).all() or not np.isfinite(ts).all():
            raise ValueError('nonfinite robot data')
        if not np.allclose(ts,np.arange(n)/60,rtol=0,atol=1e-6):raise ValueError('control clock differs from 60 Hz contract')
        if not np.allclose(a[:,7],a[:,8],rtol=0,atol=1e-6):raise ValueError('adapter needs asymmetric gripper support')
        if not np.all(np.isclose(a[:,7],0,atol=1e-5)|np.isclose(a[:,7],.04,atol=1e-5)):
            raise ValueError('adapter needs additional gripper levels')
        scene=read_replay_scene(f)
        obs={key:f['obs/'+key][:] for key in ['object_pos','object_quat','object_vel','object_ang_vel']}
        validate_replay_scene_consistency(scene,obs,require_verified=True)
        state=scene['robot_root']['state']
        if state.shape!=(n,7) or not np.allclose(state,state[:1],rtol=0,atol=1e-6):
            raise ValueError('adapter needs moving robot base support')
        result={'frames':n,'action_array_sha256_measured':hashlib.sha256(np.asarray(a,dtype='<f4').tobytes()).hexdigest(),
          'coarse_array_sha256_measured':hashlib.sha256(np.rint(a/.001).astype('<i8').tobytes()).hexdigest(),
          'gripper_transition_count':int(np.count_nonzero(np.diff(a[:,7]>.02))),
          'initial_joint_pos':q[0].tolist(),'initial_object_pos':obs['object_pos'][0].tolist(),
          'initial_object_vel':obs['object_vel'][0].tolist(),'raw_replay_checked':True,
          'images_checked':False,'independent_task_success_checked':False}
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    manifest=json.loads((PLAN/'split_manifest.json').read_text())
    selected=[(split,row) for split,rows in manifest['selected'].items() for row in rows]
    if len(selected)!=56:raise ValueError('frozen cohort size changed')
    if len({row['episode'] for _,row in selected})!=56:raise ValueError('duplicate episode across split')
    for split,row in selected:
        expected=PurePosixPath('Trajectory/1s1p')/manifest['task_id']/row['episode']/'replay.hdf5'
        if PurePosixPath(row['remote_replay'])!=expected:raise ValueError('unexpected remote artifact path')
    print('CPU_PLAN_OK train=40 validation=8 test_reserved=8 replay_MiB='+format(manifest['replay_bytes_selected']/2**20,'.2f'),flush=True)
    print('TARGET',RAW,flush=True)
    if not args.run:print('NO_DOWNLOAD_STARTED; add --run',flush=True);return
    RAW.mkdir(parents=True,exist_ok=True)
    results=[]
    for index,(split,row) in enumerate(selected,1):
        target=RAW/row['remote_replay']
        try:
            if not target.exists():
                downloaded=hf_hub_download(manifest['repo_id'],row['remote_replay'],repo_type='dataset',
                    revision=manifest['revision'],local_dir=RAW)
                if Path(downloaded).resolve()!=target.resolve():raise ValueError('unexpected download location')
            record=inspect_replay(target,row)
            source_meta=PLAN/'metadata'/(row['episode']+'.json')
            dest=target.parent/'meta.json'
            if dest.exists():
                if dest.read_bytes()!=source_meta.read_bytes():raise ValueError('existing source metadata differs; refusing overwrite')
            else:shutil.copyfile(source_meta,dest)
            record.update(split=split,episode=row['episode'],path=str(target.parent),status='replay_verified')
            results.append(record)
            print('DOWNLOAD_VERIFY '+str(index)+'/56 '+split+' '+row['episode']+' frames='+str(record['frames']),flush=True)
        except Exception as exc:
            results.append({'split':split,'episode':row['episode'],'status':'blocked','reason':str(exc)})
            print('BLOCKED '+row['episode']+' '+str(exc),flush=True)
    measured={};duplicates=[]
    for row in results:
        if row['status']!='replay_verified':continue
        signature=row['coarse_array_sha256_measured']
        if signature in measured:duplicates.append([measured[signature],row['episode']])
        else:measured[signature]=row['episode']
    blocked=[row for row in results if row['status']=='blocked']
    report={'status':'RAW_COHORT_VERIFIED' if not blocked and not duplicates else 'RAW_COHORT_HAS_BLOCKERS',
       'repo_id':manifest['repo_id'],'revision':manifest['revision'],'manifest_sha256':digest(PLAN/'split_manifest.json'),
       'episodes':results,'blocked_count':len(blocked),'measured_coarse_duplicate_pairs':duplicates,
       'training_ready':False,'next':'prepare 60Hz RGB rendering in isolated working copy; preserve train/validation/test assignments',
       'heldout_evaluation_executed':False,'source_success':'collector-declared; no independent success verification here'}
    (PLAN/'raw_cohort_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['status']+' report='+str(PLAN/'raw_cohort_report.json'),flush=True)
    if blocked or duplicates:raise SystemExit(2)

if __name__=='__main__':main()
