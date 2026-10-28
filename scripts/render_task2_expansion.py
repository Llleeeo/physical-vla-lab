#!/usr/bin/env python3
"""Stage immutable raw cohort copies and render 60Hz images using existing tooling."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import h5py
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'data/derived/task2_expansion_inventory_20261025'
REPO=ROOT/'data/derived/hf_pilot_render_repo'
WORK=ROOT/'data/derived/task2_expansion_render_work/Trajectory'
LAUNCHER=Path('/home/robophys/RoboPhys/isaaclab/isaaclab.sh')

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--limit',type=int,default=0,help='optional max pending trajectories for a short render')
    args=parser.parse_args()
    if args.limit<0:raise ValueError('limit must be nonnegative')
    manifest=json.loads((PLAN/'split_manifest.json').read_text())
    raw_report=json.loads((PLAN/'raw_cohort_report.json').read_text())
    if raw_report['status']!='RAW_COHORT_VERIFIED' or raw_report['manifest_sha256']!=digest(PLAN/'split_manifest.json'):
        raise ValueError('raw checks or manifest binding missing')
    taskid=manifest['task_id'];task={'task_id':taskid,'scene_family':'1s1p','usd':'Tasks/1s1p/'+taskid+'.usd'}
    script=REPO/'Scripts/tools/render_saved_trajectories.py'
    spec=importlib.util.spec_from_file_location('capstone_cohort_renderer',script)
    renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
    source_records={row['episode']:row for row in raw_report['episodes']}
    old_root=ROOT/'data/derived/hf_pilot_render_60hz_work/Trajectory'
    old_episode=old_root/'1s1p'/taskid/'traj_00001'
    def complete(root,replay):
        return renderer._render_complete(root,task,replay,camera_data=True,camera_fps=60.,render_quality='balanced')
    reuse_001=complete(old_root,old_episode/'replay.hdf5')
    if reuse_001:
        if digest(old_episode/'replay.hdf5')!=manifest['selected']['train'][0]['replay_sha256']:
            raise ValueError('existing visual replay does not match frozen source')
    rows=[];names=[];new_frames=0
    for split,selected in manifest['selected'].items():
        for item in selected:
            source=Path(source_records[item['episode']]['path'])
            target=WORK/'1s1p'/taskid/item['episode'];target.mkdir(parents=True,exist_ok=True)
            for file in ['replay.hdf5','meta.json']:
                dest=target/file
                if not dest.exists():shutil.copy2(source/file,dest)
            if digest(target/'replay.hdf5')!=item['replay_sha256']:raise ValueError('staged replay differs from immutable raw source')
            meta=json.loads((target/'meta.json').read_text())
            if meta['task']['id']!=taskid or meta['provenance']['source_task']['sha256']!=item['task_sha256']:
                raise ValueError('staged task identity changed')
            reused=item['episode']=='traj_00001' and reuse_001
            episode=old_episode if reused else target
            if not reused:names.append(item['episode']);new_frames+=item['num_steps']
            rows.append({'split':split,'episode':item['episode'],'source_replay':str(source/'replay.hdf5'),
                  'render_episode':str(episode),'reused_existing_visuals':reused,'frames':item['num_steps'],
                  'expected_camera_archive':str(episode/'video/rendered.hdf5')})
    record={'schema':'capstone_task2_cohort_render_sources_v1','manifest_sha256':digest(PLAN/'split_manifest.json'),
       'task_usd':str(REPO/task['usd']),'task_usd_sha256':digest(REPO/task['usd']),
       'camera_fps':60,'quality':'balanced','sources':rows,'training_ready':False}
    record_path=PLAN/'render_sources.json';record_path.write_text(json.dumps(record,indent=2)+'\n')
    ratio=(old_episode/'video/rendered.hdf5').stat().st_size/232
    print('COHORT_STAGED episodes=56 reused_visuals='+str(int(reuse_001))+' render_candidates='+str(len(names)),flush=True)
    print('APPROX_NEW_CAMERA_ARCHIVE_GiB='+format(ratio*new_frames/2**30,'.1f')+' (rough source-episode ratio; actual compression varies)',flush=True)
    print('RENDER_OUTPUT',WORK,flush=True)
    dependency_resolver=renderer.resolve_task_dependencies
    renderer.resolve_task_dependencies=lambda _root,tasks:dependency_resolver(ROOT,tasks)
    arguments=[str(script),'--launcher',str(LAUNCHER),'--out-root',str(WORK),'--task-id',taskid,
       '--only-missing','--device','cuda:0','--render-quality','balanced','--camera-fps','60']
    for name in names:arguments+=['--trajectory-name',name]
    if args.limit:arguments+=['--max-trajectories',str(args.limit)]
    if not args.run:arguments+=['--dry-run']
    if names:
        sys.argv=arguments
        status=renderer.main()
        if status:return status
    if not args.run:return 0
    ready=0
    for row in rows:
        episode=Path(row['render_episode']);root=old_root if row['reused_existing_visuals'] else WORK
        ok=complete(root,episode/'replay.hdf5')
        if ok:
            with h5py.File(episode/'video/rendered.hdf5','r') as f:
                if not np.array_equal(f['source_frame_index'][:],np.arange(row['frames'])):
                    raise ValueError(row['episode']+': incomplete 60Hz frame coverage')
                with h5py.File(episode/'replay.hdf5','r') as replay:
                    if not np.allclose(f['timestamp_s'][:],replay['timestamp'][:],rtol=0,atol=1e-7):
                        raise ValueError(row['episode']+': timestamp mismatch')
            ready+=1
        row['visual_artifacts_complete']=ok
    record['visuals_ready_count']=ready
    record_path.write_text(json.dumps(record,indent=2)+'\n')
    print(('COHORT_RENDER_COMPLETE' if ready==56 else 'COHORT_RENDER_PARTIAL')+' ready='+str(ready)+'/56 manifest='+str(record_path),flush=True)
    return 0 if ready==56 or args.limit else 2

if __name__=='__main__':raise SystemExit(main())
