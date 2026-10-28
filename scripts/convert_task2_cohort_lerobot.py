#!/usr/bin/env python3
"""Export frozen train/validation cohorts separately; never exports held-out test."""
import argparse
from functools import partial
import importlib.metadata
import json
from pathlib import Path
import sys
import hashlib
import h5py
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'third_party/DynamicVLA'))
from capstone_vla.franka import FrankaCartesianAdapter
from capstone_vla import local_cache
import utils
utils.memcached=local_cache;sys.modules['utils.memcached']=local_cache
import utils.datasets as upstream
import lerobot.datasets.lerobot_dataset as lr
PLAN=ROOT/'data/derived/task2_expansion_inventory_20261025'
CAMERAS=['opst_cam','wrist_cam']


def features():
    result={'observation.state':{'dtype':'float32','shape':(6,),'names':['x','y','z','rx','ry','rz']},
            'action':{'dtype':'float32','shape':(7,),'names':['x','y','z','rx','ry','rz','gripper']}}
    result.update({'observation.images.'+name:{'dtype':'video','shape':(480,640,3),'names':['height','width','channel']} for name in CAMERAS})
    return result

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def convert_split(split,rows,output,adapter,manifest_hash):
    target=output/split;repo_id='capstone-local/task2-'+split
    identity={'schema':'capstone_task2_cohort_export_v1','split':split,'manifest_sha256':manifest_hash,
       'episode_order':[row['episode'] for row in rows],'source_render_episodes':[row['render_episode'] for row in rows],
       'fps':60,'urdf_sha256':adapter.urdf_sha256,'video_codec':'h264','crf':30,'gop':2,
       'instruction':'verbatim original text via capstone_instruction adapter','heldout_evaluation':False}
    if target.exists():
        marker=target/'capstone_cohort.json'
        if not marker.exists() or json.loads(marker.read_text())!=identity:
            raise RuntimeError('Existing export not bound to this exact cohort; preserve it and choose a new --output')
        info=json.loads((target/'meta/info.json').read_text());completed=info['total_episodes']
        if completed==0:raise RuntimeError('Incomplete initial export; preserve directory and select a new --output')
        dataset=lr.LeRobotDataset(repo_id,root=target,episodes=list(range(completed)),download_videos=False,video_backend='torchcodec')
        dataset.start_image_writer(num_threads=4)
        for i in range(completed):
            if dataset.meta.episodes[i]['length']!=rows[i]['frames']:raise ValueError('completed episode length differs from cohort')
    else:
        dataset=lr.LeRobotDataset.create(repo_id=repo_id,root=target,fps=60,features=features(),
                       robot_type='franka',use_videos=True,image_writer_threads=4)
        (target/'capstone_cohort.json').write_text(json.dumps(identity,indent=2)+'\n')
        completed=0
    try:
        for i,row in enumerate(rows[completed:],completed):
            ep=Path(row['render_episode']);source=Path(row['source_replay'])
            meta=json.loads(source.with_name('meta.json').read_text())
            text=meta['task'].get('instruction_cn') or meta['task']['instruction']
            task=json.dumps({'task':'pick','objects':['moving object'],'capstone_instruction':text},ensure_ascii=False)
            with h5py.File(source,'r') as replay,h5py.File(ep/'video/rendered.hdf5','r') as camera:
                q=replay['obs/joint_pos'][:];a=replay['actions'][:];n=len(a)
                if n!=row['frames'] or not np.array_equal(camera['source_frame_index'][:],np.arange(n)):
                    raise ValueError('frame coverage differs from cohort')
                if not np.allclose(camera['timestamp_s'][:],replay['timestamp'][:],rtol=0,atol=1e-7):raise ValueError('timestamp mismatch')
                if not np.isfinite(q).all() or not np.isfinite(a).all():raise ValueError('nonfinite action/state')
                if not np.allclose(a[:,7],a[:,8],atol=1e-6) or not np.all(np.isclose(a[:,7],0,atol=1e-5)|np.isclose(a[:,7],.04,atol=1e-5)):
                    raise ValueError('unsupported gripper contract')
                state=np.stack([adapter.state(v[:7]) for v in q]).astype(np.float32)
                action=np.stack([np.r_[adapter.state(v[:7]),1. if v[7]>.02 else -1.] for v in a]).astype(np.float32)
                for frame in range(n):
                    values={'observation.state':state[frame],'action':action[frame]}
                    for name in CAMERAS:values['observation.images.'+name]=camera[name+'_rgb'][frame]
                    dataset.add_frame(values,task=task,timestamp=frame/60.)
            dataset.save_episode()
            print('COHORT_EXPORT split='+split+' episode='+str(i+1)+'/'+str(len(rows))+' source='+row['episode']+' frames='+str(n),flush=True)
    finally:dataset.stop_image_writer()
    if dataset.meta.total_episodes!=len(rows):raise RuntimeError('episode count mismatch after export')
    return {'split':split,'root':str(target),'repo_id':repo_id,'episodes':len(rows),'frames':dataset.meta.total_frames,
            'explicit_episode_indices':list(range(len(rows)))}

def reader_check(record,rows):
    original=upstream.InstructionGenerator
    class Instruction:
        @staticmethod
        def generate_instruction(value):
            value=json.loads(value) if isinstance(value,str) else value
            return value['capstone_instruction'] if 'capstone_instruction' in value else original.generate_instruction(value)
    upstream.InstructionGenerator=Instruction
    try:
        ds=upstream.LeRobotDataset(record['repo_id'],root=Path(record['root']),episodes=record['explicit_episode_indices'],
          split='train',pin_memory=False,delta_action=True,
          required_features=['observation.state','action']+['observation.images.'+name for name in CAMERAS],
          delta_timestamps={'observation':[-2,0],'action':list(range(20))})
        if len(ds)!=sum(row['frames'] for row in rows):raise ValueError('upstream frame count mismatch')
        # Every episode's start/end tests boundary padding and prevents chunk crossing.
        offset=0
        for row in rows:
            for index in [offset,offset+row['frames']-1]:
                item=ds[index]
                if item['action'].shape!=(20,7) or item['observation.state'].shape!=(2,6):raise ValueError('model-facing shape mismatch')
                if not torch.isfinite(item['action']).all():raise ValueError('nonfinite delta target')
                if item['observation.images.opst_cam'].shape!=(2,3,480,640):raise ValueError('history image shape mismatch')
            if int(ds[offset+row['frames']-1]['action_is_pad'].sum())!=19:raise ValueError('episode boundary action padding mismatch')
            offset+=row['frames']
        return {'actual_upstream_reader_pass':True,'episode_boundary_samples_checked':2*len(rows),
                'explicit_episode_selection':True,'heldout_evaluation_executed':False}
    finally:upstream.InstructionGenerator=original

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'data/derived/task2_multiepisode_20261026')
    args=parser.parse_args()
    for name,expected in [('lerobot','0.3.3'),('av','14.2.0'),('numpy','1.26.4'),('pyarrow','19.0.1')]:
        if importlib.metadata.version(name)!=expected:raise ValueError(name+' must be '+expected)
    manifest=json.loads((PLAN/'split_manifest.json').read_text());render=json.loads((PLAN/'render_sources.json').read_text())
    if render.get('visuals_ready_count')!=56 or render['manifest_sha256']!=sha(PLAN/'split_manifest.json'):raise ValueError('render/split manifest not ready')
    groups={split:[row for row in render['sources'] if row['split']==split] for split in ['train','validation']}
    if [len(groups[k]) for k in ['train','validation']]!=[40,8]:raise ValueError('frozen split changed')
    if set(r['episode'] for r in groups['train'])&set(r['episode'] for r in groups['validation']):raise ValueError('split overlap')
    cfg=json.loads((ROOT/'configs/franka_action_interface.json').read_text());adapter=FrankaCartesianAdapter(cfg['urdf'])
    for rows in groups.values():
        for row in rows:
            if not Path(row['expected_camera_archive']).is_file():raise FileNotFoundError(row['expected_camera_archive'])
    print('CPU_PREFLIGHT_OK train=40/11128frames validation=8/1848frames test_reserved=not_exported',flush=True)
    if not args.run:print('NO_EXPORT_STARTED; add --run',flush=True);return
    args.output.mkdir(parents=True,exist_ok=True)
    import datasets
    datasets.config.HF_DATASETS_CACHE=args.output.resolve()/'hf_datasets_cache'
    encoder=lr.encode_video_frames;lr.encode_video_frames=partial(encoder,vcodec='h264',crf=30,g=2)
    results=[]
    try:
        for split,rows in groups.items():
            record=convert_split(split,rows,args.output,adapter,sha(PLAN/'split_manifest.json'))
            record.update(reader_check(record,rows));results.append(record)
    finally:lr.encode_video_frames=encoder
    report={'status':'MULTIEPISODE_LEROBOT_EXPORT_PASS','datasets':results,'fps':60,
       'test_reserved_exported':False,'test_model_evaluation_executed':False,'training_started':False,
       'action_format':'absolute pose stored; delta applied by actual upstream reader relative to current state',
       'normalization_warning':'future training must compute delta statistics from TRAIN only, never from validation or test',
       'source_assistance':'declared attached in original demonstrations; not a new unassisted controller rollout'}
    (args.output/'export_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('MULTIEPISODE_LEROBOT_EXPORT_PASS report='+str(args.output/'export_report.json'),flush=True)

if __name__=='__main__':main()
