#!/usr/bin/env python3
"""Local LeRobot v2.1 export + actual pinned DynamicVLA loader, no model/GPU."""
import argparse
from functools import partial
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import h5py
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from capstone_vla.pilot_dataset import PilotH5Dataset,CAMERAS
BASE=ROOT/'data/derived/task2_dynamicvla_development_20261025'
PIN='434dc7105cf6d559018d15af5fecbcdad44f036e'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,default=ROOT/'third_party/DynamicVLA')
    parser.add_argument('--dataset',type=Path,default=BASE/'lerobot_v21_av14')
    args=parser.parse_args()
    upstream=args.upstream.resolve();target=args.dataset.resolve()
    commit=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    if commit!=PIN:raise ValueError('upstream commit does not match the audited revision')
    if importlib.metadata.version('lerobot')!='0.3.3':raise ValueError('requires isolated lerobot==0.3.3')
    import lerobot.datasets.lerobot_dataset as lr
    repo_id='capstone-local/task2-development'
    payload=BASE/'episode_000000.hdf5'
    if not target.exists():
        features={'observation.state':{'dtype':'float32','shape':(6,),'names':['x','y','z','rx','ry','rz']},
                  'action':{'dtype':'float32','shape':(7,),'names':['x','y','z','rx','ry','rz','gripper']}}
        for name in CAMERAS:
            features['observation.images.'+name]={'dtype':'video','shape':(480,640,3),'names':['height','width','channel']}
        # Ubuntu's installed encoder supports H.264; no system upgrade needed.
        original_encoder=lr.encode_video_frames
        lr.encode_video_frames=partial(original_encoder,vcodec='h264')
        dataset=lr.LeRobotDataset.create(repo_id=repo_id,root=target,fps=60,
              features=features,robot_type='franka',use_videos=True,image_writer_threads=4)
        try:
            with h5py.File(payload,'r') as f:
                instruction=json.dumps({'task':'pick','objects':['moving object'],
                       'capstone_instruction':str(f.attrs['instruction'])},ensure_ascii=False)
                for i in range(len(f['action'])):
                    frame={'observation.state':f['observation.state'][i],'action':f['action'][i]}
                    for name in CAMERAS:frame['observation.images.'+name]=f['observation.images/'+name][i]
                    dataset.add_frame(frame,task=instruction,timestamp=i/60.)
                    if i%40==0:print('LEROBOT_EXPORT',i,'/232',flush=True)
            dataset.save_episode()
            (target/'capstone_provenance.json').write_text(json.dumps({
                 'purpose':'development_overfit_only','source_payload':str(payload),
                 'upstream_commit':PIN,'training_started':False,'held_out_evaluation':False},indent=2)+'\n')
        finally:
            dataset.stop_image_writer()
            lr.encode_video_frames=original_encoder
    else:
        provenance=target/'capstone_provenance.json'
        if not provenance.is_file():raise RuntimeError('Existing export is incomplete; preserve it and use a new --dataset directory')
        if json.loads(provenance.read_text())['source_payload']!=str(payload):raise ValueError('wrong source payload')
    sys.path.insert(0,str(upstream))
    # Upstream imports pylibmc unconditionally even for pin_memory=False.
    # Bind the real local file-reading implementation for this process only.
    import utils
    from capstone_vla import local_cache
    utils.memcached=local_cache
    sys.modules['utils.memcached']=local_cache
    import utils.datasets as upstream_data
    # This is a capstone runtime adapter, not an edit to upstream files.
    original_generator=upstream_data.InstructionGenerator
    class CapstoneInstructionGenerator:
        @staticmethod
        def generate_instruction(metadata):
            value=json.loads(metadata) if isinstance(metadata,str) else metadata
            if 'capstone_instruction' in value:return value['capstone_instruction']
            return original_generator.generate_instruction(metadata)
    upstream_data.InstructionGenerator=CapstoneInstructionGenerator
    required=['observation.images.'+c for c in CAMERAS]+['observation.state','action']
    try:
        ds=upstream_data.LeRobotDataset(repo_id=repo_id,root=target,split='train',episodes=[0],
             pin_memory=False,delta_action=True,required_features=required,
             delta_timestamps={'observation':[-2,0],'action':list(range(20))})
        reference=PilotH5Dataset(payload)
        if len(ds)!=232:raise ValueError('upstream reader dropped frames')
        shapes={}
        for i in [0,1,112,113,114,116,231]:
            item=ds[i];ref=reference[i]
            for key in ['action','observation.state','action_is_pad','observation.state_is_pad']:
                np.testing.assert_allclose(item[key].numpy(),ref[key],rtol=0,atol=1e-6)
            if item['task']!=ref['task']:raise ValueError('instruction text changed')
            for name in CAMERAS:
                image=item['observation.images.'+name]
                if tuple(image.shape)!=(2,3,480,640) or not torch.isfinite(image).all():raise ValueError('image history invalid')
                if image.min()<0 or image.max()>1:raise ValueError('image scaling invalid')
            shapes[str(i)]={k:list(v.shape) for k,v in item.items() if isinstance(v,torch.Tensor)}
        samples=[ds[112],ds[113]]
        batch=torch.utils.data.default_collate(samples)
        if tuple(batch['action'].shape)!=(2,20,7):raise ValueError('batch shape mismatch')
        report={'status':'ACTUAL_UPSTREAM_LOADER_PASS','frames':len(ds),'fps':ds.meta.fps,
           'upstream_commit':PIN,'lerobot_version':importlib.metadata.version('lerobot'),
           'torch_version':torch.__version__,'cache_backend':'capstone_local_files_no_memcached','episode_indices':[0],'split':'development_overfit_only',
           'instruction_adapter':'preserve verbatim capstone instruction',
           'delta_and_padding_match_numpy_reference':True,'batch_action_shape':list(batch['action'].shape),
           'sample_shapes':shapes,'video_encoding':'H.264 lossy; not pixel-identical to HDF5',
           'model_forward_executed':False,'training_started':False,'official_benchmark_success':None}
        (BASE/'upstream_loader_smoke.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print('ACTUAL_UPSTREAM_LOADER_PASS frames=232 batch_action=(2,20,7) model_forward=false training_started=false',flush=True)
    finally:
        upstream_data.InstructionGenerator=original_generator

if __name__=='__main__':main()
