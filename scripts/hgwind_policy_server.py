#!/usr/bin/env python3
"""Isolated DynamicVLA absolute29 inference; live input contains RGB/state only."""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import time
import traceback
import h5py
import numpy as np
import torch
from safetensors.torch import load_model
from finetune_hgwind_development import ROOT,build_policy,make_config,batch_from,upstream_dataset
from predict_hgwind_development import inspect_run,check_normalizers
from capstone_vla.ball_live_protocol import validate_observation
from capstone_vla.live_policy_protocol import atomic_npz
from capstone_vla.pilot_dataset import CAMERAS, temporal_indices
from capstone_vla.hf_archive import sha256


def prepare_input(states,images,instruction,device='cpu'):
    validate_observation(states,images)
    item={'observation.state':torch.tensor(states,dtype=torch.float32)}
    for name in CAMERAS:
        item['observation.images.'+name]=torch.from_numpy(np.ascontiguousarray(images[name].transpose(0,3,1,2),dtype=np.float32)/255.)
    item=upstream_dataset.ImageTransforms(img_size=[360,480])(item,['observation.images.'+name for name in CAMERAS])
    return {**{k:v.unsqueeze(0).to(device).contiguous() for k,v in item.items()},'task':[instruction]}


def check_cpu(run,load=False):
    ds,cfg,stats,pin,weights,_=inspect_run(run)
    with h5py.File(ds.meta['images'],'r') as f:
        for frame in [0,1,2,4,268]:
            ids=temporal_indices(frame,len(ds))[0]
            images={name:np.stack([f[name+'_rgb'][int(i)] for i in ids]) for name in CAMERAS}
            online=prepare_input(ds.states[ids],images,ds.meta['instruction'])
            training=batch_from(ds,[frame],OrderedDict(),'cpu')
            for key,value in online.items():
                if torch.is_tensor(value):torch.testing.assert_close(value,training[key],rtol=0,atol=0)
                elif value!=training[key]:raise ValueError('Live/training instruction differs')
    if load:
        policy=build_policy(cfg,stats,pin,weights,ROOT/'data/models/smollm2_metadata'/pin['backbone_revision'])
        load_model(policy,str(run/'checkpoint_final/model.safetensors'),strict=True,device='cpu');check_normalizers(policy,stats)
        policy.prepare_images(policy.normalize_inputs(online))
    if torch.cuda.is_initialized():raise RuntimeError('CPU preflight initialized CUDA')
    print('BALL_LIVE_MODEL_PREFLIGHT_PASS preprocessing=exact frames=0,1,2,4,268 absolute29=true state30=true strict_load='+str(load).lower()+' CUDA=false',flush=True)


def serve(config):
    ipc=Path(config['output'])/'ipc';checkpoint=Path(config['checkpoint'])
    for path,digest in config['checkpoint_hashes'].items():
        if sha256(path)!=digest:raise ValueError('Live checkpoint changed: '+Path(path).name)
    cfg=make_config(json.loads((checkpoint/'config.json').read_text()))
    if cfg.use_delta_action or tuple(cfg.action_feature.shape)!=(29,) or tuple(cfg.input_features['observation.state'].shape)!=(30,):
        raise ValueError('Wrong learned feature contract')
    stats={k:{n:torch.tensor(v,dtype=torch.float32) for n,v in fields.items()}
           for k,fields in json.loads((checkpoint/'capstone_normalization.json').read_text()).items()}
    pin=config['pretrained'];weights=Path(config['pretrained_weights'])
    policy=build_policy(cfg,stats,pin,weights,ROOT/'data/models/smollm2_metadata'/pin['backbone_revision'])
    load_model(policy,str(checkpoint/'model.safetensors'),strict=True,device='cpu');check_normalizers(policy,stats)
    policy.to('cuda:0').eval();(ipc/'ready').write_text('strict absolute29 checkpoint loaded\n')
    expected=0
    while not (ipc/'stop').exists():
        request=ipc/f'request_{expected:06d}.npz'
        if not request.exists():time.sleep(.02);continue
        with np.load(request,allow_pickle=False) as f:
            if set(f.files)!={'frame','states',*CAMERAS} or int(f['frame'])!=expected:raise ValueError('Unexpected live policy input')
            states=f['states'].copy();images={name:f[name].copy() for name in CAMERAS}
        inputs=prepare_input(states,images,config['instruction'],'cuda:0')
        noise=torch.randn((1,20,32),device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(config['model_seed']+expected))
        policy.reset();torch.cuda.synchronize();start=time.monotonic()
        with torch.inference_mode():absolute=policy.predict_action_chunk(inputs,noise=noise).cpu().numpy()[0]
        if absolute.shape!=(20,29) or not np.isfinite(absolute).all():raise ValueError('Invalid generated absolute29 chunk')
        elapsed=time.monotonic()-start
        atomic_npz(ipc/f'response_{expected:06d}.npz',frame=np.asarray(expected,dtype=np.int64),absolute=absolute,inference_s=np.asarray(elapsed))
        print(f'BALL_POLICY_RESPONSE frame={expected} inference_s={elapsed:.3f}',flush=True)
        expected+=config['prefix']
        if expected>=config['frames']:break
    while not (ipc/'stop').exists():time.sleep(.1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path);p.add_argument('--preflight',type=Path);p.add_argument('--check-load',action='store_true');a=p.parse_args()
    torch.set_num_threads(8);torch.manual_seed(0)
    if a.preflight:check_cpu(a.preflight.resolve(),a.check_load);return
    if a.config is None:p.error('--config or --preflight is required')
    config=json.loads(a.config.read_text())
    try:serve(config)
    except Exception:
        (Path(config['output'])/'ipc/server_error.txt').write_text(traceback.format_exc());raise

if __name__=='__main__':main()
