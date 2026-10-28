#!/usr/bin/env python3
"""Pinned pretrained DynamicVLA forward/backward smoke; no optimizer or robot."""
import argparse
import copy
from dataclasses import fields
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'third_party/DynamicVLA'))
from capstone_vla.pilot_dataset import PilotH5Dataset,action_chunk
from capstone_vla import local_cache
import utils
utils.memcached=local_cache
sys.modules['utils.memcached']=local_cache
from utils.datasets import LeRobotDataset
import utils.datasets as upstream_dataset
from policies.dynamicvla.configuration_dynamicvla import DynamicVLAConfig
from policies.dynamicvla.modeling_dynamicvla import DynamicVLAPolicy,load_dynamicvla
from lerobot.configs.types import FeatureType,NormalizationMode,PolicyFeature
from transformers import AutoConfig,AutoTokenizer

BASE=ROOT/'data/derived/task2_dynamicvla_development_20261025'

def make_config(raw,device='cpu'):
    values={k:copy.deepcopy(v) for k,v in raw.items() if k in {f.name for f in fields(DynamicVLAConfig)}}
    for key in ['input_features','output_features']:
        values[key]={n:PolicyFeature(type=FeatureType(v['type']),shape=tuple(v['shape'])) for n,v in raw[key].items()}
    values['normalization_mapping']={k:NormalizationMode(v) for k,v in raw['normalization_mapping'].items()}
    values.update(device=device,freeze_vision_model=True,freeze_connector=True,freeze_text_model=True,
                  enable_streaming=False,push_to_hub=False)
    cfg=DynamicVLAConfig(**values)
    if (cfg.n_obs_steps,cfg.chunk_size,cfg.max_state_dim,cfg.max_action_dim)!=(2,20,32,32):
        raise ValueError('unexpected pretrained architecture')
    return cfg

def get_dataset():
    original=upstream_dataset.InstructionGenerator
    class VerbatimInstruction:
        @staticmethod
        def generate_instruction(value):
            obj=json.loads(value) if isinstance(value,str) else value
            return obj['capstone_instruction'] if 'capstone_instruction' in obj else original.generate_instruction(value)
    upstream_dataset.InstructionGenerator=VerbatimInstruction
    return LeRobotDataset(repo_id='capstone-local/task2-development',root=BASE/'lerobot_v21_av14',
        split='train',episodes=[0],pin_memory=False,delta_action=True,
        required_features=['observation.images.opst_cam','observation.images.wrist_cam','observation.state','action'],
        delta_timestamps={'observation':[-2,0],'action':list(range(20))})

def dataset_stats():
    ref=PilotH5Dataset(BASE/'episode_000000.hdf5')
    chunks=[]
    for i in range(len(ref)):
        values,pad=action_chunk(ref.actions,ref.states,i)
        chunks.append(values[~pad])
    def stats(values):
        std=values.std(axis=0).astype(np.float32)
        std[std<1e-6]=1.
        return {'mean':values.mean(axis=0).astype(np.float32),'std':std,
                'min':values.min(axis=0).astype(np.float32),'max':values.max(axis=0).astype(np.float32)}
    return {'observation.state':stats(ref.states),'action':stats(np.concatenate(chunks))}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true',help='download pinned policy weights and small tokenizer/config assets')
    parser.add_argument('--run',action='store_true',help='run exactly one CUDA forward/backward, without optimizer step')
    parser.add_argument('--frame',type=int,default=225)
    args=parser.parse_args()
    pin=json.loads((ROOT/'configs/dynamicvla_model_smoke.json').read_text())
    raw=json.loads((ROOT/'configs/dynamicvla_dom_pretrained.json').read_text())
    head=subprocess.check_output(['git','-C',str(ROOT/'third_party/DynamicVLA'),'rev-parse','HEAD'],text=True).strip()
    if head!=pin['upstream_commit']:raise ValueError('upstream revision changed')
    cfg=make_config(raw)
    ds=get_dataset();item=ds[args.frame]
    batch=torch.utils.data.default_collate([item])
    # Exact alias consumed by this pinned upstream forward implementation.
    batch['actions_id_pad']=batch['action_is_pad'].clone()
    assert batch['action'].shape==(1,20,7) and batch['observation.state'].shape==(1,2,6)
    ref=PilotH5Dataset(BASE/'episode_000000.hdf5')[args.frame]
    np.testing.assert_allclose(batch['action'][0].numpy(),ref['action'],rtol=0,atol=1e-6)
    stats=dataset_stats()
    for record in stats.values():
        assert np.isfinite(record['mean']).all() and (record['std']>0).all()
    print('CPU_PREFLIGHT_OK pretrained_config batch=(1,20,7) current_frame='+str(args.frame)
          +' padded_actions='+str(int(batch['action_is_pad'].sum()))+' mask_alias=actions_id_pad',flush=True)
    model_dir=ROOT/'data/models/dynamicvla_dom'/pin['revision']
    backbone_dir=ROOT/'data/models/smollm2_metadata'/pin['backbone_revision']
    if args.download:
        from huggingface_hub import snapshot_download
        print('DOWNLOAD pinned official DOM weights (~1.12 GB); backbone config/tokenizer only',flush=True)
        snapshot_download(pin['repo_id'],revision=pin['revision'],local_dir=model_dir,
                          allow_patterns=['config.json','model.safetensors','LICENSE','README.md'])
        snapshot_download(pin['backbone_repo'],revision=pin['backbone_revision'],local_dir=backbone_dir,
                          allow_patterns=['*.json','*.txt','*.model','*.tiktoken'])
    if not args.run:
        print('GPU_NOT_STARTED; use --download --run for the single pretrained batch',flush=True)
        return
    weights=model_dir/'model.safetensors'
    if not weights.is_file() or not (backbone_dir/'config.json').is_file():raise FileNotFoundError('Run with --download --run first')
    downloaded=json.loads((model_dir/'config.json').read_text())
    if downloaded!=raw:raise ValueError('downloaded policy configuration differs from audited config')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable in the isolated model environment')
    torch.manual_seed(0);torch.set_num_threads(8)
    original_config=AutoConfig.from_pretrained
    original_tokenizer=AutoTokenizer.from_pretrained
    def local_config(name,*a,**kw):
        if name!=pin['backbone_repo']:raise ValueError('Unexpected backbone request: '+str(name))
        return original_config(str(backbone_dir),*a,local_files_only=True,**kw)
    def local_tokenizer(name,*a,**kw):
        if name!=pin['backbone_repo']:raise ValueError('Unexpected tokenizer request: '+str(name))
        return original_tokenizer(str(backbone_dir),*a,local_files_only=True,**kw)
    print('BUILD_MODEL frozen vision/connector/text; loading complete official policy weights',flush=True)
    with patch.object(AutoConfig,'from_pretrained',side_effect=local_config),patch.object(AutoTokenizer,'from_pretrained',side_effect=local_tokenizer):
        policy=DynamicVLAPolicy(cfg,dataset_stats=stats)
    # Official loader rejects missing/unexpected policy keys, except normalizers.
    load_dynamicvla(policy,weights,device='cpu',checkpoint_keys_mapping='model._orig_mod.//model.')
    policy=policy.to('cuda:0');policy.train()
    # A frozen backbone remains in evaluation mode (including BatchNorm).
    vlm=policy.model.vlm_with_expert
    vlm.get_vlm_model().eval()
    trainable=[p for p in policy.parameters() if p.requires_grad]
    if not trainable:raise RuntimeError('no trainable parameters')
    batch={k:v.to('cuda:0') if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
    policy.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    print('FORWARD_BEGIN',flush=True)
    loss,details=policy(batch)
    if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
    if 'losses_after_in_ep_bound' not in details:raise RuntimeError('padding mask was ignored')
    masked=details['losses_after_in_ep_bound'][batch['action_is_pad']]
    if masked.numel() and not torch.all(masked==0):raise RuntimeError('padded targets contributed to loss')
    print('BACKWARD_BEGIN loss='+str(float(loss.detach())),flush=True)
    loss.backward()
    gradients=[p.grad for p in trainable if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):raise RuntimeError('missing or nonfinite gradients')
    norm=float(torch.sqrt(sum((g.float()**2).sum() for g in gradients)))
    if norm<=0:raise RuntimeError('all gradients are zero')
    torch.cuda.synchronize()
    out=BASE/'model_smoke'/datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    report={'status':'PRETRAINED_FORWARD_BACKWARD_PASS','frame':args.frame,'loss':float(loss.detach()),
       'gradient_l2':norm,'gradient_tensor_count':len(gradients),
       'trainable_parameters':sum(p.numel() for p in trainable),'total_parameters':sum(p.numel() for p in policy.parameters()),
       'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
       'model_repo':pin['repo_id'],'model_revision':pin['revision'],'backbone_revision':pin['backbone_revision'],
       'upstream_commit':head,'normalization':'development state statistics and valid delta-chunk targets; std<1e-6 ->1',
       'padding_alias':'action_is_pad -> actions_id_pad','padded_actions':int(batch['action_is_pad'].sum()),
       'optimizer_step':False,'finetuned_weights_saved':False,'heldout_evaluation':False,'robot_commands':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PRETRAINED_FORWARD_BACKWARD_PASS optimizer_step=false report='+str(out/'report.json'),flush=True)

if __name__=='__main__':main()
