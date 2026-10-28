#!/usr/bin/env python3
"""Single-episode pretrained fine-tuning diagnostic; no simulation/evaluation split."""
import argparse
from collections import OrderedDict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from unittest.mock import patch
from zoneinfo import ZoneInfo
import numpy as np
import torch
from safetensors import safe_open

from smoke_task2_pretrained import (ROOT,BASE,make_config,get_dataset,dataset_stats,
    DynamicVLAPolicy,load_dynamicvla,AutoConfig,AutoTokenizer,upstream_dataset)

def valid_flow_mse(losses,padding):
    valid=(~padding).unsqueeze(-1).expand(*padding.shape,7)
    if not bool(valid.any()):raise ValueError('no valid action targets')
    return losses[...,:7][valid].mean()

def validate_plan(plan,length):
    for key in ['steps','batch_size','cache_frames','log_every','checkpoint_frequency']:
        if not isinstance(plan[key],int) or plan[key]<1:raise ValueError('invalid '+key)
    for key in ['learning_rate','gradient_clip_norm']:
        if not np.isfinite(plan[key]) or plan[key]<=0:raise ValueError('invalid '+key)
    if not plan['probe_frames'] or any(i<0 or i>=length for i in plan['probe_frames']):raise ValueError('probe frame outside development episode')
    if not plan['probe_times'] or any(t<=0 or t>=1 for t in plan['probe_times']):raise ValueError('probe time must be in (0,1)')
    if plan.get('heldout_evaluation') is not False:raise ValueError('this is development-only')

def batch_from_indices(dataset,indices,cache,cache_limit,device):
    samples=[]
    for index in indices:
        index=int(index)
        if index not in cache:
            cache[index]=dataset[index]
            if len(cache)>cache_limit:cache.popitem(last=False)
        else:cache.move_to_end(index)
        samples.append(cache[index])
    batch=torch.utils.data.default_collate(samples)
    batch['actions_id_pad']=batch['action_is_pad'].clone()
    return {key:value.to(device) if isinstance(value,torch.Tensor) else value for key,value in batch.items()}

def paired_probe(policy,dataset,plan,cache):
    policy.eval()
    values=[]
    with torch.no_grad():
        for index in plan['probe_frames']:
            batch=batch_from_indices(dataset,[index],cache,plan['cache_frames'],'cuda:0')
            for t in plan['probe_times']:
                generator=torch.Generator(device='cuda:0').manual_seed(plan['probe_seed']+index)
                noise=torch.randn((1,20,32),generator=generator,device='cuda:0')
                _,details=policy(batch,noise=noise,time=torch.tensor([t],device='cuda:0'))
                value=valid_flow_mse(details['losses_after_forward'],batch['action_is_pad'])
                if not torch.isfinite(value):raise RuntimeError('nonfinite probe loss')
                values.append({'frame':index,'flow_time':t,'valid_7d_flow_mse':float(value)})
    return {'mean_valid_7d_flow_mse':float(np.mean([v['valid_7d_flow_mse'] for v in values])),
            'samples':values,'same_training_episode':True,'heldout_evaluation':False}

def checkpoint(policy,directory,stats,run_metadata):
    directory.mkdir(parents=True,exist_ok=False)
    # Official serialization includes model config, weights and normalizers.
    policy.save_pretrained(directory)
    (directory/'capstone_normalization.json').write_text(json.dumps(
        {key:{name:value.tolist() for name,value in fields.items()} for key,fields in stats.items()},indent=2)+'\n')
    (directory/'capstone_run.json').write_text(json.dumps(run_metadata,indent=2)+'\n')
    with safe_open(str(directory/'model.safetensors'),framework='pt',device='cpu') as handle:
        if 'model.action_out_proj.weight' not in handle.keys():raise RuntimeError('checkpoint missing action projection')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/task2_development_finetune.json')
    parser.add_argument('--steps',type=int)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    plan=json.loads(args.config.read_text())
    if args.steps is not None:plan['steps']=args.steps
    pin=json.loads((ROOT/'configs/dynamicvla_model_smoke.json').read_text())
    raw=json.loads((ROOT/'configs/dynamicvla_dom_pretrained.json').read_text())
    cfg=make_config(raw)
    ds=get_dataset();validate_plan(plan,len(ds))
    ds.image_transforms=upstream_dataset.ImageTransforms(img_size=plan['image_size'])
    cache=OrderedDict()
    batch=batch_from_indices(ds,[112,225],cache,plan['cache_frames'],'cpu')
    assert batch['action'].shape==(2,20,7)
    assert batch['observation.images.opst_cam'].shape==(2,2,3,360,480)
    assert int(batch['action_is_pad'][1].sum())==13
    model_dir=ROOT/'data/models/dynamicvla_dom'/pin['revision']
    backbone=ROOT/'data/models/smollm2_metadata'/pin['backbone_revision']
    weights=model_dir/'model.safetensors'
    for path in [weights,backbone/'config.json',BASE/'upstream_loader_smoke.json']:
        if not path.is_file():raise FileNotFoundError(path)
    if json.loads((model_dir/'config.json').read_text())!=raw:raise ValueError('checkpoint config changed')
    head=subprocess.check_output(['git','-C',str(ROOT/'third_party/DynamicVLA'),'rev-parse','HEAD'],text=True).strip()
    if head!=pin['upstream_commit']:raise ValueError('upstream revision changed')
    stats=dataset_stats()
    print('CPU_PREFLIGHT_OK frames=232 steps='+str(plan['steps'])+' batch_size='+str(plan['batch_size'])+
          ' purpose=development_overfit_only checkpoint=local_pretrained',flush=True)
    if not args.run:
        print('NO_TRAINING_STARTED; add --run',flush=True);return
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(8);torch.manual_seed(plan['seed']);np.random.seed(plan['seed']);random.seed(plan['seed'])
    run=BASE/'finetune_runs'/datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True,exist_ok=False)
    metadata={'purpose':plan['purpose'],'plan':plan,'model_pin':{k:pin[k] for k in ['repo_id','revision','backbone_repo','backbone_revision']},'upstream_commit':head,
              'source_payload':str(BASE/'episode_000000.hdf5'),'policy_observation':'RGB + measured EE state + instruction',
              'heldout_evaluation':False,'simulation_rollout':False,'robot_commands':False,
              'training_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (run/'run_config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('OUTPUT',run,flush=True)
    original_config=AutoConfig.from_pretrained;original_tokenizer=AutoTokenizer.from_pretrained
    def local_config(name,*a,**kw):
        if name!=pin['backbone_repo']:raise ValueError('unexpected model dependency')
        return original_config(str(backbone),*a,local_files_only=True,**kw)
    def local_tokenizer(name,*a,**kw):
        if name!=pin['backbone_repo']:raise ValueError('unexpected tokenizer dependency')
        return original_tokenizer(str(backbone),*a,local_files_only=True,**kw)
    with patch.object(AutoConfig,'from_pretrained',side_effect=local_config),patch.object(AutoTokenizer,'from_pretrained',side_effect=local_tokenizer):
        policy=DynamicVLAPolicy(cfg,dataset_stats=stats)
    load_dynamicvla(policy,weights,device='cpu',checkpoint_keys_mapping='model._orig_mod.//model.')
    policy.to('cuda:0')
    parameters=[p for p in policy.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=plan['learning_rate'],betas=(.9,.95),eps=1e-8,weight_decay=1e-10)
    print('PAIRED_PROBE_BEFORE',flush=True)
    before=paired_probe(policy,ds,plan,cache)
    (run/'probe_before.json').write_text(json.dumps(before,indent=2)+'\n')
    print('BASELINE_VALID_7D_FLOW_MSE',before['mean_valid_7d_flow_mse'],flush=True)
    initial_projection=policy.model.action_out_proj.weight.detach().clone()
    sampler=np.random.default_rng(plan['seed'])
    sampled=set();start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    with (run/'training.jsonl').open('x') as log:
        for step in range(1,plan['steps']+1):
            policy.train();policy.model.vlm_with_expert.get_vlm_model().eval()
            indices=sampler.integers(0,len(ds),size=plan['batch_size']);sampled.update(map(int,indices))
            batch=batch_from_indices(ds,indices,cache,plan['cache_frames'],'cuda:0')
            optimizer.zero_grad(set_to_none=True)
            loss,details=policy(batch)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
            valid_mse=valid_flow_mse(details['losses_after_forward'],batch['action_is_pad'])
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(parameters,plan['gradient_clip_norm'],error_if_nonfinite=True)
            optimizer.step()
            elapsed=time.monotonic()-start
            row={'step':step,'source_frames':indices.tolist(),'official_training_loss':float(loss.detach()),
                 'valid_7d_flow_mse':float(valid_mse.detach()),'gradient_norm_before_clip':float(norm),'elapsed_s':elapsed}
            log.write(json.dumps(row)+'\n');log.flush()
            if step==1 or step%plan['log_every']==0:
                print('TRAIN step='+str(step)+'/'+str(plan['steps'])+' loss='+format(row['official_training_loss'],'.5f')+
                      ' valid7='+format(row['valid_7d_flow_mse'],'.5f')+' elapsed='+format(elapsed,'.0f')+'s'+
                      ' eta='+format(elapsed/step*(plan['steps']-step),'.0f')+'s',flush=True)
            if step%plan['checkpoint_frequency']==0 and step<plan['steps']:
                checkpoint(policy,run/f'checkpoint_step_{step:06d}',stats,{**metadata,'completed_steps':step})
    optimizer.zero_grad(set_to_none=True)
    projection_change=float(torch.linalg.vector_norm(policy.model.action_out_proj.weight.detach()-initial_projection))
    if not np.isfinite(projection_change) or projection_change<=0:raise RuntimeError('action projection did not change')
    print('PAIRED_PROBE_AFTER',flush=True)
    after=paired_probe(policy,ds,plan,cache)
    (run/'probe_after.json').write_text(json.dumps(after,indent=2)+'\n')
    saved=run/'checkpoint_final'
    checkpoint(policy,saved,stats,{**metadata,'completed_steps':plan['steps']})
    first=before['mean_valid_7d_flow_mse'];last=after['mean_valid_7d_flow_mse']
    report={'status':'DEVELOPMENT_FINETUNE_COMPLETED','completed_optimizer_steps':plan['steps'],
           'batch_size':plan['batch_size'],'unique_training_frames_sampled':len(sampled),
           'paired_probe_before_valid_7d_flow_mse':first,'paired_probe_after_valid_7d_flow_mse':last,
           'paired_probe_relative_reduction':(first-last)/first if first else None,
           'paired_probe_improved':last<first,'action_projection_l2_change':projection_change,
           'checkpoint':str(saved),'peak_pytorch_allocated_bytes':torch.cuda.max_memory_allocated(),
           'heldout_evaluation':False,'task_success_verified':False,'simulation_rollout':False,
           'comparison':'same training episode, identical fixed noise and flow times; not action prediction error',
           'normalization':'valid delta chunks + measured state; checkpoint sidecar preserves these stats'}
    (run/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('DEVELOPMENT_FINETUNE_COMPLETED steps='+str(plan['steps'])+' paired_probe_improved='+str(last<first)+
          ' report='+str(run/'report.json'),flush=True)

if __name__=='__main__':main()
