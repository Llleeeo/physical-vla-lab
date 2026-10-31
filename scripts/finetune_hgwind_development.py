#!/usr/bin/env python3
"""One exposed ball episode: pretrained 29D/30D smoke and bounded fine-tuning.

Default is CPU preflight. --run executes the model, never Isaac Sim or a robot.
--steps 0 runs forward/backward only; positive steps save a development checkpoint.
"""
import argparse
from collections import OrderedDict
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time
from unittest.mock import patch
from zoneinfo import ZoneInfo
import numpy as np
import torch
from safetensors import safe_open
from smoke_task2_pretrained import (ROOT,make_config,DynamicVLAPolicy,
    load_dynamicvla,AutoConfig,AutoTokenizer,upstream_dataset)
from capstone_vla.ball_action_contract import model_configuration, specification
from capstone_vla.hgwind_development_data import BallDevelopmentDataset,normalization,DESTINATION
from capstone_vla.hf_archive import sha256
from capstone_vla.pilot_dataset import CAMERAS


def valid_loss(losses,padding):
    if losses.ndim!=3 or losses.shape[-1]!=32 or padding.shape!=losses.shape[:2]:
        raise ValueError('Unexpected flow loss/mask dimensions')
    valid=(~padding).unsqueeze(-1).expand(*padding.shape,29)
    if not bool(valid.any()):raise ValueError('No valid action targets')
    return losses[...,:29][valid].mean()


def batch_from(ds,indices,cache,device):
    transform=upstream_dataset.ImageTransforms(img_size=[360,480])
    samples=[]
    for i in indices:
        i=int(i)
        if i not in cache:
            item={k:torch.from_numpy(v) if isinstance(v,np.ndarray) else v for k,v in ds[i].items()}
            item=transform(item,['observation.images.'+name for name in CAMERAS])
            cache[i]=item
            if len(cache)>32:cache.popitem(last=False)
        else:cache.move_to_end(i)
        samples.append(cache[i])
    batch=torch.utils.data.default_collate(samples)
    batch['actions_id_pad']=batch['action_is_pad'].clone()
    return {k:v.to(device).contiguous() if isinstance(v,torch.Tensor) else v for k,v in batch.items()}


def probe(policy,ds,cache):
    policy.eval();values=[]
    with torch.no_grad():
        for i in [0,100,150,162,200,268]:
            batch=batch_from(ds,[i],cache,'cuda:0')
            noise=torch.randn((1,20,32),generator=torch.Generator(device='cuda:0').manual_seed(9100+i),device='cuda:0')
            _,details=policy(batch,noise=noise,time=torch.tensor([.5],device='cuda:0'))
            values.append(float(valid_loss(details['losses_after_forward'],batch['action_is_pad'])))
    if not np.isfinite(values).all():raise ValueError('Nonfinite probe')
    return {'frames':[0,100,150,162,200,268],'valid_29d_flow_mse':values,'mean':float(np.mean(values)),
            'same_training_episode':True,'flow_time':.5,'noise_seed':'9100+frame'}


def build_policy(cfg,stats,pin,weights,backbone):
    original_config=AutoConfig.from_pretrained;original_tokenizer=AutoTokenizer.from_pretrained
    def local_config(name,*a,**kw):
        if name!=pin['backbone_repo']:raise ValueError('Unexpected dependency')
        return original_config(str(backbone),*a,local_files_only=True,**kw)
    def local_tokenizer(name,*a,**kw):
        if name!=pin['backbone_repo']:raise ValueError('Unexpected tokenizer')
        return original_tokenizer(str(backbone),*a,local_files_only=True,**kw)
    print('BUILD_MODEL local official pretrained; new absolute29/state30 normalizers',flush=True)
    with patch.object(AutoConfig,'from_pretrained',side_effect=local_config),patch.object(AutoTokenizer,'from_pretrained',side_effect=local_tokenizer):
        policy=DynamicVLAPolicy(cfg,dataset_stats=stats)
    load_dynamicvla(policy,weights,device='cpu',checkpoint_keys_mapping='model._orig_mod.//model.')
    # Check that the official loader retained our newly fitted normalizers.
    for prefix,key in [('normalize_inputs.buffer_observation_state','observation.state'),('normalize_targets.buffer_action','action'),('unnormalize_outputs.buffer_action','action')]:
        for stat in ['mean','std']:
            np.testing.assert_array_equal(policy.state_dict()[prefix+'.'+stat].numpy(),stats[key][stat])
    return policy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=ROOT/DESTINATION)
    parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--check-load',action='store_true',help='Load official model and preprocess on CPU, no forward/CUDA')
    args=parser.parse_args()
    if args.run and args.check_load:parser.error('Choose --run or --check-load')
    if not 0<=args.steps<=2000:parser.error('--steps must be 0..2000')
    torch.set_num_threads(8)
    ds=BallDevelopmentDataset(args.dataset)
    pin=json.loads((ROOT/'configs/dynamicvla_model_smoke.json').read_text())
    raw=json.loads((ROOT/'configs/dynamicvla_dom_pretrained.json').read_text())
    cfg=make_config(model_configuration(raw));stats=normalization(ds.states,ds.actions)
    head=subprocess.check_output(['git','-C',str(ROOT/'third_party/DynamicVLA'),'rev-parse','HEAD'],text=True).strip()
    if head!=pin['upstream_commit']:raise ValueError('Pinned upstream revision changed')
    model_dir=ROOT/'data/models/dynamicvla_dom'/pin['revision'];weights=model_dir/'model.safetensors'
    backbone=ROOT/'data/models/smollm2_metadata'/pin['backbone_revision']
    if json.loads((model_dir/'config.json').read_text())!=raw:raise ValueError('Official checkpoint configuration changed')
    if not (backbone/'config.json').exists():raise FileNotFoundError(backbone)
    with safe_open(str(weights),framework='pt',device='cpu') as f:
        for name,shape in [('model.action_in_proj.weight',[720,32]),('model.action_out_proj.weight',[32,720]),('model.state_proj.weight',[960,32])]:
            if f.get_slice(name).get_shape()!=shape:raise ValueError('Checkpoint projection mismatch: '+name)
    cache=OrderedDict();batch=batch_from(ds,[0,268],cache,'cpu')
    if batch['action'].shape!=(2,20,29) or batch['observation.state'].shape!=(2,2,30):raise ValueError('Model batch shape mismatch')
    if int(batch['action_is_pad'][1].sum())!=19 or cfg.use_delta_action:raise ValueError('Absolute action/padding mismatch')
    for i in range(len(ds)):
        # Check temporal/action indexing without loading all RGB a second time.
        from capstone_vla.pilot_dataset import temporal_indices
        obs,_,act,pad=temporal_indices(i,len(ds))
        if (obs>i).any() or (act[~pad]<i).any():raise ValueError('Noncausal indices')
    print('CPU_PREFLIGHT_OK episode=local/traj_00018 frames=269 batch=(2,20,29) state=(2,2,30) absolute=true steps='+str(args.steps),flush=True)
    if args.check_load:
        policy=build_policy(cfg,stats,pin,weights,backbone)
        normalized=policy.normalize_targets(policy.normalize_inputs(batch))
        images,masks=policy.prepare_images(normalized)
        state=policy.prepare_state(normalized);action=policy.prepare_action(normalized)
        tokens,language_mask=policy.prepare_language(normalized)
        if state.shape!=(2,32) or action.shape!=(2,20,32):raise ValueError('Padded upstream shape mismatch')
        if not all(torch.isfinite(x).all() for x in [state,action,*images]):raise ValueError('Nonfinite preprocessing')
        losses=torch.ones((2,20,32),requires_grad=True)
        loss=valid_loss(losses,batch['action_is_pad']);loss.backward()
        if losses.grad[...,29:].count_nonzero() or losses.grad[batch['action_is_pad']].count_nonzero():raise ValueError('Invalid gradient mask')
        if torch.cuda.is_initialized():raise RuntimeError('CPU check unexpectedly initialized CUDA')
        print('CPU_PRETRAINED_LOAD_PASS normalization=exact state=(2,32) action=(2,20,32) language_tokens='+str(tokens.shape[-1])+' gradient_mask=29_only CUDA=false',flush=True)
        return
    if not args.run:
        print('GPU_NOT_STARTED; add --run (or --steps 0 --run for forward/backward only)',flush=True);return
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.manual_seed(31);rng=np.random.default_rng(31)
    out=args.dataset.resolve()/'training_runs'/datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False);print('OUTPUT',out,flush=True)
    metadata={'purpose':'development_overfit_only','source_dataset':str(args.dataset.resolve()),
        'dataset_manifest_sha256':sha256(args.dataset/'dataset.json'),'steps':args.steps,'batch_size':2,
        'seed':31,'learning_rate':1.e-4,'image_size':[360,480],'augmentation':False,
        'backbone_frozen':True,'absolute_actions':True,'schema':specification(),
        'pretrained':pin,'upstream_commit':head,'optimizer_loss':'valid unpadded first29 channels of upstream flow loss',
        'official_32d_loss_reported_separately':True,'normalization_fit':'this development episode only; valid action chunk occurrences',
        'heldout_evaluation':False,'simulation_rollout':False,'task_success_verified':False,
        'source_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'src/capstone_vla/hgwind_development_data.py',ROOT/'src/capstone_vla/ball_action_contract.py',ROOT/'scripts/smoke_task2_pretrained.py',ROOT/'src/capstone_vla/pilot_dataset.py']},
        'weights_sha256':sha256(weights)}
    (out/'run_config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    policy=build_policy(cfg,stats,pin,weights,backbone)
    policy.to('cuda:0');parameters=[p for p in policy.parameters() if p.requires_grad]
    initial=policy.model.action_out_proj.weight.detach().clone()
    optimizer=torch.optim.AdamW(parameters,lr=1.e-4,betas=(.9,.95),eps=1.e-8,weight_decay=1.e-10)
    before=probe(policy,ds,cache) if args.steps else None
    if before:print('PAIRED_FLOW_BEFORE',before['mean'],flush=True)
    start=time.monotonic();torch.cuda.reset_peak_memory_stats();seen=set()
    with (out/'training.jsonl').open('x') as log:
        for step in range(1,max(args.steps,1)+1):
            policy.train();policy.model.vlm_with_expert.get_vlm_model().eval()
            indices=rng.integers(0,len(ds),size=2) if args.steps else np.array([0,268])
            seen.update(map(int,indices));batch=batch_from(ds,indices,cache,'cuda:0')
            optimizer.zero_grad(set_to_none=True)
            official,details=policy(batch)
            loss=valid_loss(details['losses_after_forward'],batch['action_is_pad'])
            masked=details['losses_after_in_ep_bound'][batch['action_is_pad']]
            if not torch.isfinite(loss) or not torch.isfinite(official) or (masked.numel() and not torch.all(masked==0)):
                raise RuntimeError('Nonfinite loss or ignored padding')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(parameters,10.,error_if_nonfinite=True)
            if float(norm)<=0:raise RuntimeError('Zero gradients')
            if args.steps:optimizer.step()
            row={'step':step,'optimizer_step':bool(args.steps),'frames':indices.tolist(),
                'valid_29d_flow_mse':float(loss.detach()),'official_32d_loss':float(official.detach()),
                'gradient_norm_before_clip':float(norm),'elapsed_s':time.monotonic()-start}
            log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
            if step==1 or step%25==0:print('TRAIN',step,'/',args.steps,'valid29='+format(row['valid_29d_flow_mse'],'.6f'),flush=True)
    after=probe(policy,ds,cache) if args.steps else None
    optimizer.zero_grad(set_to_none=True)
    change=float(torch.linalg.vector_norm(policy.model.action_out_proj.weight.detach()-initial))
    checkpoint=None
    if args.steps:
        if not np.isfinite(change) or change<=0:raise RuntimeError('Model projection did not update')
        checkpoint=out/'checkpoint_final';policy.save_pretrained(checkpoint)
        (checkpoint/'capstone_normalization.json').write_text(json.dumps({k:{n:v.tolist() for n,v in s.items()} for k,s in stats.items()},indent=2)+'\n')
        (checkpoint/'capstone_run.json').write_text(json.dumps(metadata,indent=2)+'\n')
        with safe_open(str(checkpoint/'model.safetensors'),framework='pt',device='cpu') as f:
            for prefix,key in [('normalize_inputs.buffer_observation_state','observation.state'),('normalize_targets.buffer_action','action'),('unnormalize_outputs.buffer_action','action')]:
                for stat in ['mean','std']:np.testing.assert_array_equal(f.get_tensor(prefix+'.'+stat).numpy(),stats[key][stat])
    report={'status':'BALL_DEVELOPMENT_FINETUNE_COMPLETED' if args.steps else 'BALL_PRETRAINED_FORWARD_BACKWARD_PASS',
        'optimizer_steps':args.steps,'unique_training_frames':len(seen),'probe_before':before,'probe_after':after,
        'paired_probe_improved':after['mean']<before['mean'] if after else None,
        'checkpoint':str(checkpoint) if checkpoint else None,'action_projection_l2_change':change,
        'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'heldout_evaluation':False,
        'simulation_rollout':False,'task_success_verified':False,'loss_is_physical_action_error':False}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(report['status']+' report='+str(out/'report.json'),flush=True)

if __name__=='__main__':main()
