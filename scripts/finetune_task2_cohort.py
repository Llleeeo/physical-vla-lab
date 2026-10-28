#!/usr/bin/env python3
"""Train40/validate8 from official weights; reserved test is never opened."""
import argparse
from collections import OrderedDict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import subprocess
from unittest.mock import patch
from zoneinfo import ZoneInfo
import time
import h5py
import numpy as np
import torch
from smoke_task2_pretrained import ROOT,make_config,DynamicVLAPolicy,load_dynamicvla,AutoConfig,AutoTokenizer,upstream_dataset
from finetune_task2_development import batch_from_indices,valid_flow_mse,checkpoint
from capstone_vla.cohort_training import wrap_delta_euler,read_split_arrays,train_normalization,CyclingFrameSampler
from capstone_vla.prediction_metrics import absolute_from_delta,action_errors
DATA=ROOT/'data/derived/task2_multiepisode_20261026'

class VerbatimInstruction:
    @staticmethod
    def generate_instruction(value):
        obj=json.loads(value) if isinstance(value,str) else value
        return obj['capstone_instruction']

def load_dataset(split):
    if split not in ['train','validation']:raise ValueError('reserved test access disabled')
    root=DATA/split;identity=json.loads((root/'capstone_cohort.json').read_text())
    if identity['split']!=split:raise ValueError('cohort split mismatch')
    upstream_dataset.InstructionGenerator=VerbatimInstruction
    ds=upstream_dataset.LeRobotDataset('capstone-local/task2-'+split,root=root,split='train',
       episodes=list(range(len(identity['episode_order']))),pin_memory=False,delta_action=True,
       required_features=['observation.state','action','observation.images.opst_cam','observation.images.wrist_cam'],
       delta_timestamps={'observation':[-2,0],'action':list(range(20))},
       image_transforms=upstream_dataset.ImageTransforms(img_size=(360,480)))
    return ds

def batch(ds,indices,cache,plan,device):
    values=batch_from_indices(ds,indices,cache,plan['cache_frames'],'cpu')
    # Retain XYZ and gripper exactly; only remove periodic full-turn Euler differences.
    values['action']=torch.from_numpy(wrap_delta_euler(values['action'].numpy()))
    return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in values.items()}

def probe_indices(rows,all_episodes,plan):
    chosen=set(range(len(rows))) if all_episodes else set(np.linspace(0,len(rows)-1,8,dtype=int).tolist())
    output=[];offset=0
    for i,row in enumerate(rows):
        if i in chosen:output.extend(offset+min(len(row['action'])-1,int(f*len(row['action']))) for f in plan['probe_fractions'])
        offset+=len(row['action'])
    return output

def flow_probe(policy,ds,indices,plan,cache):
    policy.eval();values=[]
    with torch.no_grad():
        for i in indices:
            item=batch(ds,[i],cache,plan,'cuda:0')
            for t in plan['probe_times']:
                noise=torch.randn((1,20,32),device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(plan['probe_seed']+i))
                _,details=policy(item,noise=noise,time=torch.tensor([t],device='cuda:0'))
                value=float(valid_flow_mse(details['losses_after_forward'],item['action_is_pad']))
                if not np.isfinite(value):raise RuntimeError('nonfinite validation/probe loss')
                values.append(value)
    return float(np.mean(values))

def validation_actions(policy,ds,rows,plan,cache,archive,phase):
    policy.eval();predictions=[];state=np.concatenate([r['state'] for r in rows]);target=np.concatenate([r['action'] for r in rows]);n=len(state)
    with torch.no_grad():
        for first in range(0,n,plan['action_eval_batch_size']):
            indices=list(range(first,min(n,first+plan['action_eval_batch_size'])))
            item=batch(ds,indices,cache,plan,'cuda:0')
            inputs={k:v for k,v in item.items() if k=='task' or (k.startswith('observation.') and not k.endswith('_is_pad'))}
            noise=torch.stack([torch.randn((20,32),device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(60000+i)) for i in indices])
            policy.reset();delta=policy.predict_action_chunk(inputs,noise=noise).cpu().numpy()
            predictions.append(absolute_from_delta(delta,state[indices])[:,0,:])
            if first==0 or (first//plan['action_eval_batch_size']+1)%40==0:
                print('VALIDATION_ACTIONS '+phase+' frames='+str(first+len(indices))+'/'+str(n),flush=True)
    pred=np.concatenate(predictions)
    metrics,*_=action_errors(pred[:,None,:],target[:,None,:],np.zeros((n,1),dtype=bool))
    detail=[];offset=0
    for row in rows:
        end=offset+len(row['action']);m,*_=action_errors(pred[offset:end,None,:],target[offset:end,None,:],np.zeros((end-offset,1),dtype=bool))
        events=(np.flatnonzero(np.diff(row['action'][:,6]>0)!=0)+1).tolist()
        detail.append({'episode':row['episode'],'first_action':m['first_action'],
          'gripper_transition_checks':[{'local_frame':i,'target':float(target[offset+i,6]),'prediction':float(pred[offset+i,6]),
              'correct_sign':bool((pred[offset+i,6]>0)==(target[offset+i,6]>0))} for i in events]})
        offset=end
    with h5py.File(archive,'a') as f:
        group=f.create_group(phase);group.create_dataset('predicted_first_absolute',data=pred);group.create_dataset('target_absolute',data=target)
    return {'all_validation_frames_first_action':metrics['first_action'],'per_episode':detail,
       'scope':'recorded observations, one fixed per-frame noise draw; not a simulation rollout'}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    parser.add_argument('--steps',type=int);args=parser.parse_args()
    plan=json.loads((ROOT/'configs/task2_multiepisode_finetune.json').read_text())
    if args.steps is not None:plan['steps']=args.steps
    if plan['steps']<1 or plan['batch_size']<1 or plan['test_evaluation_enabled']:raise ValueError('invalid training configuration')
    pin=json.loads((ROOT/'configs/dynamicvla_model_smoke.json').read_text());raw=json.loads((ROOT/'configs/dynamicvla_dom_pretrained.json').read_text())
    export=json.loads((DATA/'export_report.json').read_text())
    if export['status']!='MULTIEPISODE_LEROBOT_EXPORT_PASS' or export['test_reserved_exported']:raise ValueError('cohort not ready')
    manifest_path=ROOT/'data/derived/task2_expansion_inventory_20261025/split_manifest.json'
    manifest=json.loads(manifest_path.read_text());manifest_sha=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    for split in ['train','validation']:
        identity=json.loads((DATA/split/'capstone_cohort.json').read_text())
        expected=[row['episode'] for row in manifest['selected'][split]]
        if identity['manifest_sha256']!=manifest_sha or identity['episode_order']!=expected:
            raise ValueError('export does not match frozen split manifest: '+split)
    train_rows=read_split_arrays(DATA/'train');val_rows=read_split_arrays(DATA/'validation')
    if len(train_rows)!=40 or len(val_rows)!=8 or {r['episode'] for r in train_rows}&{r['episode'] for r in val_rows}:raise ValueError('split mismatch')
    stats,norm_audit=train_normalization(DATA/'train')
    train=load_dataset('train');validation=load_dataset('validation')
    tc=OrderedDict();vc=OrderedDict()
    sample=batch(train,[0,len(train)-1],tc,plan,'cpu')
    assert sample['action'].shape==(2,20,7) and sample['action_is_pad'][1].sum()==19
    if not np.isfinite(sample['action'].numpy()).all():raise ValueError('invalid wrapped deltas')
    head=subprocess.check_output(['git','-C',str(ROOT/'third_party/DynamicVLA'),'rev-parse','HEAD'],text=True).strip()
    if head!=pin['upstream_commit']:raise ValueError('source revision changed')
    weights=ROOT/'data/models/dynamicvla_dom'/pin['revision']/'model.safetensors'
    backbone=ROOT/'data/models/smollm2_metadata'/pin['backbone_revision']
    for path in [weights,backbone/'config.json']:
        if not path.is_file():raise FileNotFoundError(path)
    print('CPU_PREFLIGHT_OK train=40/11128 validation=8/1848 train_only_norm=true wrapped_targets='+str(norm_audit['raw_euler_delta_over_pi_count'])+'/'+str(norm_audit['valid_chunk_targets']),flush=True)
    print('PLAN steps='+str(plan['steps'])+' batch='+str(plan['batch_size'])+' source=official_pretrained reserved_test=disabled',flush=True)
    if not args.run:print('GPU_NOT_STARTED; add --run',flush=True);return
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(8);torch.manual_seed(plan['seed']);random.seed(plan['seed']);np.random.seed(plan['seed'])
    out=DATA/'training_runs'/datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True,exist_ok=False)
    print('OUTPUT',out,flush=True)
    metadata={'plan':plan,'initialization':'official DOM checkpoint, not previous single-episode weights','upstream_commit':head,
       'model_revision':pin['revision'],'frozen_manifest_sha256':manifest_sha,'normalization_fit':norm_audit,'label_version':'wrapped_euler_delta_v1',
       'train_episodes':[r['episode'] for r in train_rows],'validation_episodes':[r['episode'] for r in val_rows],
       'reserved_test_opened':False,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'run_config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (out/'normalization_fit_audit.json').write_text(json.dumps(norm_audit,indent=2)+'\n')
    ac=AutoConfig.from_pretrained;at=AutoTokenizer.from_pretrained
    def local_config(name,*a,**kw):
        if name!=pin['backbone_repo']:raise ValueError('unexpected dependency')
        return ac(str(backbone),*a,local_files_only=True,**kw)
    def local_tokenizer(name,*a,**kw):
        if name!=pin['backbone_repo']:raise ValueError('unexpected tokenizer')
        return at(str(backbone),*a,local_files_only=True,**kw)
    with patch.object(AutoConfig,'from_pretrained',side_effect=local_config),patch.object(AutoTokenizer,'from_pretrained',side_effect=local_tokenizer):
        policy=DynamicVLAPolicy(make_config(raw),dataset_stats=stats)
    load_dynamicvla(policy,weights,device='cpu',checkpoint_keys_mapping='model._orig_mod.//model.');policy.to('cuda:0')
    parameters=[p for p in policy.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=plan['learning_rate'],betas=(.9,.95),eps=1e-8,weight_decay=1e-10)
    probes={'train':probe_indices(train_rows,False,plan),'validation':probe_indices(val_rows,True,plan)}
    history=[]
    def probe(step):
        record={'step':step,'train_fixed_flow_mse':flow_probe(policy,train,probes['train'],plan,tc),
                 'validation_fixed_flow_mse':flow_probe(policy,validation,probes['validation'],plan,vc)}
        history.append(record);(out/'validation_history.json').write_text(json.dumps(history,indent=2)+'\n')
        print('PAIRED_FLOW_PROBE',json.dumps(record),flush=True)
    probe(0)
    initial_actions=validation_actions(policy,validation,val_rows,plan,vc,out/'validation_actions.hdf5','initial')
    (out/'validation_actions_initial.json').write_text(json.dumps(initial_actions,indent=2)+'\n')
    first_projection=policy.model.action_out_proj.weight.detach().clone()
    sampler=CyclingFrameSampler(len(train),plan['seed']);seen=set();start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    with (out/'training.jsonl').open('x') as log:
        for step in range(1,plan['steps']+1):
            policy.train();policy.model.vlm_with_expert.get_vlm_model().eval()
            indices=sampler.draw(plan['batch_size']);seen.update(indices)
            item=batch(train,indices,tc,plan,'cuda:0');optimizer.zero_grad(set_to_none=True)
            loss,details=policy(item)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
            valid=float(valid_flow_mse(details['losses_after_forward'],item['action_is_pad']).detach())
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(parameters,plan['gradient_clip_norm'],error_if_nonfinite=True);optimizer.step()
            elapsed=time.monotonic()-start
            row={'step':step,'training_indices':indices,'official_loss':float(loss.detach()),'valid_7d_flow_mse':valid,
                 'gradient_norm_before_clip':float(grad),'elapsed_s':elapsed}
            log.write(json.dumps(row)+'\n');log.flush()
            if step==1 or step%plan['log_every']==0:
                print('TRAIN step='+str(step)+'/'+str(plan['steps'])+' loss='+format(row['official_loss'],'.5f')+
                      ' elapsed='+format(elapsed,'.0f')+'s eta='+format(elapsed/step*(plan['steps']-step),'.0f')+'s',flush=True)
            if step%plan['validation_every']==0 or step==plan['steps']:
                optimizer.zero_grad(set_to_none=True);probe(step)
                destination=out/('checkpoint_final' if step==plan['steps'] else f'checkpoint_step_{step:06d}')
                checkpoint(policy,destination,stats,{**metadata,'completed_steps':step})
    optimizer.zero_grad(set_to_none=True)
    final_actions=validation_actions(policy,validation,val_rows,plan,vc,out/'validation_actions.hdf5','final')
    change=float(torch.linalg.vector_norm(policy.model.action_out_proj.weight.detach()-first_projection))
    if not np.isfinite(change) or change<=0:raise RuntimeError('no finite action-head update')
    best=min(history,key=lambda row:row['validation_fixed_flow_mse'])
    report={'status':'MULTIEPISODE_FINETUNE_COMPLETED','optimizer_steps':plan['steps'],'batch_size':plan['batch_size'],
       'train_episodes':40,'validation_episodes':8,'distinct_train_frame_indices_seen':len(seen),
       'normalization_fitted_on_train_only':True,'wrapped_euler_targets':norm_audit,
       'validation_flow_history':history,'best_validation_flow_step':best['step'],
       'validation_actions_initial':initial_actions,'validation_actions_final':final_actions,
       'checkpoint_final':str(out/'checkpoint_final'),'action_projection_l2_change':change,
       'peak_pytorch_allocated_bytes':torch.cuda.max_memory_allocated(),'reserved_test_evaluated':False,
       'simulation_rollout':False,'task_success_verified':False,
       'comparison_limits':'same-task validation only; new training budget, normalizers and periodic-label adaptation differ from earlier single-episode diagnostic'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('MULTIEPISODE_FINETUNE_COMPLETED report='+str(out/'report.json'),flush=True)

if __name__=='__main__':main()
