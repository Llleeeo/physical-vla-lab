#!/usr/bin/env python3
"""Fixed-observation/multiple-noise and shared-noise diagnostics, no fitting."""
import argparse
from collections import OrderedDict
from datetime import datetime
import json
from pathlib import Path
import subprocess
from unittest.mock import patch
from zoneinfo import ZoneInfo
import h5py
import numpy as np
import torch
from safetensors.torch import load_model
from predict_task2_offline import (ROOT,BASE,DEFAULT,make_config,get_dataset,DynamicVLAPolicy,
 AutoConfig,AutoTokenizer,upstream_dataset,PilotH5Dataset,batch_from_indices,absolute_from_delta,action_errors)

FRAMES=[60,108,111,112,113,114,115,160,225]
SEEDS=list(range(9000,9008))
PREVIOUS=BASE/'offline_predictions/20261025_214141_802738/predictions.hdf5'

def noise_summary(predicted,target):
    pred=np.asarray(predicted,dtype=float)
    target=np.asarray(target,dtype=float)
    mean=pred[:,:3].mean(axis=0)
    variance=float(np.mean(np.sum((pred[:,:3]-mean)**2,axis=1)))*1e6
    bias=float(np.sum((mean-target[:3])**2))*1e6
    mse=float(np.mean(np.sum((pred[:,:3]-target[:3])**2,axis=1)))*1e6
    if not np.isclose(mse,bias+variance,rtol=1e-8,atol=1e-8):raise ValueError('bias/variance identity failed')
    return {'position_mse_mm2':mse,'position_mean_bias_mm':float(np.sqrt(bias)),
       'position_sample_spread_rms_mm':float(np.sqrt(variance)),
       'gripper_open_fraction':float(np.mean(pred[:,6]>0)),
       'gripper_sign_accuracy':float(np.mean((pred[:,6]>0)==(target[6]>0))),
       'raw_gripper_values':pred[:,6].tolist(),'note':'empirical 8-draw decomposition, not proof of causal source or estimated irreducible error'}

def residual_changes(pred,target):
    residual=pred[:,:3]-target[:,:3]
    values=np.linalg.norm(np.diff(residual,axis=0),axis=1)*1000
    return {'residual_step_mean_mm':float(values.mean()),'residual_step_p95_mm':float(np.percentile(values,95)),
            'gripper_switch_count':int(np.count_nonzero(np.diff(pred[:,6]>0)))}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    pin=json.loads((ROOT/'configs/dynamicvla_model_smoke.json').read_text())
    checkpoint=DEFAULT
    cfg=make_config(json.loads((checkpoint/'config.json').read_text()))
    stats={k:{name:np.asarray(v,dtype=np.float32) for name,v in item.items()}
       for k,item in json.loads((checkpoint/'capstone_normalization.json').read_text()).items()}
    backbone=ROOT/'data/models/smollm2_metadata'/pin['backbone_revision']
    head=subprocess.check_output(['git','-C',str(ROOT/'third_party/DynamicVLA'),'rev-parse','HEAD'],text=True).strip()
    if head!=pin['upstream_commit']:raise ValueError('upstream revision changed')
    ref=PilotH5Dataset(BASE/'episode_000000.hdf5')
    with h5py.File(PREVIOUS,'r') as f:
        previous=f['finetuned/predicted_absolute'][:,0,:]
        if len(previous)!=len(ref):raise ValueError('previous diagnostic has wrong length')
    ds=get_dataset();ds.image_transforms=upstream_dataset.ImageTransforms(img_size=(360,480))
    batch=batch_from_indices(ds,[112,113,114],OrderedDict(),8,'cpu')
    if batch['action'].shape!=(3,20,7):raise ValueError('invalid batch')
    print('CPU_PREFLIGHT_OK repeated_observations=9 noise_draws=8 shared_noise_frames=232 no_training=true',flush=True)
    if not args.run:print('GPU_NOT_STARTED; add --run',flush=True);return
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(8);torch.manual_seed(0)
    out=BASE/'sampling_diagnostics'/datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False);print('OUTPUT',out,flush=True)
    ac=AutoConfig.from_pretrained;at=AutoTokenizer.from_pretrained
    def config(name,*a,**kw):
        if name!=pin['backbone_repo']:raise ValueError('unexpected dependency')
        return ac(str(backbone),*a,local_files_only=True,**kw)
    def tokenizer(name,*a,**kw):
        if name!=pin['backbone_repo']:raise ValueError('unexpected tokenizer')
        return at(str(backbone),*a,local_files_only=True,**kw)
    with patch.object(AutoConfig,'from_pretrained',side_effect=config),patch.object(AutoTokenizer,'from_pretrained',side_effect=tokenizer):
        policy=DynamicVLAPolicy(cfg,dataset_stats=stats)
    load_model(policy,str(checkpoint/'model.safetensors'),strict=True,device='cpu')
    policy.to('cuda:0').eval();cache=OrderedDict()
    def predict(frames,seeds):
        batch=batch_from_indices(ds,frames,cache,32,'cuda:0')
        state=batch['observation.state'][:,-1,:6].cpu().numpy().copy()
        inputs={k:v for k,v in batch.items() if k=='task' or (k.startswith('observation.') and not k.endswith('_is_pad'))}
        noise=torch.stack([torch.randn((20,32),device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(s)) for s in seeds])
        policy.reset()
        delta=policy.predict_action_chunk(inputs,noise=noise).cpu().numpy()
        return absolute_from_delta(delta,state)
    rows=[];repeated=[];shared=[]
    with torch.inference_mode():
        for frame in FRAMES:
            chunks=predict([frame]*len(SEEDS),SEEDS);first=chunks[:,0]
            rows.append({'frame':frame,'target_gripper':float(ref.actions[frame,6]),**noise_summary(first,ref.actions[frame])})
            repeated.append(first)
            print('REPEATED_OBSERVATION frame='+str(frame)+' open_fraction='+str(rows[-1]['gripper_open_fraction']),flush=True)
        for first in range(0,len(ref),4):
            frames=list(range(first,min(first+4,len(ref))))
            shared.append(predict(frames,[9000]*len(frames)))
            if first%40==0:print('SHARED_NOISE frames='+str(first+len(frames))+'/232',flush=True)
    shared=np.concatenate(shared)
    indices=np.clip(np.arange(len(ref))[:,None]+np.arange(20),0,len(ref)-1)
    pad=np.arange(len(ref))[:,None]+np.arange(20)>=len(ref)
    metrics,*_=action_errors(shared,ref.actions[indices],pad)
    report={'purpose':'development_sampling_sensitivity','checkpoint':str(checkpoint),'frames':FRAMES,'seeds':SEEDS,
      'repeated_observation_results':rows,'shared_noise_action_metrics':metrics,
      'independent_noise_residual_changes':residual_changes(previous,ref.actions),
      'shared_noise_residual_changes':residual_changes(shared[:,0],ref.actions),
      'shared_noise_gripper_at_112_114':[float(v) for v in shared[112:115,0,6]],
      'shared_noise_seed':9000,'previous_independent_noise_file':str(PREVIOUS),
      'training_started':False,'simulation_rollout':False,'heldout_evaluation':False,
      'interpretation':'One shared random draw is a diagnostic control, not a deployment recommendation. Spread/bias from 8 samples is not a complete attribution of model/data errors.'}
    with h5py.File(out/'sampling_predictions.hdf5','x') as f:
        f.create_dataset('repeated_frames',data=FRAMES);f.create_dataset('seeds',data=SEEDS)
        f.create_dataset('repeated_first_absolute',data=np.stack(repeated));f.create_dataset('shared_noise_absolute',data=shared)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('SAMPLING_DIAGNOSTIC_COMPLETED report='+str(out/'report.json'),flush=True)

if __name__=='__main__':main()
