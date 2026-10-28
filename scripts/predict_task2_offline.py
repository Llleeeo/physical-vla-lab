#!/usr/bin/env python3
"""Compare full generated actions on recorded Task2 observations; no robot."""
import argparse
from collections import OrderedDict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
from unittest.mock import patch
from zoneinfo import ZoneInfo
import h5py
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_model
from smoke_task2_pretrained import (ROOT,BASE,make_config,get_dataset,DynamicVLAPolicy,
    load_dynamicvla,AutoConfig,AutoTokenizer,upstream_dataset,PilotH5Dataset)
from finetune_task2_development import batch_from_indices
from capstone_vla.prediction_metrics import absolute_from_delta,action_errors

DEFAULT=BASE/'finetune_runs/20261025_213301_391756/checkpoint_final'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=DEFAULT)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.batch_size<1:raise ValueError('positive batch size required')
    checkpoint=args.checkpoint.resolve()
    pin=json.loads((ROOT/'configs/dynamicvla_model_smoke.json').read_text())
    config=json.loads((checkpoint/'config.json').read_text());cfg=make_config(config)
    record=json.loads((checkpoint/'capstone_run.json').read_text())
    if record['upstream_commit']!=pin['upstream_commit'] or record['completed_steps']!=200:
        raise ValueError('this diagnostic expects the audited 200-step checkpoint')
    stats={key:{name:np.asarray(value,dtype=np.float32) for name,value in item.items()}
           for key,item in json.loads((checkpoint/'capstone_normalization.json').read_text()).items()}
    trained_weights=checkpoint/'model.safetensors'
    baseline_weights=ROOT/'data/models/dynamicvla_dom'/pin['revision']/'model.safetensors'
    backbone=ROOT/'data/models/smollm2_metadata'/pin['backbone_revision']
    head=subprocess.check_output(['git','-C',str(ROOT/'third_party/DynamicVLA'),'rev-parse','HEAD'],text=True).strip()
    if head!=pin['upstream_commit']:raise ValueError('upstream revision changed')
    with safe_open(str(trained_weights),framework='pt',device='cpu') as handle:
        for prefix,key in [('normalize_inputs.buffer_observation_state','observation.state'),
                           ('normalize_targets.buffer_action','action'),('unnormalize_outputs.buffer_action','action')]:
            for field in ['mean','std']:
                np.testing.assert_array_equal(handle.get_tensor(prefix+'.'+field).numpy(),stats[key][field])
    ds=get_dataset();ds.image_transforms=upstream_dataset.ImageTransforms(img_size=(360,480))
    ref=PilotH5Dataset(BASE/'episode_000000.hdf5');n=len(ref)
    probe=batch_from_indices(ds,[112,113,114,231],OrderedDict(),8,'cpu')
    raw_state=probe['observation.state'][:,-1].numpy()
    restored=absolute_from_delta(probe['action'].numpy(),raw_state)
    for row,i in enumerate([112,113,114,231]):
        indices=np.clip(np.arange(i,i+20),0,n-1)
        np.testing.assert_allclose(restored[row],ref.actions[indices],atol=1e-6,rtol=0)
    print('CPU_PREFLIGHT_OK checkpoint_norms=exact frames=232 gripper_events=preserved delta_to_absolute=once',flush=True)
    if not args.run:print('GPU_NOT_STARTED; add --run',flush=True);return
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(8);torch.manual_seed(0)
    out=BASE/'offline_predictions'/datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    print('OUTPUT',out,flush=True)
    original_config=AutoConfig.from_pretrained;original_tokenizer=AutoTokenizer.from_pretrained
    def local_config(name,*a,**kw):
        if name!=pin['backbone_repo']:raise ValueError('unexpected dependency')
        return original_config(str(backbone),*a,local_files_only=True,**kw)
    def local_tokenizer(name,*a,**kw):
        if name!=pin['backbone_repo']:raise ValueError('unexpected tokenizer')
        return original_tokenizer(str(backbone),*a,local_files_only=True,**kw)
    with patch.object(AutoConfig,'from_pretrained',side_effect=local_config),patch.object(AutoTokenizer,'from_pretrained',side_effect=local_tokenizer):
        policy=DynamicVLAPolicy(cfg,dataset_stats=stats)
    all_indices=np.clip(np.arange(n)[:,None]+np.arange(20),0,n-1)
    padding=np.arange(n)[:,None]+np.arange(20)>=n
    target=ref.actions[all_indices].astype(np.float64)
    report={'purpose':'same_episode_recorded_observation_diagnostic','checkpoint':str(checkpoint),
            'upstream_commit':head,'heldout_evaluation':False,'simulation_rollout':False,'task_success_verified':False,
            'normalization':'same saved development normalizers for both models',
            'action_conversion':'generated unnormalized delta xyz/Euler + current observed raw state; gripper unchanged',
            'seed':'9000 + source frame; identical initial noise for both models',
            'num_flow_steps':cfg.num_steps,'fps':60,'batch_size':args.batch_size,'models':{}}
    with h5py.File(out/'predictions.hdf5','x') as archive:
        archive.attrs['heldout_evaluation']=False
        archive.create_dataset('target_absolute',data=target)
        archive.create_dataset('action_is_pad',data=padding)
        archive.create_dataset('timestamp_s',data=np.arange(n)/60.)
        for name,weights in [('pretrained',baseline_weights),('finetuned',trained_weights)]:
            policy.cpu()
            if name=='pretrained':
                load_dynamicvla(policy,weights,device='cpu',checkpoint_keys_mapping='model._orig_mod.//model.')
            else:
                # Restore ALL checkpoint tensors, including saved normalization buffers.
                load_model(policy,str(weights),strict=True,device='cpu')
            policy.to('cuda:0').eval();policy.reset()
            predictions=[];cache=OrderedDict();start=time.monotonic()
            with torch.inference_mode():
                for first in range(0,n,args.batch_size):
                    indices=list(range(first,min(first+args.batch_size,n)))
                    batch=batch_from_indices(ds,indices,cache,32,'cuda:0')
                    current=batch['observation.state'][:,-1,:6].detach().cpu().numpy().copy()
                    # Labels and padding are for the offline evaluator only.
                    inputs={k:v for k,v in batch.items() if k=='task' or
                            (k.startswith('observation.') and not k.endswith('_is_pad'))}
                    noise=torch.stack([torch.randn((20,32),device='cuda:0',
                           generator=torch.Generator(device='cuda:0').manual_seed(9000+i)) for i in indices])
                    delta=policy.predict_action_chunk(inputs,noise=noise).cpu().numpy()
                    if delta.shape!=(len(indices),20,7):raise ValueError('predicted chunk shape mismatch')
                    predictions.append(absolute_from_delta(delta,current))
                    if first==0 or (first//args.batch_size+1)%10==0 or first+len(indices)==n:
                        done=first+len(indices);elapsed=time.monotonic()-start
                        print('PREDICT '+name+' frames='+str(done)+'/232 elapsed='+format(elapsed,'.0f')+
                              's eta='+format(elapsed/done*(n-done),'.0f')+'s',flush=True)
            pred=np.concatenate(predictions)
            metrics,pos,angle,grip=action_errors(pred,target,padding)
            target_events=(np.flatnonzero(np.diff(target[:,0,6]>0)!=0)+1).tolist()
            predicted_events=(np.flatnonzero(np.diff(pred[:,0,6]>0)!=0)+1).tolist()
            metrics['target_gripper_transition_frames']=target_events
            metrics['first_action_predicted_gripper_transition_frames']=predicted_events
            metrics['gripper_event_window_108_118']=[{'frame':i,'target':float(target[i,0,6]),
                'predicted_raw':float(pred[i,0,6]),'sign_correct':bool(grip[i,0])} for i in range(108,119)]
            metrics['elapsed_s']=time.monotonic()-start
            report['models'][name]=metrics
            group=archive.create_group(name)
            group.create_dataset('predicted_absolute',data=pred)
            group.create_dataset('position_error_mm',data=pos)
            group.create_dataset('orientation_error_deg',data=angle)
            group.create_dataset('gripper_sign_correct',data=grip)
            archive.flush()
            print(name.upper()+'_FIRST_ACTION',json.dumps(metrics['first_action']),flush=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    subprocess.run([str(ROOT/'data/derived/hf_pilot_render_repo/.venv/bin/python'),
                    str(ROOT/'scripts/plot_task2_predictions.py'),str(out)],check=True)
    print('OFFLINE_ACTION_PREDICTION_COMPLETED report='+str(out/'report.json'),flush=True)

if __name__=='__main__':main()
