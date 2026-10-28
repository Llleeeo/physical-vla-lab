#!/usr/bin/env python3
"""Compare action-prefix execution on recorded validation observations, not rollout."""
import argparse,json,hashlib
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from unittest.mock import patch
import h5py,numpy as np,torch
from safetensors.torch import load_model
from finetune_task2_cohort import ROOT,DATA,load_dataset,read_split_arrays,make_config,DynamicVLAPolicy,AutoConfig,AutoTokenizer
from finetune_task2_development import batch_from_indices
from capstone_vla.prediction_metrics import absolute_from_delta,action_errors
from capstone_vla.chunk_execution import execute_prefix
from diagnose_task2_sampling import residual_changes
DEFAULT=DATA/'training_runs/20261026_185108_323227'
PREFIXES=[1,4,8,20]

def summarize(out,chunks,rows,state,target,previous):
 lengths=[len(row['action']) for row in rows];mask=np.zeros((len(target),1),bool);results=[]
 for k in PREFIXES:
  stream,anchor,horizon=execute_prefix(chunks,lengths,k)
  metric,*_=action_errors(stream[:,None],target[:,None],mask)
  hold=target.copy();hold[:,:6]=state[anchor]
  holdmetric,*_=action_errors(hold[:,None],target[:,None],mask)
  holdmetric=holdmetric['first_action'];holdmetric.pop('gripper_sign_accuracy')
  detail=[];offset=0
  for row,n in zip(rows,lengths):
   sl=slice(offset,offset+n);m,*_=action_errors(stream[sl,None],target[sl,None],np.zeros((n,1),bool))
   detail.append({'episode':row['episode'],'metrics':m['first_action'],'continuity':residual_changes(stream[sl],target[sl])});offset+=n
  results.append({'prefix':k,'observation_update_interval_ms':k/60*1000,'metrics':metric['first_action'],'hold_at_chunk_anchor_pose_baseline':holdmetric,'per_episode':detail})
  print('PREFIX',k,json.dumps(metric['first_action']),flush=True)
 result={'status':'ACTION_CHUNK_CHECK_COMPLETED','prefix_comparisons':results,
  'first_step_max_abs_difference_vs_previous':float(np.max(np.abs(chunks[:,0]-previous))),
  'scope':'recorded-observation prefix replay; anchor observations remain expert-recorded, not model-generated feedback',
  'noise':'60000 + global frame index, same convention as previous validation inference',
  'causality':'each chunk adds only its prediction-time raw state, once; queues reset at episode boundaries',
  'training':False,'reserved_test_opened':False,'simulation_rollout':False,
  'limits':'Offline accuracy cannot establish grasp success or optimal prefix. No real-time latency modeled; 20-step update interval is333ms at60Hz. Fixed image history remains[-2,0].'}
 text=json.dumps(result,indent=2,allow_nan=False)+'\n'
 if (out/'report.json').exists():raise FileExistsError(out/'report.json')
 (out/'report.json').write_text(text);print('ACTION_CHUNK_CHECK_COMPLETED report='+str(out/'report.json'),flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--run-dir',type=Path,default=DEFAULT);p.add_argument('--recover',type=Path);a=p.parse_args()
 if a.run and a.recover:p.error('choose --run or --recover')
 checkpoint=a.run_dir/'checkpoint_final';record=json.loads((a.run_dir/'run_config.json').read_text())
 identity=json.loads((DATA/'validation/capstone_cohort.json').read_text());manifest=ROOT/'data/derived/task2_expansion_inventory_20261025/split_manifest.json'
 if identity['manifest_sha256']!=record['frozen_manifest_sha256'] or hashlib.sha256(manifest.read_bytes()).hexdigest()!=identity['manifest_sha256']:raise ValueError('frozen manifest mismatch')
 rows=read_split_arrays(DATA/'validation')
 if [r['episode'] for r in rows]!=record['validation_episodes']:raise ValueError('wrong validation episodes')
 state=np.concatenate([row['state'] for row in rows]);target=np.concatenate([row['action'] for row in rows]);n=len(target)
 with h5py.File(a.run_dir/'validation_actions.hdf5') as f:
  previous=f['final/predicted_first_absolute'][:];np.testing.assert_array_equal(f['final/target_absolute'][:],target)
 if a.recover:
  if a.recover.resolve().parent!=(a.run_dir/'chunk_checks').resolve():raise ValueError('wrong recovery parent')
  with h5py.File(a.recover/'predictions.hdf5') as f:
   chunks=f['absolute_chunks'][:];np.testing.assert_array_equal(f['target_absolute'][:],target)
  if chunks.shape!=(n,20,7):raise ValueError('incomplete chunks')
  summarize(a.recover,chunks,rows,state,target,previous);return
 ds=load_dataset('validation');cache=OrderedDict();item=batch_from_indices(ds,[0,n-1],cache,64,'cpu')
 assert len(ds)==n and item['action'].shape==(2,20,7) and int(item['action_is_pad'][-1].sum())==19
 print('CPU_PREFLIGHT_OK frames='+str(n)+' chunk=20 prefixes=1,4,8,20 train=false test=false',flush=True)
 if not a.run:print('GPU_NOT_STARTED; add --run');return
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 torch.set_num_threads(8);torch.manual_seed(0)
 pin=json.loads((ROOT/'configs/dynamicvla_model_smoke.json').read_text());cfg=make_config(json.loads((checkpoint/'config.json').read_text()))
 stats={k:{name:torch.tensor(v,dtype=torch.float32) for name,v in fields.items()} for k,fields in json.loads((checkpoint/'capstone_normalization.json').read_text()).items()}
 backbone=ROOT/'data/models/smollm2_metadata'/pin['backbone_revision'];ac=AutoConfig.from_pretrained;at=AutoTokenizer.from_pretrained
 def config(name,*args,**kw):
  if name!=pin['backbone_repo']:raise ValueError('unexpected dependency')
  return ac(str(backbone),*args,local_files_only=True,**kw)
 def tokenizer(name,*args,**kw):
  if name!=pin['backbone_repo']:raise ValueError('unexpected tokenizer')
  return at(str(backbone),*args,local_files_only=True,**kw)
 with patch.object(AutoConfig,'from_pretrained',side_effect=config),patch.object(AutoTokenizer,'from_pretrained',side_effect=tokenizer):policy=DynamicVLAPolicy(cfg,dataset_stats=stats)
 load_model(policy,str(checkpoint/'model.safetensors'),strict=True,device='cpu')
 for prefix,key in [('normalize_inputs.buffer_observation_state','observation.state'),('normalize_targets.buffer_action','action'),('unnormalize_outputs.buffer_action','action')]:
  for field in ['mean','std']:torch.testing.assert_close(policy.state_dict()[prefix+'.'+field],stats[key][field],rtol=0,atol=0)
 policy.to('cuda:0').eval()
 out=a.run_dir/'chunk_checks'/datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True,exist_ok=False);print('OUTPUT',out,flush=True)
 chunks=[]
 with torch.inference_mode():
  for start in range(0,n,8):
   indices=list(range(start,min(start+8,n)));item=batch_from_indices(ds,indices,cache,64,'cuda:0')
   np.testing.assert_allclose(item['observation.state'][:,-1,:6].cpu().numpy(),state[indices],rtol=0,atol=1e-6)
   inputs={k:v for k,v in item.items() if k=='task' or (k.startswith('observation.') and not k.endswith('_is_pad'))}
   noise=torch.stack([torch.randn((20,32),device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(60000+i)) for i in indices])
   policy.reset();delta=policy.predict_action_chunk(inputs,noise=noise).cpu().numpy();chunks.append(absolute_from_delta(delta,state[indices]))
   if start%320==0:print('PREDICT_CHUNKS '+str(start+len(indices))+'/'+str(n),flush=True)
 chunks=np.concatenate(chunks)
 with h5py.File(out/'predictions.hdf5','x') as f:
  f.create_dataset('absolute_chunks',data=chunks);f.create_dataset('target_absolute',data=target);f.attrs['checkpoint']=str(checkpoint)
 summarize(out,chunks,rows,state,target,previous)
if __name__=='__main__':main()
