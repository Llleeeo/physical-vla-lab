#!/usr/bin/env python3
"""No-training sampling controls on the eight development validation episodes."""
import argparse,json,hashlib
from pathlib import Path
from collections import OrderedDict
from datetime import datetime
from zoneinfo import ZoneInfo
from unittest.mock import patch
import h5py,numpy as np,torch
from safetensors.torch import load_model
from finetune_task2_cohort import ROOT,DATA,load_dataset,read_split_arrays,make_config,DynamicVLAPolicy,AutoConfig,AutoTokenizer
from finetune_task2_development import batch_from_indices
from capstone_vla.prediction_metrics import absolute_from_delta,action_errors
from diagnose_task2_sampling import noise_summary,residual_changes
DEFAULT=DATA/'training_runs/20261026_185108_323227'
SEEDS=list(range(9000,9008))

def write_saved_report(out,checkpoint,rows,target,previous,selected,recovered=False):
 """Recompute the full diagnostic from saved predictions, without model inference."""
 with h5py.File(out/'predictions.hdf5','r') as f:
  indices=f['repeated_indices'][:];seeds=f['seeds'][:]
  repeated=f['repeated_first_absolute'][:];shared=f['shared_noise_first_absolute'][:]
 np.testing.assert_array_equal(indices,[r['index'] for r in selected]);np.testing.assert_array_equal(seeds,SEEDS)
 if repeated.shape!=(len(selected),len(SEEDS),7) or shared.shape!=target.shape or previous.shape!=target.shape:
  raise ValueError('incomplete or mismatched saved predictions')
 if not np.isfinite(repeated).all() or not np.isfinite(shared).all():raise ValueError('nonfinite saved predictions')
 details=[{**row,**noise_summary(pred,target[row['index']])} for row,pred in zip(selected,repeated)]
 metrics,*_=action_errors(shared[:,None],target[:,None],np.zeros((len(target),1),bool))
 per_episode=[];offset=0
 for row in rows:
  end=offset+len(row['action']);sl=slice(offset,end)
  per_episode.append({'episode':row['episode'],'independent_noise':residual_changes(previous[sl],target[sl]),'shared_noise':residual_changes(shared[sl],target[sl])});offset=end
 result={'status':'COHORT_SAMPLING_DIAGNOSTIC_COMPLETED','checkpoint':str(checkpoint),'repeated_observation_results':details,
 'shared_noise_first_action_metrics':metrics['first_action'],'per_episode':per_episode,'shared_noise_seed':9000,
 'training':False,'reserved_test_opened':False,'simulation_rollout':False,'recovered_from_saved_predictions':recovered,
 'archive_sha256':hashlib.sha256((out/'predictions.hdf5').read_bytes()).hexdigest(),
 'limits':'Eight noise draws per selected observation and one shared seed are diagnostic controls, not a deployment recommendation or exhaustive causal attribution. Sampling-control validation is development data.'}
 text=json.dumps(result,indent=2,allow_nan=False)+'\n'
 destination=out/'report.json'
 if destination.exists():raise FileExistsError(destination)
 temporary=out/'report.json.tmp';temporary.write_text(text);temporary.replace(destination)
 print('COHORT_SAMPLING_DIAGNOSTIC_COMPLETED report='+str(destination),flush=True)
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--check-load',action='store_true',help='strict checkpoint reload on CPU, without inference');p.add_argument('--run-dir',type=Path,default=DEFAULT);p.add_argument('--recover',type=Path,help='rebuild missing report from existing predictions on CPU');a=p.parse_args()
 if a.recover and (a.run or a.check_load):p.error('--recover is CPU-only; do not combine with --run/--check-load')
 checkpoint=a.run_dir/'checkpoint_final';run=json.loads((a.run_dir/'run_config.json').read_text())
 pin=json.loads((ROOT/'configs/dynamicvla_model_smoke.json').read_text())
 cfg=make_config(json.loads((checkpoint/'config.json').read_text()))
 stats={k:{n:np.asarray(v,dtype=np.float32) for n,v in values.items()} for k,values in json.loads((checkpoint/'capstone_normalization.json').read_text()).items()}
 identity=json.loads((DATA/'validation/capstone_cohort.json').read_text());rows=read_split_arrays(DATA/'validation')
 manifest=ROOT/'data/derived/task2_expansion_inventory_20261025/split_manifest.json'
 if identity['manifest_sha256']!=run['frozen_manifest_sha256'] or hashlib.sha256(manifest.read_bytes()).hexdigest()!=identity['manifest_sha256']:raise ValueError('frozen cohort changed')
 if [r['episode'] for r in rows]!=run['validation_episodes']:raise ValueError('wrong validation episodes')
 state=np.concatenate([r['state'] for r in rows]);target=np.concatenate([r['action'] for r in rows]);selected=[];offset=0
 for row in rows:
  n=len(row['action']);events=np.flatnonzero(np.diff(row['action'][:,6]>0))+1
  local=set([n//4,n//2,3*n//4])
  for event in events:
   local.update(i for i in [event-1,event,event+1] if 0<=i<n)
  selected.extend({'episode':row['episode'],'local_frame':int(i),'index':int(offset+i)} for i in sorted(local));offset+=n
 # Catch metadata serialization errors before any GPU workload.
 json.dumps(selected,allow_nan=False)
 with h5py.File(a.run_dir/'validation_actions.hdf5') as f:
  previous=f['final/predicted_first_absolute'][:];np.testing.assert_array_equal(f['final/target_absolute'][:],target)
 if a.recover:
  if a.recover.resolve().parent!=(a.run_dir/'sampling_diagnostics').resolve():raise ValueError('recovery archive must belong to selected training run')
  write_saved_report(a.recover,checkpoint,rows,target,previous,selected,recovered=True);return
 ds=load_dataset('validation');cache=OrderedDict()
 sample=batch_from_indices(ds,[0,len(ds)-1],cache,8,'cpu')
 assert sample['action'].shape==(2,20,7) and len(ds)==len(target)
 print('CPU_PREFLIGHT_OK validation_episodes=8 frames='+str(len(ds))+' repeated_observations='+str(len(selected))+' draws=8 training=false test_access=false',flush=True)
 if not a.run and not a.check_load:print('GPU_NOT_STARTED; add --run');return
 if a.run and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 torch.set_num_threads(8);torch.manual_seed(0)
 if a.run:
  out=a.run_dir/'sampling_diagnostics'/datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True,exist_ok=False);print('OUTPUT',out,flush=True)
 backbone=ROOT/'data/models/smollm2_metadata'/pin['backbone_revision'];ac=AutoConfig.from_pretrained;at=AutoTokenizer.from_pretrained
 def config(name,*args,**kw):
  if name!=pin['backbone_repo']:raise ValueError('unexpected dependency')
  return ac(str(backbone),*args,local_files_only=True,**kw)
 def tokenizer(name,*args,**kw):
  if name!=pin['backbone_repo']:raise ValueError('unexpected tokenizer')
  return at(str(backbone),*args,local_files_only=True,**kw)
 # Tensor stats take upstream's clone path: CPU normalizers must not share numpy storage.
 tensor_stats={k:{name:torch.tensor(value) for name,value in fields.items()} for k,fields in stats.items()}
 with patch.object(AutoConfig,'from_pretrained',side_effect=config),patch.object(AutoTokenizer,'from_pretrained',side_effect=tokenizer):policy=DynamicVLAPolicy(cfg,dataset_stats=tensor_stats)
 load_model(policy,str(checkpoint/'model.safetensors'),strict=True,device='cpu')
 for prefix,key in [('normalize_inputs.buffer_observation_state','observation.state'),('normalize_targets.buffer_action','action'),('unnormalize_outputs.buffer_action','action')]:
  for field in ['mean','std']:
   np.testing.assert_array_equal(policy.state_dict()[prefix+'.'+field].numpy(),stats[key][field])
 print('STRICT_CHECKPOINT_LOAD_OK normalization=exact device=cpu',flush=True)
 if not a.run:print('GPU_NOT_STARTED; CPU_LOAD_CHECK_PASS',flush=True);return
 policy.to('cuda:0').eval()
 def predict(indices,seeds):
  item=batch_from_indices(ds,indices,cache,64,'cuda:0');observed=item['observation.state'][:,-1,:6].cpu().numpy()
  np.testing.assert_allclose(observed,state[indices],rtol=0,atol=1e-6)
  inputs={k:v for k,v in item.items() if k=='task' or (k.startswith('observation.') and not k.endswith('_is_pad'))}
  noise=torch.stack([torch.randn((20,32),device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(seed)) for seed in seeds])
  policy.reset();delta=policy.predict_action_chunk(inputs,noise=noise).cpu().numpy()
  return absolute_from_delta(delta,observed)[:,0,:]
 repeated=[];details=[];shared=[]
 with torch.inference_mode():
  for j,row in enumerate(selected):
   pred=predict([row['index']]*8,SEEDS);repeated.append(pred);details.append({**row,**noise_summary(pred,target[row['index']])})
   if j%10==0:print('REPEATED_OBSERVATIONS '+str(j+1)+'/'+str(len(selected)),flush=True)
  for first in range(0,len(ds),8):
   indices=list(range(first,min(first+8,len(ds))));shared.append(predict(indices,[9000]*len(indices)))
   if first%320==0:print('SHARED_NOISE '+str(first+len(indices))+'/'+str(len(ds)),flush=True)
 shared=np.concatenate(shared)
 with h5py.File(out/'predictions.hdf5','x') as f:
  f.create_dataset('repeated_indices',data=[r['index'] for r in selected]);f.create_dataset('seeds',data=SEEDS);f.create_dataset('repeated_first_absolute',data=np.stack(repeated));f.create_dataset('shared_noise_first_absolute',data=shared)
 write_saved_report(out,checkpoint,rows,target,previous,selected)
if __name__=='__main__':main()
