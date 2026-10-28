#!/usr/bin/env python3
"""Isolated local DynamicVLA inference process; only RGB/state enter the policy."""
import argparse,json,time,traceback
from pathlib import Path
from unittest.mock import patch
import numpy as np,torch
from safetensors.torch import load_model
from smoke_task2_pretrained import ROOT,make_config,DynamicVLAPolicy,AutoConfig,AutoTokenizer,upstream_dataset
from capstone_vla.live_policy_protocol import CAMERAS,validate_observation,atomic_npz
from capstone_vla.prediction_metrics import absolute_from_delta

def prepare_input(states,images,instruction,device='cpu'):
 validate_observation(states,images)
 item={'observation.state':torch.tensor(states,dtype=torch.float32)}
 keys=[]
 for name in CAMERAS:
  key='observation.images.'+name;keys.append(key);item[key]=torch.from_numpy(images[name].copy()).permute(0,3,1,2).float()/255.
 item=upstream_dataset.ImageTransforms(img_size=(360,480))(item,keys)
 # Upstream prepare_images merges history and channels with view(); materialize its required layout.
 return {**{k:v.unsqueeze(0).to(device).contiguous() for k,v in item.items()},'task':[instruction]}

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--check-load',action='store_true');a=p.parse_args();config=json.loads(a.config.read_text());ipc=Path(config['output'])/'ipc'
 try:
  torch.set_num_threads(8);torch.manual_seed(config['seed']);checkpoint=Path(config['checkpoint'])
  pin=json.loads((ROOT/'configs/dynamicvla_model_smoke.json').read_text());backbone=ROOT/'data/models/smollm2_metadata'/pin['backbone_revision']
  cfg=make_config(json.loads((checkpoint/'config.json').read_text()));stats={k:{n:torch.tensor(v,dtype=torch.float32) for n,v in fields.items()} for k,fields in json.loads((checkpoint/'capstone_normalization.json').read_text()).items()}
  ac=AutoConfig.from_pretrained;at=AutoTokenizer.from_pretrained
  def local_config(name,*args,**kw):
   if name!=pin['backbone_repo']:raise ValueError('unexpected backbone')
   return ac(str(backbone),*args,local_files_only=True,**kw)
  def local_tokenizer(name,*args,**kw):
   if name!=pin['backbone_repo']:raise ValueError('unexpected tokenizer')
   return at(str(backbone),*args,local_files_only=True,**kw)
  with patch.object(AutoConfig,'from_pretrained',side_effect=local_config),patch.object(AutoTokenizer,'from_pretrained',side_effect=local_tokenizer):policy=DynamicVLAPolicy(cfg,dataset_stats=stats)
  load_model(policy,str(checkpoint/'model.safetensors'),strict=True,device='cpu')
  for prefix,key in [('normalize_inputs.buffer_observation_state','observation.state'),('normalize_targets.buffer_action','action'),('unnormalize_outputs.buffer_action','action')]:
   for field in ['mean','std']:torch.testing.assert_close(policy.state_dict()[prefix+'.'+field],stats[key][field],rtol=0,atol=0)
  if a.check_load:print('CPU_STRICT_POLICY_SERVER_LOAD_PASS normalization=exact GPU_NOT_STARTED',flush=True);return
  policy.to('cuda:0').eval();(ipc/'ready').write_text('strict checkpoint and normalization verified\n');processed=set()
  while not (ipc/'stop').exists():
   requests=sorted(ipc.glob('request_*.npz'))
   for request in requests:
    if request.name in processed:continue
    with np.load(request,allow_pickle=False) as f:frame=int(f['frame']);states=f['states'].copy();images={name:f[name].copy() for name in CAMERAS}
    inputs=prepare_input(states,images,config['instruction'],'cuda:0');noise=torch.randn((1,20,32),device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(config['seed']+frame))
    policy.reset();torch.cuda.synchronize();start=time.perf_counter()
    with torch.inference_mode():delta=policy.predict_action_chunk(inputs,noise=noise).cpu().numpy()
    elapsed=time.perf_counter()-start;absolute=absolute_from_delta(delta,states[-1:])[0]
    atomic_npz(ipc/f'response_{frame:06d}.npz',frame=np.asarray(frame,dtype=np.int64),absolute=absolute,inference_s=np.asarray(elapsed))
    processed.add(request.name);print('POLICY_RESPONSE frame='+str(frame)+' inference_s='+format(elapsed,'.3f'),flush=True)
   time.sleep(.02)
 except Exception:
  (ipc/'server_error.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
