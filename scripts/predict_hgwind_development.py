#!/usr/bin/env python3
"""Compare generated29D chunks before/after ball fine-tuning; never runs physics."""
import argparse
from collections import OrderedDict
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo
import h5py
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_model
from finetune_hgwind_development import ROOT,make_config,build_policy,batch_from
from capstone_vla.hgwind_development_data import BallDevelopmentDataset,normalization,DESTINATION
from capstone_vla.ball_prediction_metrics import summarize_chunks
from capstone_vla.hf_archive import sha256
from capstone_vla.pilot_dataset import CAMERAS

DEFAULT=ROOT/DESTINATION/'training_runs/20261031_115841_081174'


def check_normalizers(policy,stats):
    for prefix,key in [('normalize_inputs.buffer_observation_state','observation.state'),
        ('normalize_targets.buffer_action','action'),('unnormalize_outputs.buffer_action','action')]:
        for name in ['mean','std']:
            torch.testing.assert_close(policy.state_dict()[prefix+'.'+name].cpu(),stats[key][name],rtol=0,atol=0)


def inspect_run(run):
    run=Path(run).resolve();checkpoint=run/'checkpoint_final'
    report=json.loads((run/'report.json').read_text());record=json.loads((run/'run_config.json').read_text())
    if report['status']!='BALL_DEVELOPMENT_FINETUNE_COMPLETED' or report['optimizer_steps']!=record['steps']:
        raise ValueError('Training not complete')
    if json.loads((checkpoint/'capstone_run.json').read_text())!=record:raise ValueError('Checkpoint provenance differs')
    log=[json.loads(row) for row in (run/'training.jsonl').read_text().splitlines()]
    if [x['step'] for x in log]!=list(range(1,record['steps']+1)) or not all(x['optimizer_step'] for x in log):
        raise ValueError('Missing optimizer steps')
    for row in log:
        if not np.isfinite([row[k] for k in ['valid_29d_flow_mse','official_32d_loss','gradient_norm_before_clip']]).all():
            raise ValueError('Nonfinite training history')
    sampled={int(i) for row in log for i in row['frames']}
    if len(sampled)!=report['unique_training_frames']:raise ValueError('Sample history mismatch')
    for name,digest in record['source_hashes'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Training implementation changed: '+name)
    ds=BallDevelopmentDataset(record['source_dataset'])
    if sha256(Path(record['source_dataset'])/'dataset.json')!=record['dataset_manifest_sha256']:
        raise ValueError('Dataset differs from checkpoint')
    pin=json.loads((ROOT/'configs/dynamicvla_model_smoke.json').read_text())
    head=subprocess.check_output(['git','-C',str(ROOT/'third_party/DynamicVLA'),'rev-parse','HEAD'],text=True).strip()
    if pin!=record['pretrained'] or head!=record['upstream_commit']:raise ValueError('Upstream/model identity changed')
    config=json.loads((checkpoint/'config.json').read_text());cfg=make_config(config)
    if cfg.use_delta_action or tuple(cfg.action_feature.shape)!=(29,) or tuple(cfg.input_features['observation.state'].shape)!=(30,):
        raise ValueError('Wrong absolute29D/state30 checkpoint')
    saved=json.loads((checkpoint/'capstone_normalization.json').read_text());computed=normalization(ds.states,ds.actions)
    stats={k:{n:torch.tensor(v,dtype=torch.float32) for n,v in fields.items()} for k,fields in saved.items()}
    for key,fields in computed.items():
        for name,value in fields.items():np.testing.assert_array_equal(stats[key][name].numpy(),value)
    with safe_open(str(checkpoint/'model.safetensors'),framework='pt',device='cpu') as f:
        for prefix,key in [('normalize_inputs.buffer_observation_state','observation.state'),
            ('normalize_targets.buffer_action','action'),('unnormalize_outputs.buffer_action','action')]:
            for name in ['mean','std']:
                torch.testing.assert_close(f.get_tensor(prefix+'.'+name),stats[key][name],rtol=0,atol=0)
        trained_projection=f.get_tensor('model.action_out_proj.weight')
    weights=ROOT/'data/models/dynamicvla_dom'/pin['revision']/'model.safetensors'
    if sha256(weights)!=record['weights_sha256']:raise ValueError('Pretrained weights changed')
    with safe_open(str(weights),framework='pt',device='cpu') as f:
        delta=float(torch.linalg.vector_norm(trained_projection-f.get_tensor('model.action_out_proj.weight')))
    if not np.isclose(delta,report['action_projection_l2_change'],atol=1.e-6,rtol=1.e-5):
        raise ValueError('Saved weights disagree with recorded update')
    return ds,cfg,stats,pin,weights,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,default=DEFAULT)
    p.add_argument('--run',action='store_true')
    p.add_argument('--check-load',action='store_true',help='Strict CPU checkpoint reload, no inference')
    p.add_argument('--batch-size',type=int,default=4)
    a=p.parse_args()
    if a.run and a.check_load:p.error('Choose --run or --check-load')
    if not 1<=a.batch_size<=16:p.error('batch size must be 1..16')
    torch.set_num_threads(8);torch.manual_seed(0)
    run=a.run_dir.resolve();checkpoint=run/'checkpoint_final'
    ds,cfg,stats,pin,weights,trained=inspect_run(run)
    check=batch_from(ds,[0,len(ds)-1],OrderedDict(),'cpu')
    if check['action'].shape!=(2,20,29) or int(check['action_is_pad'][-1].sum())!=19:
        raise ValueError('Prediction input boundary mismatch')
    print('CPU_PREFLIGHT_OK frames=269 checkpoint_norms=exact absolute29=true delta_conversion=none train=false physics=false',flush=True)
    if not (a.run or a.check_load):print('GPU_NOT_STARTED; use --run');return
    backbone=ROOT/'data/models/smollm2_metadata'/pin['backbone_revision']
    policy=build_policy(cfg,stats,pin,weights,backbone)
    if a.check_load:
        load_model(policy,str(checkpoint/'model.safetensors'),strict=True,device='cpu')
        check_normalizers(policy,stats)
        if torch.cuda.is_initialized():raise RuntimeError('Unexpected CUDA initialization')
        print('STRICT_BALL_CHECKPOINT_RELOAD_PASS normalization=exact CUDA=false',flush=True);return
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    out=run/'offline_predictions'/datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False);print('OUTPUT',out,flush=True)
    result={'purpose':'same_training_episode_generated_action_diagnostic','heldout_evaluation':False,
        'simulation_rollout':False,'task_success_verified':False,'action_format':'absolute29D; no state addition/subtraction',
        'noise':'90000+source_frame, identical per-frame noise for both models','chunk_size':20,'fps':60,
        'batch_size':a.batch_size,'num_flow_steps':cfg.num_steps,'models':{},
        'limits':'Official baseline has original pretrained channel semantics; both comparisons use the new development normalizers. This is adaptation debugging, not a zero-shot benchmark.',
        'source_hashes':{str(x):sha256(x) for x in [checkpoint/'config.json',checkpoint/'model.safetensors',
            checkpoint/'capstone_normalization.json',run/'report.json',Path(ds.directory)/'dataset.json',Path(__file__),
            ROOT/'src/capstone_vla/ball_prediction_metrics.py']}}
    with h5py.File(out/'predictions.hdf5','x') as archive:
        archive.create_dataset('target_absolute29',data=ds.actions)
        archive.create_dataset('timestamp_s',data=np.arange(len(ds))/60.)
        archive.create_dataset('action_is_pad',data=np.arange(len(ds))[:,None]+np.arange(20)>=len(ds))
        for name in ['pretrained','finetuned']:
            policy.cpu()
            if name=='finetuned':load_model(policy,str(checkpoint/'model.safetensors'),strict=True,device='cpu')
            check_normalizers(policy,stats);policy.to('cuda:0').eval()
            cache=OrderedDict();pred=[];started=time.monotonic()
            saved=archive.create_dataset(name+'/absolute_chunks',shape=(len(ds),20,29),dtype='float32')
            for start in range(0,len(ds),a.batch_size):
                indices=list(range(start,min(start+a.batch_size,len(ds))))
                batch=batch_from(ds,indices,cache,'cuda:0')
                inputs={key:batch[key] for key in ['task','observation.state',*['observation.images.'+c for c in CAMERAS]]}
                noise=torch.stack([torch.randn((20,32),device='cuda:0',generator=
                    torch.Generator(device='cuda:0').manual_seed(90000+i)) for i in indices])
                policy.reset()
                with torch.inference_mode():values=policy.predict_action_chunk(inputs,noise=noise).cpu().numpy()
                if values.shape!=(len(indices),20,29):raise ValueError('Wrong output dimensions')
                saved[start:start+len(indices)]=values;archive.flush();pred.append(values)
                if start==0 or start%40==0 or start+len(indices)==len(ds):
                    print('PREDICT',name,str(start+len(indices))+'/269',flush=True)
            metrics=summarize_chunks(np.concatenate(pred),ds.actions);metrics['elapsed_s']=time.monotonic()-started
            result['models'][name]=metrics
            first=metrics['recorded_observation_prefixes']['1']
            print(name.upper()+'_FIRST_ACTION',json.dumps({k:first[k] for k in ['position_error_mm','orientation_error_deg','decoder_rejected_frames']}),flush=True)
        archive.attrs['complete']=True
    result['prediction_archive_sha256']=sha256(out/'predictions.hdf5')
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('BALL_OFFLINE_PREDICTION_COMPLETED report='+str(out/'report.json'),flush=True)

if __name__=='__main__':main()
