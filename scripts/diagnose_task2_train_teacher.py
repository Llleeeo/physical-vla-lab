#!/usr/bin/env python3
"""TRAIN001 recorded-input vs saved closed-loop diagnosis; never simulates or trains."""
import argparse, hashlib, json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from safetensors.torch import load_model
from finetune_task2_cohort import ROOT, DATA, load_dataset, read_split_arrays, make_config, DynamicVLAPolicy, AutoConfig, AutoTokenizer
from finetune_task2_development import batch_from_indices
from task2_policy_server import prepare_input
from capstone_vla.franka import FrankaCartesianAdapter
from capstone_vla.live_policy_protocol import CAMERAS, binary_action
from capstone_vla.prediction_metrics import absolute_from_delta, action_errors
BASE=DATA/'training_runs/20261026_185108_323227'
DEFAULT=BASE/'seed_batches/20261027_002943_851190/seed_60000'

def fingerprint(inputs):
    result={}
    for k,v in inputs.items():
        if isinstance(v,torch.Tensor):
            arr=v.detach().cpu().contiguous().numpy()
            result[k]=hashlib.sha256(arr.tobytes()).hexdigest()
        else:result[k]=v
    return result

def queue_stream(chunks,anchors,length,prefix=4):
    if list(anchors)!=list(range(0,length,prefix)):raise ValueError('incomplete or shifted prediction anchors')
    values=np.asarray(chunks)
    if values.shape!=(len(anchors),20,7) or not np.isfinite(values).all():raise ValueError('invalid chunks')
    out=np.empty((length,7))
    for i,k in enumerate(anchors):
        count=min(prefix,length-k)
        out[k:k+count]=values[i,:count]
    out[:,6]=np.where(out[:,6]>0,1.,-1.)
    return out


def prediction_difference(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or a.shape!=(20,7):raise ValueError('expected two20x7 chunks')
    angles=np.rad2deg((Rotation.from_euler('xyz',a[:,3:6]).inv()*Rotation.from_euler('xyz',b[:,3:6])).magnitude())
    return {'max_position_difference_mm':float(np.linalg.norm(a[:,:3]-b[:,:3],axis=1).max()*1000),
            'max_orientation_difference_deg':float(angles.max()),
            'max_raw_gripper_difference':float(np.abs(a[:,6]-b[:,6]).max()),
            'gripper_sign_disagreements':int(np.count_nonzero((a[:,6]>0)!=(b[:,6]>0)))}

def compare(pred,target):
    metrics=action_errors(pred[:,None,:],target[:,None,:],np.zeros((len(pred),1),bool))[0]['first_action']
    closed=np.flatnonzero(pred[:,6]<0)
    metrics['first_close_frame']=int(closed[0]) if len(closed) else None
    metrics['switch_frames']=(np.flatnonzero(np.diff(pred[:,6]>0))+1).tolist()
    return metrics

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--live-run',type=Path,default=DEFAULT)
    args=parser.parse_args()
    live=args.live_run.resolve();config=json.loads((live/'run_config.json').read_text())
    if config.get('episode_role')!='train-debug' or Path(config['episode']).name!='traj_00001' or config['prefix']!=4 or config['frames']!=232:raise ValueError('requires completed full TRAIN001 prefix4 diagnostic')
    record=json.loads((BASE/'run_config.json').read_text())
    manifest=ROOT/'data/derived/task2_expansion_inventory_20261025/split_manifest.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=record['frozen_manifest_sha256']:raise ValueError('frozen split changed')
    if 'traj_00001' not in record['train_episodes'] or 'traj_00001' in record['validation_episodes']:raise ValueError('not a TRAIN debug episode')
    checkpoint=Path(config['checkpoint'])
    if checkpoint.resolve()!=(BASE/'checkpoint_final').resolve():raise ValueError('unexpected checkpoint')
    rows=read_split_arrays(DATA/'train');offset=0
    for row in rows:
        if row['episode']=='traj_00001':break
        offset+=len(row['action'])
    else:raise ValueError('TRAIN001 missing')
    state,target=row['state'],row['action'];n=len(target)
    if n!=232:raise ValueError('episode length changed')
    source=Path(config['episode'])/'replay.hdf5'
    source_rel=str(source.relative_to(ROOT))
    if hashlib.sha256(source.read_bytes()).hexdigest()!=config['sha256'][source_rel]:raise ValueError('source changed since rollout')
    robot=json.loads((ROOT/'configs/franka_action_interface.json').read_text());fk=FrankaCartesianAdapter(robot['urdf'])
    with h5py.File(source) as f:
        measured=np.stack([fk.state(q[:7]) for q in f['obs/joint_pos'][:]])
    np.testing.assert_allclose(state,measured,rtol=0,atol=1e-6)
    sources=json.loads((ROOT/'data/derived/task2_expansion_inventory_20261025/render_sources.json').read_text())['sources']
    camera=Path(next(x for x in sources if x['episode']=='traj_00001')['expected_camera_archive'])
    anchors=list(range(0,n,4));ds=load_dataset('train');cache=OrderedDict()
    def loader(k,device):
        item=batch_from_indices(ds,[offset+k],cache,8,device)
        inputs={key:(v.contiguous() if isinstance(v,torch.Tensor) else v) for key,v in item.items() if key=='task' or (key.startswith('observation.') and not key.endswith('_is_pad'))}
        np.testing.assert_allclose(inputs['observation.state'][0].cpu().numpy(),state[[max(0,k-2),k]],rtol=0,atol=1e-6)
        if inputs['task']!=[config['instruction']]:raise ValueError('language mismatch')
        return inputs,item
    for k in [0,112,128,228]:
        _,item=loader(k,'cpu')
        restored=absolute_from_delta(item['action'].numpy(),state[k:k+1])[0]
        expected=target[np.minimum(np.arange(k,k+20),n-1)]
        errors=compare(restored,expected)
        if errors['position_mean_mm']>.01 or errors['orientation_mean_deg']>.001 or errors['gripper_sign_accuracy']!=1:raise ValueError('training action target alignment failure')
    with h5py.File(live/'live_observations.hdf5') as f:
        if len(f['state'])!=n:raise ValueError('incomplete live archive')
        for k in anchors:
            with np.load(live/'ipc'/f'request_{k:06d}.npz',allow_pickle=False) as req:
                np.testing.assert_array_equal(req['states'],f['state'][[max(0,k-2),k]] if k else np.repeat(f['state'][:1],2,axis=0))
    with h5py.File(camera) as f:
        for name in CAMERAS:
            if f[name+'_rgb'].shape!=(n,480,640,3):raise ValueError('wrong recorded camera shape')
    print(f'CPU_PREFLIGHT_OK TRAIN001 frames={n} anchors={len(anchors)} labels=aligned history=-2,0 prefix=4 train=false simulation=false test=false',flush=True)
    if not args.run:print('GPU_NOT_STARTED; add --run');return
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(8);torch.manual_seed(config['seed'])
    pin=json.loads((ROOT/'configs/dynamicvla_model_smoke.json').read_text());backbone=ROOT/'data/models/smollm2_metadata'/pin['backbone_revision']
    cfg=make_config(json.loads((checkpoint/'config.json').read_text()))
    stats={k:{name:torch.tensor(v,dtype=torch.float32) for name,v in fields.items()} for k,fields in json.loads((checkpoint/'capstone_normalization.json').read_text()).items()}
    ac,at=AutoConfig.from_pretrained,AutoTokenizer.from_pretrained
    def local_config(name,*a,**kw):
        if name!=pin['backbone_repo']:raise ValueError('unexpected backbone')
        return ac(str(backbone),*a,local_files_only=True,**kw)
    def local_tokenizer(name,*a,**kw):
        if name!=pin['backbone_repo']:raise ValueError('unexpected tokenizer')
        return at(str(backbone),*a,local_files_only=True,**kw)
    with patch.object(AutoConfig,'from_pretrained',side_effect=local_config),patch.object(AutoTokenizer,'from_pretrained',side_effect=local_tokenizer):policy=DynamicVLAPolicy(cfg,dataset_stats=stats)
    load_model(policy,str(checkpoint/'model.safetensors'),strict=True,device='cpu')
    for prefix,key in [('normalize_inputs.buffer_observation_state','observation.state'),('normalize_targets.buffer_action','action'),('unnormalize_outputs.buffer_action','action')]:
        for name in ['mean','std']:torch.testing.assert_close(policy.state_dict()[prefix+'.'+name],stats[key][name],rtol=0,atol=0)
    policy.to('cuda:0').eval()
    out=BASE/'train_teacher_diagnostics'/datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True,exist_ok=False);print('OUTPUT',out,flush=True)
    result={'scope':'TRAIN001 recorded-input diagnosis; not robot rollout, training, held-out evaluation or grasp success','live_run':str(live),'seed':config['seed'],'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'sampling':'same local anchor seeds as saved rollout; batch_size=1','normalization':'strict reload and exact saved means/stds','repeats':[],'metrics':{},'preprocessing_differences':[]}
    def predict(inputs,k,raw_state):
        noise=torch.randn((1,20,32),device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(config['seed']+k))
        policy.reset()
        with torch.inference_mode():delta=policy.predict_action_chunk(inputs,noise=noise.clone()).cpu().numpy()
        return absolute_from_delta(delta,np.asarray(raw_state)[None])[0]
    chunks={'training_loader':[],'original_render_live_preprocessing':[],'saved_live_responses':[]}
    with h5py.File(camera) as rendered,h5py.File(live/'live_observations.hdf5') as observed,h5py.File(out/'predictions.hdf5','x') as archive:
        for index,k in enumerate(anchors):
            inputs,_=loader(k,'cuda:0');loaded=predict(inputs,k,state[k]);chunks['training_loader'].append(loaded)
            history=[max(0,k-2),k];rgb={name:np.stack([rendered[name+'_rgb'][i] for i in history]) for name in CAMERAS}
            formal_inputs=prepare_input(state[history],rgb,config['instruction'],'cuda:0')
            formal=predict(formal_inputs,k,state[k]);chunks['original_render_live_preprocessing'].append(formal)
            with np.load(live/'ipc'/f'response_{k:06d}.npz',allow_pickle=False) as saved:chunks['saved_live_responses'].append(saved['absolute'].copy())
            if k in [0,112,128,220]:
                differences={key:float((inputs[key]-formal_inputs[key]).abs().mean().item()) for key in inputs if key.startswith('observation.images.')}
                result['preprocessing_differences'].append({'frame':k,'mean_absolute_pixel_difference_after_transform':differences,'note':'LeRobot encoded video vs original RGB; difference is not necessarily a preprocessing bug'})
                with np.load(live/'ipc'/f'request_{k:06d}.npz',allow_pickle=False) as req:
                    rs=req['states'].copy();ri={name:req[name].copy() for name in CAMERAS}
                repeated_inputs=prepare_input(rs,ri,config['instruction'],'cuda:0');fp=fingerprint(repeated_inputs)
                predictions=[predict(repeated_inputs,k,rs[-1]) for _ in range(3)]
                with np.load(live/'ipc'/f'response_{k:06d}.npz',allow_pickle=False) as saved:previous=saved['absolute'].copy()
                rep={'frame':k,'repetitions':3,'preprocessed_input_sha256':fp,'identical_input_and_noise_max_abs_difference':float(max(np.max(np.abs(x-predictions[0])) for x in predictions)),'reloaded_vs_saved_response_max_abs_difference':float(np.max(np.abs(predictions[0]-previous))),'fresh_preprocessing_same_hash':fingerprint(prepare_input(rs,ri,config['instruction'],'cuda:0'))==fp,'units_note':'max component difference mixes metres, radians and raw gripper; not a position metric'}
                rep['same_input_physical_differences']=[prediction_difference(x,predictions[0]) for x in predictions[1:]];rep['reloaded_vs_saved_physical_difference']=prediction_difference(predictions[0],previous)
                result['repeats'].append(rep);archive.create_dataset(f'repeats/frame_{k:06d}',data=np.stack(predictions))
            if index%10==0:print(f'DIAGNOSE anchors={index+1}/{len(anchors)}',flush=True)
        archive.create_dataset('anchor_frames',data=anchors);archive.create_dataset('target_absolute',data=target)
        for name,values in chunks.items():
            stream=queue_stream(values,anchors,n);archive.create_dataset(name+'/chunks',data=np.stack(values));archive.create_dataset(name+'/queued_actions',data=stream)
            if name=='saved_live_responses':np.testing.assert_array_equal(stream,observed['action'][:])
            result['metrics'][name]={'all_frames':compare(stream,target),'pre_expert_first_close_frames_0_111':compare(stream[:112],target[:112]),'approach_window_frames_80_131':compare(stream[80:132],target[80:132])}
        hold=target.copy()
        for k in anchors:hold[k:k+4,:6]=state[k]
        m=compare(hold,target);m.pop('gripper_sign_accuracy');m.pop('first_close_frame');m.pop('switch_frames');result['hold_prediction_anchor_pose_baseline']=m
        live_state=observed['state'][:];deviation=np.linalg.norm(live_state[:,:3]-state[:,:3],axis=1)*1000
        result['live_measured_state_deviation_from_expert']={'initial_mm':float(deviation[0]),'first_over_10mm_frame':int(np.flatnonzero(deviation>10)[0]) if (deviation>10).any() else None,'mean_mm':float(deviation.mean()),'p95_mm':float(np.percentile(deviation,95))}
        result['expert_first_close_frame']=int(np.flatnonzero(target[:,6]<0)[0])
    result['limits']=['Recorded expert observations do not test recovery from off-trajectory states.','Comparing saved live actions to expert targets is diagnostic; the expert target need not be optimal in a diverged scene.','Four-step prefixes are scored without padded final targets.','Same-process repeats do not establish cross-process or renderer determinism.','Mixed-unit maximum component differences are not physical pose distances.']
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('TRAIN_TEACHER_DIAGNOSTIC_COMPLETED report='+str(out/'report.json'),flush=True)
    for name,value in result['metrics'].items():print(name,json.dumps(value['all_frames']),flush=True)
    print('REPEATS',json.dumps(result['repeats']),flush=True)
if __name__=='__main__':main()
