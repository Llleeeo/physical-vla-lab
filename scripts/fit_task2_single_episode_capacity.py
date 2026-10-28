#!/usr/bin/env python3
"""Fit only TRAIN001 from the existing40-episode checkpoint; no simulation/test data."""
import argparse,hashlib,json,random,subprocess,time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo
import h5py,numpy as np,torch
from safetensors.torch import load_model
from finetune_task2_cohort import ROOT,DATA,load_dataset,read_split_arrays,batch,CyclingFrameSampler,make_config,DynamicVLAPolicy,AutoConfig,AutoTokenizer
from finetune_task2_development import checkpoint,valid_flow_mse
from diagnose_task2_train_teacher import queue_stream,compare
from task2_policy_server import prepare_input
from capstone_vla.live_policy_protocol import CAMERAS
from capstone_vla.prediction_metrics import absolute_from_delta
from capstone_vla.franka import FrankaCartesianAdapter
BASE=DATA/'training_runs/20261026_185108_323227'
DIAGNOSTIC=BASE/'train_teacher_diagnostics/20261027_103424_626252'

def episode_window(rows,record,episode):
    if episode!='traj_00001' or episode not in record['train_episodes'] or episode in record['validation_episodes']:raise ValueError('requires frozen TRAIN001')
    if [r['episode'] for r in rows].count(episode)!=1:raise ValueError('episode missing or duplicated')
    offset=0
    for row in rows:
        if row['episode']==episode:return offset,row
        offset+=len(row['action'])
    raise ValueError('episode missing')

def restricted_indices(local_indices,length,offset):
    local=np.asarray(local_indices)
    if local.ndim!=1 or local.dtype.kind not in 'iu' or not len(local) or (local<0).any() or (local>=length).any():raise ValueError('sample outside diagnostic episode')
    return (local+offset).tolist()

def closure_runs(stream):
    closed=np.asarray(stream)[:,6]<0
    edges=np.diff(np.r_[False,closed,False].astype(int));start=np.flatnonzero(edges==1);end=np.flatnonzero(edges==-1)
    return [{'start_frame':int(i),'end_exclusive':int(j),'duration_s':float((j-i)/60)} for i,j in zip(start,end)]

def generated_metrics(stream,target):
    steps=np.linalg.norm(np.diff(stream[:,:3],axis=0),axis=1)*1000;boundary=np.arange(1,len(stream))%4==0
    return {'all_frames':compare(stream,target),'approach_frames_80_131':compare(stream[80:132],target[80:132]),'closed_command_runs':closure_runs(stream),'step_p95_mm':float(np.percentile(steps,95)),'chunk_boundary_step_mean_mm':float(steps[boundary].mean())}

def validate_plan(p,n):
    if p['episode']!='traj_00001' or any(p[k] is not False for k in ['validation_evaluation','test_evaluation','simulation_rollout']):raise ValueError('development scope changed')
    for key in ['steps','batch_size','cache_frames','log_every','probe_every']:
        if not isinstance(p[key],int) or p[key]<1:raise ValueError('invalid '+key)
    if not np.isfinite([p['learning_rate'],p['gradient_clip_norm']]).all() or min(p['learning_rate'],p['gradient_clip_norm'])<=0:raise ValueError('invalid optimization settings')
    if p['normalization']!='reuse_parent_train40_exact':raise ValueError('normalization must remain fixed')
    if not p['probe_frames'] or any(k<0 or k>=n for k in p['probe_frames']):raise ValueError('probe outside episode')
    if not p['probe_times'] or any(t<=0 or t>=1 for t in p['probe_times']):raise ValueError('invalid flow probe time')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/task2_single_episode_capacity.json');p.add_argument('--steps',type=int)
    g=p.add_mutually_exclusive_group();g.add_argument('--run',action='store_true');g.add_argument('--check-load',action='store_true');args=p.parse_args()
    plan=json.loads(args.config.read_text())
    if args.steps is not None:plan['steps']=args.steps
    record=json.loads((BASE/'run_config.json').read_text());manifest=ROOT/'data/derived/task2_expansion_inventory_20261025/split_manifest.json';digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(manifest)!=record['frozen_manifest_sha256']:raise ValueError('split changed')
    identity=json.loads((DATA/'train/capstone_cohort.json').read_text())
    if identity['manifest_sha256']!=record['frozen_manifest_sha256'] or identity['episode_order']!=record['train_episodes']:raise ValueError('TRAIN export changed')
    rows=read_split_arrays(DATA/'train');offset,row=episode_window(rows,record,plan['episode']);n=len(row['action']);validate_plan(plan,n)
    if n!=232:raise ValueError('TRAIN001 length changed')
    target,state=row['action'],row['state'];parent=BASE/'checkpoint_final';stats_raw=json.loads((parent/'capstone_normalization.json').read_text());weights_hash=digest(parent/'model.safetensors')
    previous=json.loads((DIAGNOSTIC/'report.json').read_text())
    if previous['checkpoint_sha256']!=weights_hash or previous['seed']!=plan['prediction_seed']:raise ValueError('saved baseline checkpoint/noise mismatch')
    old_config=json.loads((Path(previous['live_run'])/'run_config.json').read_text());source=Path(old_config['episode'])/'replay.hdf5'
    if digest(source)!=old_config['sha256'][str(source.relative_to(ROOT))]:raise ValueError('source changed')
    robot=json.loads((ROOT/'configs/franka_action_interface.json').read_text());adapter=FrankaCartesianAdapter(robot['urdf'])
    with h5py.File(source) as f:measured=np.stack([adapter.state(q[:7]) for q in f['obs/joint_pos'][:]])
    np.testing.assert_allclose(state,measured,rtol=0,atol=1e-6)
    ds=load_dataset('train');cache=OrderedDict()
    def training_batch(local,device):return batch(ds,restricted_indices(local,n,offset),cache,plan,device)
    for local in [[0,1],[109,112,113,114,120],[231]]:
        item=training_batch(local,'cpu');idx=np.asarray(local)
        np.testing.assert_allclose(item['observation.state'][:,-1,:6].numpy(),state[idx],rtol=0,atol=1e-6)
        # Evaluate SO(3) equivalence as wrapped Euler labels may differ by2pi.
        restored=absolute_from_delta(item['action'].numpy(),state[idx])
        from scipy.spatial.transform import Rotation
        expected=target[np.minimum(idx[:,None]+np.arange(20),n-1)]
        np.testing.assert_allclose(restored[...,:3],expected[...,:3],rtol=0,atol=1e-6)
        np.testing.assert_array_equal(restored[...,6],expected[...,6])
        angle=(Rotation.from_euler('xyz',restored[...,3:6].reshape(-1,3)).inv()*Rotation.from_euler('xyz',expected[...,3:6].reshape(-1,3))).magnitude()
        if angle.max()>1e-5:raise ValueError('rotation target mismatch')
        if local==[231] and int(item['action_is_pad'].sum())!=19:raise ValueError('padding mismatch')
        if item['task']!=[old_config['instruction']]*len(local):raise ValueError('language mismatch')
    anchors=list(range(0,n,4))
    with h5py.File(DIAGNOSTIC/'predictions.hdf5') as f:
        baseline=f['original_render_live_preprocessing/chunks'][:];np.testing.assert_array_equal(f['target_absolute'][:],target);np.testing.assert_array_equal(f['anchor_frames'][:],anchors)
    camera=Path(next(x for x in json.loads((ROOT/'data/derived/task2_expansion_inventory_20261025/render_sources.json').read_text())['sources'] if x['episode']==plan['episode'])['expected_camera_archive'])
    pin=json.loads((ROOT/'configs/dynamicvla_model_smoke.json').read_text())
    head=subprocess.check_output(['git','-C',str(ROOT/'third_party/DynamicVLA'),'rev-parse','HEAD'],text=True).strip()
    if head!=pin['upstream_commit']:raise ValueError('upstream revision changed')
    print(f"CPU_PREFLIGHT_OK episode={plan['episode']} frames={n} steps={plan['steps']} batch={plan['batch_size']} initialization=existing_multiepisode_checkpoint normalization=unchanged validation=false test=false simulation=false",flush=True)
    if not (args.run or args.check_load):print('GPU_NOT_STARTED; add --run');return
    torch.set_num_threads(8);torch.manual_seed(plan['seed']);np.random.seed(plan['seed']);random.seed(plan['seed'])
    cfg=make_config(json.loads((parent/'config.json').read_text()));stats={k:{key:torch.tensor(v,dtype=torch.float32) for key,v in values.items()} for k,values in stats_raw.items()}
    backbone=ROOT/'data/models/smollm2_metadata'/pin['backbone_revision'];ac,at=AutoConfig.from_pretrained,AutoTokenizer.from_pretrained
    def local_config(name,*a,**kw):
        if name!=pin['backbone_repo']:raise ValueError('unexpected dependency')
        return ac(str(backbone),*a,local_files_only=True,**kw)
    def local_tokenizer(name,*a,**kw):
        if name!=pin['backbone_repo']:raise ValueError('unexpected tokenizer')
        return at(str(backbone),*a,local_files_only=True,**kw)
    with patch.object(AutoConfig,'from_pretrained',side_effect=local_config),patch.object(AutoTokenizer,'from_pretrained',side_effect=local_tokenizer):policy=DynamicVLAPolicy(cfg,dataset_stats=stats)
    load_model(policy,str(parent/'model.safetensors'),strict=True,device='cpu')
    def check_norms():
        for prefix,key in [('normalize_inputs.buffer_observation_state','observation.state'),('normalize_targets.buffer_action','action'),('unnormalize_outputs.buffer_action','action')]:
            for field in ['mean','std']:torch.testing.assert_close(policy.state_dict()[prefix+'.'+field].cpu(),stats[key][field],rtol=0,atol=0)
    check_norms()
    if args.check_load:print('CPU_STRICT_RELOAD_PASS normalization=exact GPU_NOT_STARTED');return
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    policy.to('cuda:0');parameters=[p for p in policy.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=plan['learning_rate'],betas=(.9,.95),eps=1e-8,weight_decay=1e-10)
    out=BASE/'single_episode_capacity_runs'/datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True,exist_ok=False);print('OUTPUT',out,flush=True)
    metadata={'purpose':plan['purpose'],'plan':plan,'train_episodes':[plan['episode']],'episode_global_offset':offset,'initial_checkpoint':str(parent),'initial_checkpoint_sha256':weights_hash,'initial_optimizer':'fresh AdamW, not resumed optimizer state','normalization_source':str(parent/'capstone_normalization.json'),'normalization_sha256':digest(parent/'capstone_normalization.json'),'source_replay_sha256':digest(source),'frozen_manifest_sha256':record['frozen_manifest_sha256'],'upstream_commit':head,'script_sha256':digest(Path(__file__)),'trainable_parameters':sum(p.numel() for p in parameters),'frozen_parameters':sum(p.numel() for p in policy.parameters() if not p.requires_grad),'vision_connector_text_frozen':True,'training_data_only':'RGB/proprioception/instruction and demonstration action labels; no object ground truth','validation_opened':False,'reserved_test_opened':False,'generalization_claim':False,'simulation_rollout':False}
    (out/'run_config.json').write_text(json.dumps(metadata,indent=2)+'\n');history=[]
    def flow_probe():
        policy.eval();values=[]
        with torch.inference_mode():
            for k in plan['probe_frames']:
                item=training_batch([k],'cuda:0')
                for t in plan['probe_times']:
                    noise=torch.randn((1,20,32),device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(20261027+k))
                    _,details=policy(item,noise=noise,time=torch.tensor([t],device='cuda:0'))
                    values.append(float(valid_flow_mse(details['losses_after_forward'],item['action_is_pad'])))
        return float(np.mean(values))
    with h5py.File(out/'generated_actions.hdf5','x') as saved,h5py.File(camera) as rgb:
        saved.create_dataset('target_absolute',data=target);saved.create_dataset('anchor_frames',data=anchors)
        def probe(step,reuse=None):
            policy.eval();chunks=[]
            if reuse is not None:chunks=reuse
            else:
                for j,k in enumerate(anchors):
                    ids=[max(0,k-2),k];images={name:np.stack([rgb[name+'_rgb'][i] for i in ids]) for name in CAMERAS};inputs=prepare_input(state[ids],images,old_config['instruction'],'cuda:0')
                    noise=torch.randn((1,20,32),device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(plan['prediction_seed']+k));policy.reset()
                    with torch.inference_mode():delta=policy.predict_action_chunk(inputs,noise=noise).cpu().numpy()
                    chunks.append(absolute_from_delta(delta,state[k:k+1])[0])
                    if j%20==0:print(f'GENERATE step={step} anchors={j+1}/{len(anchors)}',flush=True)
            chunks=np.asarray(chunks);stream=queue_stream(chunks,anchors,n);group=saved.create_group(f'step_{step:06d}');group.create_dataset('absolute_chunks',data=chunks);group.create_dataset('queued_actions',data=stream);group.attrs['baseline_reused']=reuse is not None;saved.flush()
            row={'step':step,'fixed_flow_mse':flow_probe(),'generated_actions':generated_metrics(stream,target),'source':'reused prior exact-checkpoint/noise baseline' if reuse is not None else 'new generation on original expert RGB through live preprocessing'}
            history.append(row);(out/'fit_history.json').write_text(json.dumps(history,indent=2,allow_nan=False)+'\n');print('FIT_PROBE '+json.dumps(row),flush=True)
        probe(0,baseline);sampler=CyclingFrameSampler(n,plan['seed']);seen=set();start=time.monotonic();first_projection=policy.model.action_out_proj.weight.detach().clone()
        with (out/'training.jsonl').open('x') as log:
            for step in range(1,plan['steps']+1):
                policy.train();policy.model.vlm_with_expert.get_vlm_model().eval();indices=sampler.draw(plan['batch_size']);seen.update(indices)
                item=training_batch(indices,'cuda:0');optimizer.zero_grad(set_to_none=True);loss,details=policy(item)
                if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
                valid=float(valid_flow_mse(details['losses_after_forward'],item['action_is_pad']).detach());loss.backward();gradient=torch.nn.utils.clip_grad_norm_(parameters,plan['gradient_clip_norm'],error_if_nonfinite=True);optimizer.step()
                elapsed=time.monotonic()-start;row={'step':step,'local_frame_indices':indices,'official_loss':float(loss.detach()),'valid7_flow_mse':valid,'gradient_norm_before_clip':float(gradient),'elapsed_s':elapsed};log.write(json.dumps(row)+'\n');log.flush()
                if step==1 or step%plan['log_every']==0:print(f"FIT step={step}/{plan['steps']} loss={row['official_loss']:.5f} valid7={valid:.5f} elapsed={elapsed:.0f}s eta={elapsed/step*(plan['steps']-step):.0f}s",flush=True)
                if step%plan['probe_every']==0 or step==plan['steps']:
                    optimizer.zero_grad(set_to_none=True);probe(step)
        optimizer.zero_grad(set_to_none=True);check_norms();change=float(torch.linalg.vector_norm(policy.model.action_out_proj.weight.detach()-first_projection))
        if not np.isfinite(change) or change<=0:raise RuntimeError('no finite action projection update')
        checkpoint(policy,out/'checkpoint_final',stats,{**metadata,'completed_steps':plan['steps']})
        if json.loads((out/'checkpoint_final/capstone_normalization.json').read_text())!=stats_raw:raise RuntimeError('normalization sidecar changed')
    if digest(parent/'model.safetensors')!=weights_hash:raise RuntimeError('initial checkpoint changed')
    report={'status':'SINGLE_EPISODE_CAPACITY_COMPLETED','scope':'TRAIN001 fit diagnostic, not evaluation or grasp success','optimizer_steps':plan['steps'],'distinct_frame_indices_seen':len(seen),'current_observation_sample_passes':plan['steps']*plan['batch_size']/n,'normalization_unchanged':True,'initial':history[0],'final':history[-1],'history':history,'checkpoint_final':str(out/'checkpoint_final'),'action_projection_l2_change':change,'validation_opened':False,'reserved_test_opened':False,'generalization_claim':False,'simulation_rollout':False,'task_success_verified':False}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('SINGLE_EPISODE_CAPACITY_COMPLETED report='+str(out/'report.json'),flush=True)
if __name__=='__main__':main()
