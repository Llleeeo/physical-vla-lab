#!/usr/bin/env python3
"""Change training objective only; capped 1000 steps, generated gates, no physics."""
import argparse
from collections import OrderedDict
from datetime import datetime
import json
from pathlib import Path
import sys,time
from zoneinfo import ZoneInfo
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_model
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from finetune_hgwind_development import build_policy,batch_from,valid_loss
from predict_hgwind_development import inspect_run,check_normalizers
from capstone_vla.ball_precision import CONFIG,precision_loss,metrics,accept
from capstone_vla.hf_archive import sha256
from capstone_vla.pilot_dataset import CAMERAS
PARENT=ROOT/'data/derived/hgwind_traj18_development_20261031/training_runs/20261031_133555_985430'
SEEDS=[90000,190000]


def objective(policy,batch):
    prepared=policy.prepare_action(policy.normalize_targets(policy.normalize_inputs(batch)))
    noise=policy.model._sample_noise(prepared.shape,prepared.device)
    times=policy.model._sample_time(len(prepared),prepared.device)
    captured=[]
    handle=policy.model.action_out_proj.register_forward_hook(lambda module,inputs,output:captured.append(output))
    try:official,detail=policy(batch,noise=noise,time=times)
    finally:handle.remove()
    if len(captured)!=1:raise ValueError('Expected exactly one flow projection')
    velocity=captured[0]
    torch.testing.assert_close((velocity-(noise-prepared)).square(),detail['losses_after_forward'],rtol=1e-5,atol=1e-6)
    std=policy.normalize_targets.buffer_action['std']
    loss,parts=precision_loss(velocity,prepared,noise,times,batch['action_is_pad'],std)
    if not torch.isfinite(loss):raise ValueError('Nonfinite precision objective')
    return loss,parts,official,valid_loss(detail['losses_after_forward'],batch['action_is_pad'])


def generated(policy,ds,cache,seed):
    policy.eval();stream=[]
    with torch.inference_mode():
        for frame in range(0,len(ds),4):
            batch=batch_from(ds,[frame],cache,'cuda:0')
            inputs={k:batch[k] for k in ['task','observation.state',*['observation.images.'+c for c in CAMERAS]]}
            noise=torch.randn((1,20,32),device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(seed+frame))
            policy.reset();a=policy.predict_action_chunk(inputs,noise=noise).cpu().numpy()[0]
            if a.shape!=(20,29) or not np.isfinite(a).all():raise ValueError('Invalid generated action')
            stream.extend(a[:min(4,len(ds)-frame)])
    return np.asarray(stream)


def cpu_math_check():
    actions=torch.zeros((2,4,32));actions[:,2:,0]=10 # true discontinuity must not be smoothed away
    noise=torch.ones_like(actions);times=torch.tensor([.3,.8]);pad=torch.tensor([[False]*4,[False,True,True,True]])
    std=torch.ones(29);v=(noise-actions).clone().requires_grad_()
    perfect,parts=precision_loss(v,actions,noise,times,pad,std)
    assert float(perfect)==0 and float(parts['expert_increment_residual'])==0
    perfect.backward();assert torch.count_nonzero(v.grad)==0
    bad=(noise-actions+1).clone().requires_grad_()
    loss,_=precision_loss(bad,actions,noise,times,pad,std);loss.backward()
    assert bad.grad.abs().sum()>0 and not bad.grad[...,29:].count_nonzero() and not bad.grad[pad].count_nonzero()
    assert not torch.cuda.is_initialized()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--check-load',action='store_true')
    p.add_argument('--steps',type=int,choices=[250,500,1000],default=1000);args=p.parse_args()
    if args.run and args.check_load:p.error('Choose run or CPU check')
    torch.set_num_threads(8);torch.manual_seed(31);cpu_math_check()
    ds,cfg,stats,pin,weights,_=inspect_run(PARENT);cache=OrderedDict();batch=batch_from(ds,[80],cache,'cpu')
    print('CPU_PRECISION_PREFLIGHT_PASS parent_lineage=3000 xyz_weight=4 jaw_weight=4 endpoint_scale_mm=10 temporal=expert_increment padding=verified',flush=True)
    if not(args.run or args.check_load):print('GPU_NOT_STARTED; --check-load verifies actual CPU forward/backward; --run trains');return
    policy=build_policy(cfg,stats,pin,weights,ROOT/'data/models/smollm2_metadata'/pin['backbone_revision'])
    load_model(policy,str(PARENT/'checkpoint_final/model.safetensors'),strict=True,device='cpu');check_normalizers(policy,stats)
    if args.check_load:
        policy.train();policy.model.vlm_with_expert.get_vlm_model().eval()
        loss,parts,official,valid=objective(policy,batch);loss.backward()
        assert torch.isfinite(policy.model.action_out_proj.weight.grad).all() and policy.model.action_out_proj.weight.grad.abs().sum()>0
        assert not torch.cuda.is_initialized()
        print('ACTUAL_CPU_FLOW_GRADIENT_PASS hook_matches_upstream=true strict_normalizers=true CUDA=false loss='+str(float(loss.detach())));return
    policy.to('cuda:0');cache.clear()
    out=Path(ds.directory)/'training_runs'/datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True,exist_ok=False)
    print('OUTPUT '+str(out),flush=True)
    parent=json.loads((PARENT/'run_config.json').read_text());record=dict(parent)
    if parent['selected_lineage_optimizer_steps']!=3000:raise ValueError('Unexpected parent lineage')
    for key in ['baseline_batch1_reference_verified','batch1_reference_sha256','baseline_batch1_vs_archived_batch4_max_abs']:record.pop(key,None)
    record.update(steps=args.steps,learning_rate=1e-5,parent_run=str(PARENT),parent_checkpoint_sha256=sha256(PARENT/'checkpoint_final/model.safetensors'),
        initialization='strict_parent_finetuned_weights',optimizer_restarted=True,precision_objective=CONFIG,
        optimizer_loss='weighted flow + physical endpoint Huber + demonstrated XYZ increment residual',
        parent_run_config_sha256=sha256(PARENT/'run_config.json'),probe_noise='seed+frame for each of probe_seeds',probe_seeds=SEEDS,selection='both seeds: approach/near errors -10%, increment error -20%; no worse full p95, orientation, gripper or modes; native rejects zero',
        total_lineage_optimizer_steps=3000+args.steps,selected_checkpoint_step=0,selected_lineage_optimizer_steps=3000)
    files=[Path(__file__),ROOT/'src/capstone_vla/ball_precision.py',ROOT/'scripts/predict_hgwind_development.py',ROOT/'src/capstone_vla/ball_prediction_metrics.py']
    record['source_hashes']={**parent['source_hashes'],**{str(x.relative_to(ROOT)):sha256(x) for x in files}}
    (out/'run_config.json').write_text(json.dumps(record,indent=2)+'\n')
    anchor=ds.states[0,21:28]
    def probe(step):
        result=[]
        for seed in SEEDS:
            stream=generated(policy,ds,cache,seed);np.save(out/f'generated_step{step:04d}_seed{seed}.npy',stream,allow_pickle=False)
            m=metrics(stream,ds.actions,anchor);result.append(m)
            print('GENERATED '+json.dumps(dict(step=step,seed=seed,approach_mm=m['approach_position_mean_mm'],near_mm=m['precontact_80_119_mean_mm'],increment_error_mm=m['precontact_increment_error_mm'],grip_errors=m['gripper_error_frames'],grip_transitions=m['gripper_transitions'])),flush=True)
        return result
    baseline=probe(0)
    reference=PARENT/'generated_step1000.npy'
    np.testing.assert_allclose(np.load(out/'generated_step0000_seed90000.npy'),np.load(reference),rtol=1e-4,atol=1e-5)
    record['baseline_generated_parent_reference_sha256']=sha256(reference)
    print('BASELINE_GENERATION_MATCHED parent_selected_predictions=true',flush=True)
    best=baseline;selected=0;checkpoint=out/'checkpoint_final';policy.save_pretrained(checkpoint)
    parameters=[v for v in policy.parameters() if v.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=1e-10)
    rng=np.random.default_rng(31);seen=set();probes=[];start=time.monotonic()
    with (out/'training.jsonl').open('x') as log:
        for step in range(1,args.steps+1):
            policy.train();policy.model.vlm_with_expert.get_vlm_model().eval()
            ids=rng.integers(0,len(ds),size=2);seen.update(map(int,ids));batch=batch_from(ds,ids,cache,'cuda:0')
            optimizer.zero_grad(set_to_none=True);loss,parts,official,valid=objective(policy,batch);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(parameters,10.,error_if_nonfinite=True)
            if float(norm)<=0:raise ValueError('Zero gradients')
            optimizer.step()
            row=dict(step=step,optimizer_step=True,frames=ids.tolist(),valid_29d_flow_mse=float(valid.detach()),official_32d_loss=float(official.detach()),gradient_norm_before_clip=float(norm),precision_total=float(loss.detach()),**{k:float(v.detach()) for k,v in parts.items()})
            log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
            if step==1 or step%50==0:print(f'PRECISION_TRAIN {step}/{args.steps} loss={row["precision_total"]:.5f} elapsed={time.monotonic()-start:.0f}s',flush=True)
            if step%250==0:
                candidate=probe(step);ok=accept(candidate,baseline,best)
                if ok:best=candidate;selected=step;policy.save_pretrained(checkpoint)
                probes.append(dict(step=step,accepted=ok,metrics=candidate))
                (out/'generated_probes.json').write_text(json.dumps(probes,indent=2)+'\n')
                print(f'PRECISION_SELECTION accepted={ok} selected_step={selected}',flush=True)
    record.update(selected_checkpoint_step=selected,selected_lineage_optimizer_steps=3000+selected)
    (out/'run_config.json').write_text(json.dumps(record,indent=2)+'\n')
    (checkpoint/'capstone_run.json').write_text(json.dumps(record,indent=2)+'\n')
    (checkpoint/'capstone_normalization.json').write_bytes((PARENT/'checkpoint_final/capstone_normalization.json').read_bytes())
    load_model(policy,str(checkpoint/'model.safetensors'),strict=True,device='cpu');check_normalizers(policy,stats)
    with safe_open(str(weights),framework='pt',device='cpu') as f:official_projection=f.get_tensor('model.action_out_proj.weight')
    change=float(torch.linalg.vector_norm(policy.model.action_out_proj.weight.detach().cpu()-official_projection))
    report=dict(status='BALL_DEVELOPMENT_FINETUNE_COMPLETED',optimizer_steps=args.steps,unique_training_frames=len(seen),
        checkpoint=str(checkpoint),action_projection_l2_change=change,selected_checkpoint_step=selected,selected_improved=selected>0,
        baseline_generated=baseline,selected_generated=best,probes=probes,heldout_evaluation=False,simulation_rollout=False,task_success_verified=False,
        scope='Single exposed training case; two seeds are repeated development probes, not independent data',
        next_step='Review generated metrics before deciding on physics' if selected else 'No candidate passed. Parent retained; do not launch another rollout.')
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    inspect_run(out)
    print('BALL_PRECISION_TRAINING_COMPLETED selected_step='+str(selected)+' report='+str(out/'report.json'),flush=True)


if __name__=='__main__':main()
