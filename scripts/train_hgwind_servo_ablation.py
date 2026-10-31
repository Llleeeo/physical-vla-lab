#!/usr/bin/env python3
"""Matched 500-step flow/control-aware objective pair; no runtime changes."""
import argparse
from collections import OrderedDict
from datetime import datetime
import hashlib,json,time
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_model
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from finetune_hgwind_development import build_policy,batch_from,valid_loss
from predict_hgwind_development import inspect_run,check_normalizers
from capstone_vla.ball_servo_loss import auxiliary_loss, conditional_command, command_numpy, command_metrics, future_pinch, AUX_WEIGHT
from train_hgwind_prefix_ablation import generated as base_generated
from capstone_vla.ball_noise_replanning import SEEDS,assemble_prefix,stream_metrics
from capstone_vla.ball_sampling_diagnostic import predict_preserving_noise
from capstone_vla.hf_archive import sha256
from capstone_vla.pilot_dataset import CAMERAS
PARENT=ROOT/'data/derived/hgwind_traj18_development_20261031/training_runs/20261031_133555_985430'
REFERENCE=PARENT/'noise_replanning_diagnostics/20261031_150546_311797'
STEPS=500


def objective(policy,batch,pinch,noise,times,factor):
    prepared=policy.prepare_action(policy.normalize_targets(policy.normalize_inputs(batch)))
    captured=[]
    handle=policy.model.action_out_proj.register_forward_hook(lambda module,inputs,output:captured.append(output))
    try:official,details=policy(batch,noise=noise,time=times)
    finally:handle.remove()
    if len(captured)!=1:raise ValueError('Expected one flow projection')
    velocity=captured[0]
    torch.testing.assert_close((velocity-(noise-prepared)).square(),details['losses_after_forward'],rtol=1e-5,atol=1e-6)
    raw=valid_loss(details['losses_after_forward'],batch['action_is_pad'])
    norm=policy.normalize_targets.buffer_action
    aux=auxiliary_loss(velocity,prepared,noise,times,batch['action_is_pad'],norm['mean'],norm['std'],pinch)
    total=raw+factor*aux
    if not torch.isfinite(total):raise ValueError('Nonfinite objective')
    return total,raw,aux,official


def cpu_math_check(ds):
    import importlib.util
    native=ROOT/'data/derived/hgwind_production_20261027/worker_bundle/data/derived/hgwind_current_baseline_20261027/repo/Scripts/expert_data/kinematic_servo.py'
    evidence=json.loads((ROOT/'results/summaries/2026-10-31-ball-upper-target-audit.json').read_text())
    if sha256(native)!=evidence['source_hashes'][str(native.relative_to(ROOT))]:raise ValueError('Native servo changed')
    spec=importlib.util.spec_from_file_location('frozen_servo_loss_check',native)
    servo=importlib.util.module_from_spec(spec);sys.modules[spec.name]=servo;spec.loader.exec_module(servo)
    teacher=torch.from_numpy(ds.actions).double();current=torch.from_numpy(ds.states[:,14:17]).double()
    rng=np.random.default_rng(52)
    # Cover original commands and perturbed requests, including saturation.
    from capstone_vla.ball_action_contract import decode_action
    for scale in (0.,.1,1.):
        xyz=teacher[:,:3]+torch.tensor(rng.normal(size=(269,3))*scale)
        ff=teacher[:,7:10]+torch.tensor(rng.normal(size=(269,3))*scale)
        actual=conditional_command(xyz,ff,current,teacher).numpy();expected=[]
        for j in range(269):
            inp=decode_action(ds.actions[j],j)['inputs']
            expected.append(servo.linear_task_velocity((xyz[j]-current[j]).numpy(),ff[j].numpy(),config=servo.KinematicServoConfig(position_gain=9.,maximum_linear_speed_mps=1.7),linear_velocity_reference_mps=inp['linear_velocity_reference'],maximum_relative_linear_speed_mps=inp['maximum_relative_linear_speed_mps']))
            np.testing.assert_allclose(command_numpy(xyz[j],ff[j],current[j],inp['linear_velocity_reference'],inp['maximum_relative_linear_speed_mps']),expected[-1],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
    # Correct discontinuities get zero loss; padding, tail and unrelated dimensions get zero gradient.
    a=torch.zeros(2,20,32,dtype=torch.float64);a[:,1:,:3]=.01
    n=torch.ones_like(a);t=torch.tensor([.3,.8],dtype=a.dtype);pad=torch.zeros(2,20,dtype=torch.bool);pad[1,1:]=True
    mean=torch.zeros(29,dtype=a.dtype);std=torch.ones_like(mean);q=torch.zeros(2,20,3,dtype=a.dtype)
    v=(n-a).clone().requires_grad_();loss=auxiliary_loss(v,a,n,t,pad,mean,std,q)
    assert loss.item()==0;loss.backward();assert not v.grad.count_nonzero()
    v=(n-a).clone();v[...,0]+=.02;v.requires_grad_();auxiliary_loss(v,a,n,t,pad,mean,std,q).backward()
    assert torch.isfinite(v.grad).all() and v.grad.abs().sum()>0
    assert not v.grad[pad].count_nonzero() and not v.grad[:,4:].count_nonzero()
    inactive=[i for i in range(32) if i not in (0,1,2,7,8,9)]
    assert not v.grad[...,inactive].count_nonzero()
    # Finite-difference check in smooth interior and saturated regions.
    for shift in (.02,1.):
        x=(teacher[20:23,:3]+shift).clone().requires_grad_();f=teacher[20:23,7:10].clone().requires_grad_()
        assert torch.autograd.gradcheck(lambda z,w:conditional_command(z,w,current[20:23],teacher[20:23]),(x,f),eps=1e-6,atol=1e-5,rtol=1e-3)
    assert not torch.cuda.is_initialized()


def generated(policy,ds,cache,out,label,expected=None):
    metrics=base_generated(policy,ds,cache,out,label,expected)
    arrays=np.load(out/(label+'_predictions.npz'),allow_pickle=False)['chunks']
    for si,m in enumerate(metrics):
        m.update(command_metrics(assemble_prefix(arrays[si]),ds.actions,ds.states))
        print('SERVO_PROBE '+json.dumps(dict(model=label,seed=m['seed'],**{k:m[k] for k in ['startup_command_mean_mps','near_command_mean_mps','full_command_mean_mps','command_rejected_frames']})),flush=True)
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group();group.add_argument('--run',action='store_true');group.add_argument('--check-load',action='store_true');args=parser.parse_args()
    torch.set_num_threads(8);torch.manual_seed(31)
    ds,cfg,stats,pin,weights,_=inspect_run(PARENT);cache=OrderedDict();parent=json.loads((PARENT/'run_config.json').read_text())
    cpu_math_check(ds)
    ref=json.loads((REFERENCE/'report.json').read_text());refcfg=json.loads((REFERENCE/'run_config.json').read_text())
    if sha256(REFERENCE/'predictions.npz')!=ref['predictions_sha256'] or sha256(REFERENCE/'run_config.json')!=ref['config_sha256']:raise ValueError('Prior evidence changed')
    for name,digest in refcfg['source_hashes'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Prior source changed: '+name)
    if cfg.num_steps!=10 or parent['selected_lineage_optimizer_steps']!=3000:raise ValueError('Wrong parent')
    schedule=np.random.default_rng(31).integers(0,len(ds),size=(STEPS,2))
    batch=batch_from(ds,[0,268],cache,'cpu')
    print('CPU_SERVO_ABLATION_PASS branches=uniform,servo steps_each=500 parent=3000 auxiliary_weight=0.1 scale_mps=0.1 first4=true native_parity=807 matched_batches_noise_time=true eval_seeds=8',flush=True)
    if not(args.run or args.check_load):print('GPU_NOT_STARTED; --run executes sequential training pair');return
    policy=build_policy(cfg,stats,pin,weights,ROOT/'data/models/smollm2_metadata'/pin['backbone_revision'])
    load_model(policy,str(PARENT/'checkpoint_final/model.safetensors'),strict=True,device='cpu');check_normalizers(policy,stats)
    if args.check_load:
        policy.train();policy.model.vlm_with_expert.get_vlm_model().eval()
        noise=policy.model._sample_noise((2,20,32),'cpu');times=policy.model._sample_time(2,'cpu')
        pinch=torch.from_numpy(future_pinch(ds.states,[0,268],batch['action_is_pad'].numpy()))
        loss,raw,aux,official=objective(policy,batch,pinch,noise,times,AUX_WEIGHT)
        grad_aux=torch.autograd.grad(aux,policy.model.action_out_proj.weight,retain_graph=True)[0]
        assert torch.isfinite(grad_aux).all() and grad_aux.abs().sum()>0
        loss.backward();grad=policy.model.action_out_proj.weight.grad
        assert torch.isfinite(grad).all() and grad.abs().sum()>0 and not torch.cuda.is_initialized()
        print('ACTUAL_CPU_SERVO_GRADIENT_PASS hook_matches_upstream=true auxiliary_gradient_nonzero=true padding_and_tail_excluded=true CUDA=false');return
    policy.to('cuda:0');cache.clear()
    out=Path(ds.directory)/'servo_ablation_runs'/datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True,exist_ok=False)
    print('OUTPUT '+str(out),flush=True)
    files=[Path(__file__),ROOT/'src/capstone_vla/ball_servo_loss.py',ROOT/'scripts/train_hgwind_prefix_ablation.py',ROOT/'src/capstone_vla/ball_noise_replanning.py',ROOT/'src/capstone_vla/ball_sampling_diagnostic.py',ROOT/'src/capstone_vla/ball_precision.py',ROOT/'src/capstone_vla/ball_refinement.py',ROOT/'src/capstone_vla/ball_prediction_metrics.py',ROOT/'scripts/finetune_hgwind_development.py',ROOT/'scripts/predict_hgwind_development.py',ROOT/'results/summaries/2026-10-31-ball-upper-target-audit.json']
    hashes={**parent['source_hashes'],**{str(f.relative_to(ROOT)):sha256(f) for f in files}}
    protocol=dict(parent=str(PARENT),parent_checkpoint_sha256=sha256(PARENT/'checkpoint_final/model.safetensors'),steps_each=STEPS,branches=dict(uniform=0.,servo=AUX_WEIGHT),auxiliary_scale_mps=.1,auxiliary_prefix=4,auxiliary_context="Teacher caps/reference/masks and future measured pinch states; only XYZ/feedforward predicted; offline supervision only",flow="unchanged valid29 uniform full20",learning_rate=1e-5,batch_size=2,training_seed_rule='torch.manual_seed(310000+step); explicit Gaussian and Beta flow time; matched dropout seed',schedule=schedule.tolist(),evaluation_seeds=list(SEEDS),source_hashes=hashes,reference_report_sha256=sha256(REFERENCE/'report.json'),selection='none; save both final candidates; parent/default unchanged',purpose='single exposed training episode; paired objective ablation')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    expected=np.load(REFERENCE/'predictions.npz',allow_pickle=False)['chunks'][0]
    baseline=generated(policy,ds,cache,out,'parent',expected)
    (out/'parent_metrics.json').write_text(json.dumps(baseline,indent=2,allow_nan=False)+'\n')
    with safe_open(str(weights),framework='pt',device='cpu') as f:official_projection=f.get_tensor('model.action_out_proj.weight')
    trainable=[p for p in policy.parameters() if p.requires_grad];reference_random=[];candidates={}
    for name,factor in [('uniform',0.),('servo',AUX_WEIGHT)]:
        policy.zero_grad(set_to_none=True);load_model(policy,str(PARENT/'checkpoint_final/model.safetensors'),strict=True,device='cpu');check_normalizers(policy,stats);policy.reset()
        branch=out/name;branch.mkdir();seen=set();start=time.monotonic()
        record={k:parent[k] for k in ['source_dataset','dataset_manifest_sha256','pretrained','upstream_commit','weights_sha256','schema','image_size','augmentation','backbone_frozen','absolute_actions','normalization_fit']}
        record.update(steps=STEPS,batch_size=2,seed=31,learning_rate=1e-5,purpose='development_overfit_only',parent_run=str(PARENT),parent_checkpoint_sha256=protocol['parent_checkpoint_sha256'],source_hashes=hashes,
            optimizer_loss='valid29 uniform flow + conditional bounded translation endpoint Huber',auxiliary_weight=factor,auxiliary_scale_mps=.1,auxiliary_prefix=4,optimizer_restarted=True,optimizer_state_resumed=False,
            total_lineage_optimizer_steps=3500,selected_lineage_optimizer_steps=3500,selected_checkpoint_step=500,selection='final candidate saved without promotion',heldout_evaluation=False,simulation_rollout=False,task_success_verified=False,protocol_sha256=sha256(out/'protocol.json'))
        (branch/'run_config.json').write_text(json.dumps(record,indent=2)+'\n')
        optimizer=torch.optim.AdamW(trainable,lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=1e-10)
        with (branch/'training.jsonl').open('x') as log:
            for step,ids in enumerate(schedule,1):
                policy.train();policy.model.vlm_with_expert.get_vlm_model().eval();seen.update(map(int,ids))
                batch=batch_from(ds,ids,cache,'cuda:0');torch.manual_seed(310000+step)
                noise=policy.model._sample_noise((2,20,32),'cuda:0');times=policy.model._sample_time(2,'cuda:0')
                digest=hashlib.sha256(noise.cpu().numpy().tobytes()+times.cpu().numpy().tobytes()).hexdigest()
                if name=='uniform':reference_random.append(digest)
                elif digest!=reference_random[step-1]:raise ValueError('Training random inputs not paired')
                optimizer.zero_grad(set_to_none=True)
                pinch=torch.as_tensor(future_pinch(ds.states,ids,batch['action_is_pad'].cpu().numpy()),device='cuda:0')
                loss,raw,aux,official=objective(policy,batch,pinch,noise,times,factor)
                if not torch.isfinite(loss) or not torch.isfinite(official):raise ValueError('Nonfinite loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(trainable,10.,error_if_nonfinite=True)
                if float(norm)<=0:raise ValueError('Zero gradients')
                optimizer.step()
                row=dict(step=step,frames=ids.tolist(),optimizer_step=True,valid_29d_flow_mse=float(raw.detach()),official_32d_loss=float(official.detach()),total_objective=float(loss.detach()),conditional_command_huber=float(aux.detach()),gradient_norm_before_clip=float(norm),noise_time_sha256=digest)
                log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                if step==1 or step%50==0:print(f'SERVO_TRAIN branch={name} step={step}/500 loss={row["total_objective"]:.6f} elapsed={time.monotonic()-start:.0f}s',flush=True)
        optimizer.zero_grad(set_to_none=True);del optimizer
        policy.eval();check_normalizers(policy,stats);checkpoint=branch/'checkpoint_final';policy.save_pretrained(checkpoint)
        (checkpoint/'capstone_run.json').write_text(json.dumps(record,indent=2)+'\n');(checkpoint/'capstone_normalization.json').write_bytes((PARENT/'checkpoint_final/capstone_normalization.json').read_bytes())
        m=generated(policy,ds,cache,branch,'final');candidates[name]=m
        change=float(torch.linalg.vector_norm(policy.model.action_out_proj.weight.detach().cpu()-official_projection))
        report=dict(status='BALL_DEVELOPMENT_FINETUNE_COMPLETED',optimizer_steps=STEPS,unique_training_frames=len(seen),checkpoint=str(checkpoint),action_projection_l2_change=change,selected_checkpoint_step=STEPS,selected_improved=None,generated_metrics=m,heldout_evaluation=False,simulation_rollout=False,task_success_verified=False,scope='Final candidate only; no automatic promotion',checkpoint_sha256=sha256(checkpoint/'model.safetensors'))
        (branch/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');inspect_run(branch)
        (out/'partial_metrics.json').write_text(json.dumps(candidates,indent=2,allow_nan=False)+'\n')
    comparison={}
    for key in ['approach_position_mean_mm','precontact_80_119_mean_mm','precontact_increment_error_mm','full_position_p95_mm','orientation_p95_deg','gripper_error_frames','gripper_transitions','native_rejected_frames','startup_command_mean_mps','near_command_mean_mps','full_command_mean_mps']:
        if any(x[key] is None for group in [baseline,*candidates.values()] for x in group):
            comparison[key]=dict(comparable=False,reason='Invalid full predicted command; inspect per-seed rejects');continue
        b=np.array([x[key] for x in baseline]);u=np.array([x[key] for x in candidates['uniform']]);v=np.array([x[key] for x in candidates['servo']]);d=v-u
        comparison[key]=dict(parent_mean=float(b.mean()),uniform_mean=float(u.mean()),servo_mean=float(v.mean()),servo_minus_uniform=d.tolist(),improved_pairs=int((d<0).sum()),worsened_pairs=int((d>0).sum()))
    result=dict(status='BALL_SERVO_ABLATION_COMPLETED',paired_comparison=comparison,protocol_sha256=sha256(out/'protocol.json'),random_inputs_exactly_paired=True,checkpoints={n:sha256(out/n/'checkpoint_final/model.safetensors') for n in candidates},selected_model=None,default_changed=False,heldout=False,physics=False,task_success_verified=False,command_rejections={n:[m['command_rejected_frames'] for m in ms] for n,ms in dict(parent=baseline,**candidates).items()},limitations='One training episode and one paired training schedule; eight evaluation noise seeds are not independent demonstrations. Final candidates retained; improvements require review, not automatic rollout.')
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('SERVO_COMPARISON '+json.dumps(comparison),flush=True)
    print('BALL_SERVO_ABLATION_COMPLETED report='+str(out/'report.json'),flush=True)


if __name__=='__main__':main()
