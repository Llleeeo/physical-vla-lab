#!/usr/bin/env python3
"""Matched500-step control/prefix4 training pair; no automatic checkpoint promotion."""
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
from capstone_vla.ball_prefix_loss import prefix_loss
from capstone_vla.ball_noise_replanning import SEEDS,assemble_prefix,stream_metrics
from capstone_vla.ball_sampling_diagnostic import predict_preserving_noise
from capstone_vla.hf_archive import sha256
from capstone_vla.pilot_dataset import CAMERAS
PARENT=ROOT/'data/derived/hgwind_traj18_development_20261031/training_runs/20261031_133555_985430'
REFERENCE=PARENT/'noise_replanning_diagnostics/20261031_150546_311797'
STEPS=500


def cpu_math_check():
    pad=torch.zeros((2,20),dtype=torch.bool);pad[1,1:]=True
    losses=torch.arange(2*20*32,dtype=torch.float32).reshape(2,20,32).requires_grad_()
    torch.testing.assert_close(prefix_loss(losses,pad,1),valid_loss(losses,pad),rtol=1e-6,atol=1e-6)
    prefix_loss(losses,pad,4).backward()
    assert not losses.grad[pad].count_nonzero() and not losses.grad[...,29:].count_nonzero()
    torch.testing.assert_close(losses.grad[0,0,0],4*losses.grad[0,4,0])
    ones=torch.ones_like(losses);torch.testing.assert_close(prefix_loss(ones,pad,4),torch.tensor(1.))
    assert not torch.cuda.is_initialized()


def generated(policy,ds,cache,out,label,expected=None):
    policy.eval();all_chunks=[];all_metrics=[]
    with torch.inference_mode():
        for si,seed in enumerate(SEEDS):
            chunks=[]
            for frame in range(0,len(ds),4):
                batch=batch_from(ds,[frame],cache,'cuda:0')
                inputs={k:batch[k] for k in ['task','observation.state',*['observation.images.'+c for c in CAMERAS]]}
                noise=torch.randn((1,20,32),device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(seed+frame))
                chunk=predict_preserving_noise(policy,inputs,noise,10).cpu().numpy()[0];chunks.append(chunk)
            chunks=np.asarray(chunks)
            if expected is not None:np.testing.assert_allclose(chunks,expected[si],rtol=1e-4,atol=1e-5)
            stream=assemble_prefix(chunks);m=stream_metrics(stream,ds.actions,ds.states[0,21:28]);all_chunks.append(chunks);all_metrics.append(dict(seed=seed,**m))
            np.savez_compressed(out/(label+'_predictions.npz'),seeds=np.asarray(SEEDS[:si+1]),chunks=np.asarray(all_chunks))
            print('PREFIX_PROBE '+json.dumps(dict(model=label,seed=seed,near_mm=m['precontact_80_119_mean_mm'],increment_error_mm=m['precontact_increment_error_mm'],startup_mm=m['startup_0_39_position_mm']['mean'],jaw_errors=m['gripper_error_frames'],jaw_transitions=m['gripper_transitions'],native_rejects=m['native_rejected_frames'])),flush=True)
    return all_metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group();group.add_argument('--run',action='store_true');group.add_argument('--check-load',action='store_true');args=parser.parse_args()
    torch.set_num_threads(8);torch.manual_seed(31);cpu_math_check()
    ds,cfg,stats,pin,weights,_=inspect_run(PARENT);cache=OrderedDict();parent=json.loads((PARENT/'run_config.json').read_text())
    ref=json.loads((REFERENCE/'report.json').read_text());refcfg=json.loads((REFERENCE/'run_config.json').read_text())
    if sha256(REFERENCE/'predictions.npz')!=ref['predictions_sha256'] or sha256(REFERENCE/'run_config.json')!=ref['config_sha256']:raise ValueError('Prior evidence changed')
    for name,digest in refcfg['source_hashes'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Prior source changed: '+name)
    if cfg.num_steps!=10 or parent['selected_lineage_optimizer_steps']!=3000:raise ValueError('Wrong parent')
    schedule=np.random.default_rng(31).integers(0,len(ds),size=(STEPS,2))
    batch=batch_from(ds,[0,268],cache,'cpu')
    print('CPU_PREFIX_ABLATION_PASS branches=uniform,prefix4 steps_each=500 parent=3000 lr=1e-5 batch=2 prefix_weights=1,4 tail_weight=1 matched_batches_noise_time=true eval_seeds=8',flush=True)
    if not(args.run or args.check_load):print('GPU_NOT_STARTED; --run executes sequential training pair');return
    policy=build_policy(cfg,stats,pin,weights,ROOT/'data/models/smollm2_metadata'/pin['backbone_revision'])
    load_model(policy,str(PARENT/'checkpoint_final/model.safetensors'),strict=True,device='cpu');check_normalizers(policy,stats)
    if args.check_load:
        policy.train();policy.model.vlm_with_expert.get_vlm_model().eval()
        _,details=policy(batch)
        torch.testing.assert_close(prefix_loss(details['losses_after_forward'],batch['action_is_pad'],1),valid_loss(details['losses_after_forward'],batch['action_is_pad']),rtol=1e-6,atol=1e-6)
        loss=prefix_loss(details['losses_after_forward'],batch['action_is_pad'],4);loss.backward()
        grad=policy.model.action_out_proj.weight.grad
        assert torch.isfinite(grad).all() and grad.abs().sum()>0 and not torch.cuda.is_initialized()
        print('ACTUAL_CPU_PREFIX_GRADIENT_PASS uniform_matches_existing=true prefix_padding_gradients=verified CUDA=false');return
    policy.to('cuda:0');cache.clear()
    out=Path(ds.directory)/'prefix_ablation_runs'/datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True,exist_ok=False)
    print('OUTPUT '+str(out),flush=True)
    files=[Path(__file__),ROOT/'src/capstone_vla/ball_prefix_loss.py',ROOT/'src/capstone_vla/ball_noise_replanning.py',ROOT/'src/capstone_vla/ball_sampling_diagnostic.py',ROOT/'src/capstone_vla/ball_precision.py',ROOT/'src/capstone_vla/ball_refinement.py',ROOT/'src/capstone_vla/ball_prediction_metrics.py',ROOT/'scripts/finetune_hgwind_development.py',ROOT/'scripts/predict_hgwind_development.py']
    hashes={**parent['source_hashes'],**{str(f.relative_to(ROOT)):sha256(f) for f in files}}
    protocol=dict(parent=str(PARENT),parent_checkpoint_sha256=sha256(PARENT/'checkpoint_final/model.safetensors'),steps_each=STEPS,branches=dict(uniform=1,prefix4=4),tail_weight=1,learning_rate=1e-5,batch_size=2,training_seed_rule='torch.manual_seed(310000+step); explicit Gaussian and Beta flow time; matched dropout seed',schedule=schedule.tolist(),evaluation_seeds=list(SEEDS),source_hashes=hashes,reference_report_sha256=sha256(REFERENCE/'report.json'),selection='none; save both final candidates; parent/default unchanged',purpose='single exposed training episode; paired objective ablation')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    expected=np.load(REFERENCE/'predictions.npz',allow_pickle=False)['chunks'][0]
    baseline=generated(policy,ds,cache,out,'parent',expected)
    (out/'parent_metrics.json').write_text(json.dumps(baseline,indent=2,allow_nan=False)+'\n')
    with safe_open(str(weights),framework='pt',device='cpu') as f:official_projection=f.get_tensor('model.action_out_proj.weight')
    trainable=[p for p in policy.parameters() if p.requires_grad];reference_random=[];candidates={}
    for name,factor in [('uniform',1),('prefix4',4)]:
        policy.zero_grad(set_to_none=True);load_model(policy,str(PARENT/'checkpoint_final/model.safetensors'),strict=True,device='cpu');check_normalizers(policy,stats);policy.reset()
        branch=out/name;branch.mkdir();seen=set();start=time.monotonic()
        record={k:parent[k] for k in ['source_dataset','dataset_manifest_sha256','pretrained','upstream_commit','weights_sha256','schema','image_size','augmentation','backbone_frozen','absolute_actions','normalization_fit']}
        record.update(steps=STEPS,batch_size=2,seed=31,learning_rate=1e-5,purpose='development_overfit_only',parent_run=str(PARENT),parent_checkpoint_sha256=protocol['parent_checkpoint_sha256'],source_hashes=hashes,
            optimizer_loss='valid29 flow with normalized temporal weights',prefix_weight=factor,tail_weight=1,optimizer_restarted=True,optimizer_state_resumed=False,
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
                optimizer.zero_grad(set_to_none=True);official,details=policy(batch,noise=noise,time=times)
                raw=valid_loss(details['losses_after_forward'],batch['action_is_pad']);loss=prefix_loss(details['losses_after_forward'],batch['action_is_pad'],factor)
                if not torch.isfinite(loss) or not torch.isfinite(official):raise ValueError('Nonfinite loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(trainable,10.,error_if_nonfinite=True)
                if float(norm)<=0:raise ValueError('Zero gradients')
                optimizer.step()
                row=dict(step=step,frames=ids.tolist(),optimizer_step=True,valid_29d_flow_mse=float(raw.detach()),official_32d_loss=float(official.detach()),weighted_flow=float(loss.detach()),gradient_norm_before_clip=float(norm),noise_time_sha256=digest)
                log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                if step==1 or step%50==0:print(f'PREFIX_TRAIN branch={name} step={step}/500 loss={row["weighted_flow"]:.6f} elapsed={time.monotonic()-start:.0f}s',flush=True)
        optimizer.zero_grad(set_to_none=True);del optimizer
        policy.eval();check_normalizers(policy,stats);checkpoint=branch/'checkpoint_final';policy.save_pretrained(checkpoint)
        (checkpoint/'capstone_run.json').write_text(json.dumps(record,indent=2)+'\n');(checkpoint/'capstone_normalization.json').write_bytes((PARENT/'checkpoint_final/capstone_normalization.json').read_bytes())
        m=generated(policy,ds,cache,branch,'final');candidates[name]=m
        change=float(torch.linalg.vector_norm(policy.model.action_out_proj.weight.detach().cpu()-official_projection))
        report=dict(status='BALL_DEVELOPMENT_FINETUNE_COMPLETED',optimizer_steps=STEPS,unique_training_frames=len(seen),checkpoint=str(checkpoint),action_projection_l2_change=change,selected_checkpoint_step=STEPS,selected_improved=None,generated_metrics=m,heldout_evaluation=False,simulation_rollout=False,task_success_verified=False,scope='Final candidate only; no automatic promotion',checkpoint_sha256=sha256(checkpoint/'model.safetensors'))
        (branch/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');inspect_run(branch)
        (out/'partial_metrics.json').write_text(json.dumps(candidates,indent=2,allow_nan=False)+'\n')
    comparison={}
    for key in ['approach_position_mean_mm','precontact_80_119_mean_mm','precontact_increment_error_mm','full_position_p95_mm','orientation_p95_deg','gripper_error_frames','gripper_transitions','native_rejected_frames']:
        b=np.array([x[key] for x in baseline]);u=np.array([x[key] for x in candidates['uniform']]);v=np.array([x[key] for x in candidates['prefix4']]);d=v-u
        comparison[key]=dict(parent_mean=float(b.mean()),uniform_mean=float(u.mean()),prefix4_mean=float(v.mean()),prefix4_minus_uniform=d.tolist(),improved_pairs=int((d<0).sum()),worsened_pairs=int((d>0).sum()))
    result=dict(status='BALL_PREFIX_ABLATION_COMPLETED',paired_comparison=comparison,protocol_sha256=sha256(out/'protocol.json'),random_inputs_exactly_paired=True,checkpoints={n:sha256(out/n/'checkpoint_final/model.safetensors') for n in candidates},selected_model=None,default_changed=False,heldout=False,physics=False,task_success_verified=False,limitations='One training episode and one paired training schedule; eight evaluation noise seeds are not independent demonstrations. Final candidates retained; improvements require review, not automatic rollout.')
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('PREFIX_COMPARISON '+json.dumps(comparison),flush=True)
    print('BALL_PREFIX_ABLATION_COMPLETED report='+str(out/'report.json'),flush=True)


if __name__=='__main__':main()
