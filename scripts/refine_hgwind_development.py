#!/usr/bin/env python3
"""Bounded lower-LR refinement from the saved ball model; no simulator runs."""
import argparse
from collections import OrderedDict
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo
import h5py
import numpy as np
import torch
from safetensors.torch import load_model

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from finetune_hgwind_development import build_policy,batch_from,valid_loss
from predict_hgwind_development import inspect_run,check_normalizers
from capstone_vla.ball_refinement import generated_metrics,accept_candidate
from capstone_vla.ball_prediction_metrics import prefix_stream
from capstone_vla.hf_archive import sha256
from capstone_vla.pilot_dataset import CAMERAS

DEFAULT=ROOT/'data/derived/hgwind_traj18_development_20261031/training_runs/20261031_121355_529372'
ARCHIVE=DEFAULT/'offline_predictions/20261031_121834_695770/predictions.hdf5'
BATCH1_REFERENCE=DEFAULT/'closed_loop_runs/20261031_125333_149097/failure_diagnostics/20261031_130917_641130'


def predict_stream(policy,ds,cache):
    policy.eval();stream=[]
    with torch.inference_mode():
        for frame in range(0,len(ds),4):
            batch=batch_from(ds,[frame],cache,'cuda:0')
            inputs={k:batch[k] for k in ['task','observation.state',*['observation.images.'+c for c in CAMERAS]]}
            noise=torch.randn((1,20,32),device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(90000+frame))
            policy.reset();chunk=policy.predict_action_chunk(inputs,noise=noise).cpu().numpy()[0]
            if chunk.shape!=(20,29) or not np.isfinite(chunk).all():raise RuntimeError('Invalid prediction')
            stream.extend(chunk[:min(4,len(ds)-frame)])
    return np.asarray(stream)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true')
    p.add_argument('--check-load',action='store_true');p.add_argument('--steps',type=int,choices=[250,500,1000],default=1000)
    a=p.parse_args()
    if a.run and a.check_load:p.error('Choose --run or --check-load')
    torch.set_num_threads(8);torch.manual_seed(31)
    ds,cfg,stats,pin,weights,_=inspect_run(DEFAULT)
    parent=json.loads((DEFAULT/'run_config.json').read_text());checkpoint=DEFAULT/'checkpoint_final'
    stats_json=json.loads((checkpoint/'capstone_normalization.json').read_text())
    record=json.loads(ARCHIVE.with_name('report.json').read_text())
    if sha256(ARCHIVE)!=record['prediction_archive_sha256']:raise ValueError('Prediction reference changed')
    with h5py.File(ARCHIVE) as f:
        np.testing.assert_array_equal(ds.actions,f['target_absolute29'][:])
        recorded,_,_=prefix_stream(f['finetuned/absolute_chunks'][:],4)
    verified=json.loads((BATCH1_REFERENCE/'verified_input_metrics.json').read_text())
    for name in ['report.json','input_probe_chunks.npz']:
        path=BATCH1_REFERENCE/name
        if sha256(path)!=verified['source_sha256'][str(path)]:raise ValueError('Batch1 reference changed')
    batch1_record=json.loads((BATCH1_REFERENCE/'report.json').read_text())
    for path,digest in batch1_record['input_probe']['checkpoint_hashes'].items():
        if sha256(path)!=digest:raise ValueError('Batch1 checkpoint differs')
    with np.load(BATCH1_REFERENCE/'input_probe_chunks.npz',allow_pickle=False) as f:
        batch1_chunks=f['chunks'].reshape(8,4,20,29)[:,0].copy()
    anchor=ds.states[0,21:28]
    offline=generated_metrics(recorded,ds.actions,anchor)
    batch=batch_from(ds,[0,268],OrderedDict(),'cpu')
    assert batch['action'].shape==(2,20,29) and batch['observation.state'].shape==(2,2,30)
    assert not cfg.use_delta_action and offline['approach_end_exclusive']==144
    assert not accept_candidate(offline,offline,offline)
    print(f'CPU_REFINEMENT_PREFLIGHT_PASS parent_steps=2000 additional_steps={a.steps} lr=1e-5 generated_probe_every=250 prefix=4 frames=269',flush=True)
    print('SELECTION approach_error_decreases; gripper_errors/transitions_and_orientation_do_not_worsen; native_rejections=0; training_case_only',flush=True)
    if not (a.run or a.check_load):print('GPU_NOT_STARTED; --run trains; --check-load verifies CPU reload');return
    policy=build_policy(cfg,stats,pin,weights,ROOT/'data/models/smollm2_metadata'/pin['backbone_revision'])
    load_model(policy,str(checkpoint/'model.safetensors'),strict=True,device='cpu');check_normalizers(policy,stats)
    print('PARENT_FINETUNED_WEIGHTS_LOADED steps=2000 strict=true optimizer_restart=true',flush=True)
    if a.check_load:
        assert not torch.cuda.is_initialized();print('PARENT_CHECKPOINT_CPU_RELOAD_PASS CUDA=false');return
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    policy.to('cuda:0');cache=OrderedDict()
    out=Path(ds.directory)/'training_runs'/datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False);print('OUTPUT '+str(out),flush=True)
    metadata=dict(parent)
    metadata.update(steps=a.steps,learning_rate=1e-5,initialization='strict_parent_finetuned_weights',
        parent_run=str(DEFAULT),parent_checkpoint_sha256=sha256(checkpoint/'model.safetensors'),
        optimizer_restarted=True,optimizer_state_resumed=False,probe_interval=250,probe_prefix=4,
        probe_noise='90000+frame',selection='lower approach mean; no worse baseline gripper errors/transitions or orientation p95; zero native rejects',
        selected_checkpoint_step=0,total_lineage_optimizer_steps=2000+a.steps)
    metadata['source_hashes']={**parent['source_hashes'],str(Path(__file__).relative_to(ROOT)):sha256(__file__),
        'src/capstone_vla/ball_refinement.py':sha256(ROOT/'src/capstone_vla/ball_refinement.py')}
    (out/'run_config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    baseline_stream=predict_stream(policy,ds,cache)
    # Prior full offline pass used batch4. Reproduce the available batch1
    # teacher-input probes instead, and measure the full batch1 baseline anew.
    for i,frame in enumerate(batch1_record['input_probe']['anchors']):
        np.testing.assert_allclose(baseline_stream[frame:frame+4],batch1_chunks[i,:4],rtol=1e-4,atol=1e-5)
    metadata['baseline_batch1_reference_verified']=True
    metadata['batch1_reference_sha256']=sha256(BATCH1_REFERENCE/'input_probe_chunks.npz')
    metadata['baseline_batch1_vs_archived_batch4_max_abs']=float(np.max(abs(baseline_stream-recorded)))
    baseline=generated_metrics(baseline_stream,ds.actions,anchor);best=baseline;selected=0
    print('GENERATED_BASELINE '+json.dumps({k:baseline[k] for k in ['approach_position_mean_mm','gripper_error_frames','gripper_transitions','native_rejected_frames']}),flush=True)
    np.save(out/'generated_step0000.npy',baseline_stream,allow_pickle=False)
    # This path is new and owned by this run. Retain the parent if no candidate passes.
    selected_dir=out/'checkpoint_final';policy.save_pretrained(selected_dir)
    parameters=[v for v in policy.parameters() if v.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=1e-10)
    rng=np.random.default_rng(31);seen=set();probes=[];started=time.monotonic()
    with (out/'training.jsonl').open('x') as log:
        for step in range(1,a.steps+1):
            policy.train();policy.model.vlm_with_expert.get_vlm_model().eval()
            ids=rng.integers(0,len(ds),size=2);seen.update(map(int,ids));batch=batch_from(ds,ids,cache,'cuda:0')
            optimizer.zero_grad(set_to_none=True);official,details=policy(batch)
            loss=valid_loss(details['losses_after_forward'],batch['action_is_pad'])
            if not torch.isfinite(loss) or not torch.isfinite(official):raise RuntimeError('Nonfinite training loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,10.,error_if_nonfinite=True)
            if float(norm)<=0:raise RuntimeError('Zero gradients')
            optimizer.step()
            row=dict(step=step,optimizer_step=True,frames=ids.tolist(),valid_29d_flow_mse=float(loss.detach()),
                official_32d_loss=float(official.detach()),gradient_norm_before_clip=float(norm),elapsed_s=time.monotonic()-started)
            log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
            if step==1 or step%50==0:print(f'REFINE step={step}/{a.steps} loss={float(loss.detach()):.6f}',flush=True)
            if step%250==0:
                stream=predict_stream(policy,ds,cache);metrics=generated_metrics(stream,ds.actions,anchor)
                accepted=accept_candidate(metrics,baseline,best)
                if accepted:
                    best=metrics;selected=step;policy.save_pretrained(selected_dir)
                np.save(out/f'generated_step{step:04d}.npy',stream,allow_pickle=False)
                probes.append(dict(step=step,accepted=accepted,metrics=metrics))
                (out/'generated_probes.json').write_text(json.dumps(probes,indent=2,allow_nan=False)+'\n')
                print('GENERATED_PROBE '+json.dumps(dict(step=step,accepted=accepted,selected_step=selected,
                    **{k:metrics[k] for k in ['approach_position_mean_mm','gripper_error_frames','gripper_transitions','native_rejected_frames']})),flush=True)
    metadata['selected_checkpoint_step']=selected
    metadata['selected_lineage_optimizer_steps']=2000+selected
    (out/'run_config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (selected_dir/'capstone_run.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (selected_dir/'capstone_normalization.json').write_text(json.dumps(stats_json,indent=2)+'\n')
    load_model(policy,str(selected_dir/'model.safetensors'),strict=True,device='cpu');check_normalizers(policy,stats)
    report=dict(status='BALL_DEVELOPMENT_FINETUNE_COMPLETED',optimizer_steps=a.steps,unique_training_frames=len(seen),
        selected_checkpoint_step=selected,checkpoint=str(selected_dir),selected_improved=selected>0,
        checkpoint_is_last_step=selected==a.steps,baseline_generated=baseline,selected_generated=best,
        probes=probes,heldout_evaluation=False,simulation_rollout=False,task_success_verified=False,
        scope='Same-episode development checkpoint selection; no validation split, no success/generalization claim')
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('BALL_REFINEMENT_COMPLETED selected_step='+str(selected)+' report='+str(out/'report.json'),flush=True)


if __name__=='__main__':main()
