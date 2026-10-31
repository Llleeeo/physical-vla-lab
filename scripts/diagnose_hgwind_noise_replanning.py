#!/usr/bin/env python3
"""Full expert-input comparison of independent versus shared Gaussian noise."""
import argparse
from collections import OrderedDict
from datetime import datetime
import json
from pathlib import Path
import sys,time
from zoneinfo import ZoneInfo
import numpy as np
import torch
from safetensors.torch import load_model
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from finetune_hgwind_development import build_policy,batch_from
from predict_hgwind_development import inspect_run,check_normalizers
from capstone_vla.ball_sampling_diagnostic import predict_preserving_noise
from capstone_vla.ball_noise_replanning import SEEDS,MODES,noise_seed,assemble_prefix,stream_metrics
from capstone_vla.hf_archive import sha256
from capstone_vla.pilot_dataset import CAMERAS
PARENT=ROOT/'data/derived/hgwind_traj18_development_20261031/training_runs/20261031_133555_985430'
AUDIT=PARENT/'sampling_diagnostics/20261031_145658_077694'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group();g.add_argument('--run',action='store_true');g.add_argument('--check-load',action='store_true');args=p.parse_args()
    torch.set_num_threads(8);torch.manual_seed(31)
    ds,cfg,stats,pin,weights,_=inspect_run(PARENT);cache=OrderedDict();frames=list(range(0,len(ds),4))
    if len(ds)!=269 or cfg.num_steps!=10:raise ValueError('Unexpected source configuration')
    previous=json.loads((AUDIT/'report.json').read_text());oldconfig=json.loads((AUDIT/'run_config.json').read_text())
    if sha256(AUDIT/'run_config.json')!=previous['source_config_sha256'] or sha256(AUDIT/'predictions.npz')!=previous['predictions_sha256']:raise ValueError('Prior sampling evidence changed')
    for path,digest in oldconfig['source_hashes'].items():
        if sha256(ROOT/path)!=digest:raise ValueError('Prior source changed: '+path)
    old=np.load(AUDIT/'predictions.npz',allow_pickle=False)
    if old['actions'].shape!=(2,12,8,20,29):raise ValueError('Unexpected prior sample shape')
    archived=np.load(PARENT/'generated_step1000.npy',allow_pickle=False)
    for f in frames:
        batch=batch_from(ds,[f],cache,'cpu');valid=min(4,len(ds)-f)
        if batch['action_is_pad'][0,:valid].any():raise ValueError('Consumed padding')
        if f==frames[-1] and not batch['action_is_pad'][0,valid:].all():raise ValueError('Missing terminal padding')
    stream_metrics(ds.actions,ds.actions,ds.states[0,21:28])
    assert not torch.cuda.is_initialized()
    print('CPU_NOISE_REPLAN_PREFLIGHT_PASS frames=269 replans=68 seeds=8 modes=independent,shared steps=10 prefix=4 padding=verified training=false physics=false',flush=True)
    if not(args.run or args.check_load):print('GPU_NOT_STARTED; use --run');return
    policy=build_policy(cfg,stats,pin,weights,ROOT/'data/models/smollm2_metadata'/pin['backbone_revision'])
    load_model(policy,str(PARENT/'checkpoint_final/model.safetensors'),strict=True,device='cpu');check_normalizers(policy,stats)
    if args.check_load:
        assert not torch.cuda.is_initialized()
        print('NOISE_REPLAN_STRICT_CPU_LOAD_PASS CUDA=false');return
    policy.to('cuda:0').eval();cache.clear()
    out=PARENT/'noise_replanning_diagnostics'/datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True,exist_ok=False)
    print('OUTPUT '+str(out),flush=True)
    paths=[Path(__file__),ROOT/'src/capstone_vla/ball_noise_replanning.py',ROOT/'src/capstone_vla/ball_sampling_diagnostic.py',ROOT/'src/capstone_vla/ball_precision.py',ROOT/'src/capstone_vla/ball_refinement.py',AUDIT/'report.json',AUDIT/'predictions.npz']
    config=dict(parent=str(PARENT),seeds=list(SEEDS),modes=list(MODES),frames=frames,episode_frames=len(ds),prefix=4,integration_steps=10,
        source_hashes={**oldconfig['source_hashes'],**{str(x.relative_to(ROOT)):sha256(x) for x in paths}},
        independent_noise='Gaussian seed=base+frame',shared_noise='Gaussian seed=base; same relative chunk slots at every replan; not shifted previous actions',
        inputs='Expert RGB/state history updated at every replan; same inputs for both modes',training=False,physics=False,selection='None; report all eight seed pairs')
    (out/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')
    chunks=np.empty((2,8,len(frames),20,29),np.float32);times=np.empty((2,8,len(frames)))
    results={m:[] for m in MODES};start=time.monotonic();reference_diffs=[]
    lookup={int(f):i for i,f in enumerate(old['frames'])}
    with torch.inference_mode():
        for si,seed in enumerate(SEEDS):
            for fi,f in enumerate(frames):
                batch=batch_from(ds,[f],cache,'cuda:0')
                inputs={k:batch[k] for k in ['task','observation.state',*['observation.images.'+c for c in CAMERAS]]}
                for mi,mode in enumerate(MODES):
                    noise=torch.randn((1,20,32),device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(noise_seed(mode,seed,f)))
                    original=noise.clone();torch.cuda.synchronize();tic=time.perf_counter()
                    a=predict_preserving_noise(policy,inputs,noise,10)
                    torch.cuda.synchronize();times[mi,si,fi]=time.perf_counter()-tic
                    if not torch.equal(noise,original):raise ValueError('Noise mutated')
                    a=a.cpu().numpy()[0]
                    if a.shape!=(20,29) or not np.isfinite(a).all():raise ValueError('Invalid predictions')
                    chunks[mi,si,fi]=a
                if f==0:np.testing.assert_allclose(chunks[0,si,fi],chunks[1,si,fi],rtol=1e-4,atol=1e-5)
                if f in lookup:
                    expected=old['actions'][0,lookup[f],si]
                    reference_diffs.append(float(np.max(np.abs(chunks[0,si,fi]-expected))))
                    np.testing.assert_allclose(chunks[0,si,fi],expected,rtol=1e-4,atol=1e-5)
                if fi%17==0:print(f'NOISE_REPLAN seed={si+1}/8 anchors={fi+1}/68 elapsed={time.monotonic()-start:.0f}s',flush=True)
            streams=np.stack([assemble_prefix(chunks[mi,si]) for mi in range(2)])
            if si==0:np.testing.assert_allclose(streams[0],archived,rtol=1e-4,atol=1e-5)
            for mi,mode in enumerate(MODES):
                m=stream_metrics(streams[mi],ds.actions,ds.states[0,21:28]);results[mode].append(dict(seed=seed,**m))
                print('NOISE_REPLAN_RESULT '+json.dumps(dict(mode=mode,seed=seed,near_mm=m['precontact_80_119_mean_mm'],increment_error_mm=m['precontact_increment_error_mm'],boundary_increment_error_mm=m['increments']['precontact_80_119_replan_boundary']['expert_increment_error_mm']['mean'],gripper_errors=m['gripper_error_frames'],gripper_transitions=m['gripper_transitions'],native_rejections=m['native_rejected_frames'])),flush=True)
            np.savez_compressed(out/'predictions.npz',seeds=np.asarray(SEEDS[:si+1]),frames=np.asarray(frames),chunks=chunks[:,:si+1],latency_s=times[:,:si+1])
            (out/'partial_metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    paired={}
    for k in ['approach_position_mean_mm','precontact_80_119_mean_mm','precontact_increment_error_mm','full_position_p95_mm','orientation_p95_deg','gripper_error_frames','gripper_transitions','native_rejected_frames']:
        raw=np.array([x[k] for x in results['independent']]);shared=np.array([x[k] for x in results['shared']]);delta=shared-raw
        paired[k]=dict(independent_mean=float(raw.mean()),shared_mean=float(shared.mean()),shared_minus_independent=delta.tolist(),improved_seed_pairs=int((delta<0).sum()),unchanged_seed_pairs=int((delta==0).sum()),worsened_seed_pairs=int((delta>0).sum()))
    report=dict(status='BALL_NOISE_REPLANNING_COMPLETED',per_seed=results,paired_metrics=paired,prior_reference_max_abs=max(reference_diffs),
        config_sha256=sha256(out/'run_config.json'),predictions_sha256=sha256(out/'predictions.npz'),training=False,physics=False,heldout=False,task_success_verified=False,
        limitations='Eight seeds on one exposed episode; no seed selection. Shared noise retains stochastic sample bias and relative-slot effects. Uses expert observations, not policy feedback. Smoothness alone is not success. No automatic deployment.')
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('BALL_NOISE_REPLANNING_COMPLETED report='+str(out/'report.json'),flush=True)


if __name__=='__main__':main()
