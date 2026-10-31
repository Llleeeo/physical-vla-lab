#!/usr/bin/env python3
"""Matched fixed-observation 10/30-step flow diagnostic; no training or physics."""
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
from capstone_vla.ball_sampling_diagnostic import decompose_xyz,predict_preserving_noise
from capstone_vla.ball_prediction_metrics import summarize_stream
from capstone_vla.ball_live_protocol import decoded_request
from capstone_vla.hf_archive import sha256
from capstone_vla.pilot_dataset import CAMERAS
PARENT=ROOT/'data/derived/hgwind_traj18_development_20261031/training_runs/20261031_133555_985430'
FRAMES=[0,40,80,88,96,104,112,120,128,140,144,160]
DRAWS=8
STEPS=[10,30]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group();group.add_argument('--run',action='store_true');group.add_argument('--check-load',action='store_true')
    args=parser.parse_args();torch.set_num_threads(8);torch.manual_seed(31)
    ds,cfg,stats,pin,weights,_=inspect_run(PARENT);cache=OrderedDict()
    reference=np.load(PARENT/'generated_step1000.npy',allow_pickle=False)
    if reference.shape!=(len(ds),29) or cfg.num_steps!=10:raise ValueError('Unexpected parent configuration')
    for frame in FRAMES:
        batch=batch_from(ds,[frame],cache,'cpu')
        if frame%4 or frame+4>len(ds) or batch['action_is_pad'][0,:4].any():raise ValueError('Invalid anchor prefix')
        if tuple(batch['observation.state'].shape)!=(1,2,30):raise ValueError('Invalid state history')
    if torch.cuda.is_initialized():raise ValueError('CPU preflight initialized CUDA')
    print('CPU_SAMPLING_PREFLIGHT_PASS anchors=12 draws=8 integration_steps=10,30 prefix=4 training=false physics=false',flush=True)
    if not(args.run or args.check_load):print('GPU_NOT_STARTED; use --run for paired sampling');return
    policy=build_policy(cfg,stats,pin,weights,ROOT/'data/models/smollm2_metadata'/pin['backbone_revision'])
    load_model(policy,str(PARENT/'checkpoint_final/model.safetensors'),strict=True,device='cpu');check_normalizers(policy,stats)
    if args.check_load:
        assert not torch.cuda.is_initialized()
        print('SAMPLING_STRICT_CPU_LOAD_PASS normalization=exact CUDA=false');return
    policy.to('cuda:0').eval();cache.clear()
    out=PARENT/'sampling_diagnostics'/datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True,exist_ok=False)
    print('OUTPUT '+str(out),flush=True)
    source_paths=[Path(__file__),ROOT/'src/capstone_vla/ball_sampling_diagnostic.py',ROOT/'src/capstone_vla/ball_prediction_metrics.py',ROOT/'src/capstone_vla/ball_live_protocol.py',ROOT/'scripts/finetune_hgwind_development.py',ROOT/'scripts/predict_hgwind_development.py',PARENT/'checkpoint_final/model.safetensors',PARENT/'run_config.json',PARENT/'generated_step1000.npy',Path(ds.directory)/'dataset.json',Path(ds.directory)/'arrays.npz',ROOT/'third_party/DynamicVLA/policies/dynamicvla/modeling_dynamicvla.py']
    record=dict(parent=str(PARENT),frames=FRAMES,draws=DRAWS,integration_steps=STEPS,prefix=4,noise_seed_rule='90000 + 100000*draw + frame',batch_size=1,
        source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in source_paths},scope='One exposed training episode; fixed expert observations, no feedback, no selection or deployment',noise_cloned_per_call=True)
    (out/'run_config.json').write_text(json.dumps(record,indent=2)+'\n')
    predictions=np.empty((2,len(FRAMES),DRAWS,20,29),np.float32);latencies=np.empty((2,len(FRAMES),DRAWS))
    repeatability=[];parent_differences=[]
    with torch.inference_mode():
        for i,frame in enumerate(FRAMES):
            batch=batch_from(ds,[frame],cache,'cuda:0')
            inputs={k:batch[k] for k in ['task','observation.state',*['observation.images.'+c for c in CAMERAS]]}
            for draw in range(DRAWS):
                noise=torch.randn((1,20,32),device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(90000+100000*draw+frame))
                saved=noise.clone()
                for j,steps in enumerate(STEPS):
                    torch.cuda.synchronize();start=time.perf_counter()
                    action=predict_preserving_noise(policy,inputs,noise,steps)
                    torch.cuda.synchronize();latencies[j,i,draw]=time.perf_counter()-start
                    a=action.cpu().numpy()[0]
                    if a.shape!=(20,29) or not np.isfinite(a).all():raise ValueError('Invalid prediction')
                    predictions[j,i,draw]=a
                    if not torch.equal(saved,noise):raise ValueError('Original noise mutated')
                if draw==0:
                    duplicate=predict_preserving_noise(policy,inputs,noise,10).cpu().numpy()[0]
                    repeatability.append(float(np.max(np.abs(duplicate-predictions[0,i,draw]))))
                    np.testing.assert_allclose(duplicate,predictions[0,i,draw],rtol=1e-4,atol=1e-5)
                    parent_differences.append(float(np.max(np.abs(predictions[0,i,draw,:4]-reference[frame:frame+4]))))
                    np.testing.assert_allclose(predictions[0,i,draw,:4],reference[frame:frame+4],rtol=1e-4,atol=1e-5)
            # Save completed anchors so a later failure does not discard inference work.
            np.savez_compressed(out/'predictions.npz',frames=np.asarray(FRAMES[:i+1]),actions=predictions[:,:i+1],latency_s=latencies[:,:i+1])
            print(f'SAMPLING_ANCHOR {i+1}/{len(FRAMES)} frame={frame} paired_noise=true actual_denoise_calls=verified',flush=True)
    if policy.config.num_steps!=10 or policy.model.config.num_steps!=10:raise ValueError('Integration configuration not restored')
    targets=np.stack([ds.actions[f:f+4] for f in FRAMES]);result={}
    for j,steps in enumerate(STEPS):
        p=predictions[j,:,:,:4] # anchor, draw, offset, action
        metrics=summarize_stream(p.reshape(-1,29),np.broadcast_to(targets[:,None],p.shape).reshape(-1,29))
        # Disjoint anchor/draw ordering is not a temporal stream.
        for key in ['target_gripper_transition_frames','predicted_gripper_transition_frames']:metrics.pop(key,None)
        rejects=[]
        for i,f in enumerate(FRAMES):
            for draw in range(DRAWS):
                for offset in range(4):
                    try:decoded_request(p[i,draw,offset],f+offset,ds.states[0,21:28])
                    except (ValueError,TypeError) as exc:rejects.append(dict(frame=f+offset,draw=draw,reason=str(exc)))
        groups={}
        for label,mask in [('all_selected',np.ones(len(FRAMES),bool)),('precontact_80_119',np.array([80<=f<120 for f in FRAMES]))]:
            xyz=p[mask,...,:3].transpose(1,0,2,3).reshape(DRAWS,-1,3)
            groups[label]=decompose_xyz(xyz,targets[mask,:,:3].reshape(-1,3))
        result[str(steps)]=dict(xyz_decomposition=groups,per_anchor=[dict(frame=f,**decompose_xyz(p[i,:,:,:3],targets[i,:,:3])) for i,f in enumerate(FRAMES)],
            prefix_action_metrics=metrics,native_rejected_targets=len(rejects),native_rejections=rejects,
            latency_s=dict(mean=float(latencies[j].mean()),median=float(np.median(latencies[j])),p95=float(np.percentile(latencies[j],95))))
        print('SAMPLING_RESULT '+json.dumps(dict(steps=steps,**groups['precontact_80_119'],latency_mean_s=float(latencies[j].mean()),native_rejected_targets=len(rejects))),flush=True)
    report=dict(status='BALL_SAMPLING_DIAGNOSTIC_COMPLETED',conditions=result,same_noise_repeat_max_abs=max(repeatability),parent_reference_max_abs=max(parent_differences),
        source_config_sha256=sha256(out/'run_config.json'),predictions_sha256=sha256(out/'predictions.npz'),
        training=False,physics=False,heldout_evaluation=False,task_success_verified=False,
        limitations='Empirical bias uses only eight draws; not population bias or generalization. XYZ mean is diagnostic only. No full-action averaging or sampler promotion. First invocation timing includes warmup.')
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('BALL_SAMPLING_DIAGNOSTIC_COMPLETED report='+str(out/'report.json'),flush=True)


if __name__=='__main__':main()
