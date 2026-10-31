#!/usr/bin/env python3
"""CPU contact chronology; --run adds a matched-noise GPU input diagnostic, no physics."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from analyze_hgwind_effort_probe import control_contact_arrays
from capstone_vla.hf_archive import sha256
from capstone_vla.ball_input_metrics import position_change_mm

DEFAULT = ROOT/'data/derived/hgwind_traj18_development_20261031/training_runs/20261031_121355_529372/closed_loop_runs/20261031_125333_149097'
FRAMES = [0, 8, 20, 40, 60, 80, 100, 104]


def transitions(values):
    return (np.flatnonzero(np.diff(np.asarray(values) >= .5))+1).tolist()


def first(mask):
    ids = np.flatnonzero(mask)
    return int(ids[0]) if len(ids) else None


def inspect(run):
    cfg = json.loads((run/'run_config.json').read_text())
    report = json.loads((run/'report.json').read_text())
    assert report['full_requested_integration_passed'] and report['completed_frames'] == 269
    assert cfg['prefix'] == 4 and cfg['history_offsets'] == [-2, 0]
    for name, digest in report['artifacts'].items():
        assert sha256(run/name) == digest, name
    training = Path(cfg['training_run'])
    record = json.loads((training/'run_config.json').read_text())
    dataset = Path(record['source_dataset'])
    meta = json.loads((dataset/'dataset.json').read_text())
    arrays = np.load(dataset/'arrays.npz', allow_pickle=False)
    source = Path(cfg['source'])
    sm = json.loads(source.with_name('meta.json').read_text())
    target = sm['policy']['roles']['A']
    live_source = Path(report['probe_episode'])/'replay.hdf5'
    lm = json.loads(live_source.with_name('meta.json').read_text())
    si, li = sm['object_paths'].index(target), lm['object_paths'].index(target)
    with h5py.File(source) as s, h5py.File(live_source) as l, h5py.File(run/'live_observations.hdf5') as obs:
        t = l['timestamp'][:]
        np.testing.assert_allclose(t, s['timestamp'][:], rtol=0, atol=1e-7)
        assert len(t) == 269 and np.isfinite(obs['action'][:]).all()
        sp, lp = s['obs/ee_pos'][:], l['obs/ee_pos'][:]
        so, lo = s['obs/object_pos'][:, si], l['obs/object_pos'][:, li]
        sc = np.stack(control_contact_arrays(s, target, t), axis=1)
        lc = np.stack(control_contact_arrays(l, target, t), axis=1)
        jaw = obs['action'][:, 28]
        teacher_jaw = arrays['actions'][:, 28]
        ep = np.linalg.norm(lp-sp, axis=1)*1000
        op = np.linalg.norm(lo-so, axis=1)*1000
        cframe = first(np.any(lc > 1e-12, axis=1))
        result = dict(schema='physical-vla-lab.ball-live-failure.v1', integration_passed=True,
            grasp_passed=False, heldout_evaluation=False, model_probe_executed=False,
            chronology=dict(first_live_contact_frame=cframe,
                first_teacher_contact_frame=first(np.any(sc > 1e-12, axis=1)),
                first_teacher_bilateral_frame=first(np.all(sc > 1e-12, axis=1)),
                live_bilateral_frames=int(np.all(lc > 1e-12, axis=1).sum()),
                live_left_contact_frames=int((lc[:, 0] > 1e-12).sum()),
                live_right_contact_frames=int((lc[:, 1] > 1e-12).sum()),
                live_jaw_transitions=transitions(jaw), teacher_jaw_transitions=transitions(teacher_jaw),
                first_ball_position_difference_above_1mm_frame=first(op > 1),
                ee_difference_at_first_contact_mm=float(ep[cframe]) if cframe is not None else None,
                jaw_requested_closed_at_first_contact=bool(jaw[cframe] >= .5) if cframe is not None else None,
                finger_positions_at_first_contact_m=l['obs/joint_pos'][cframe, 7:].tolist() if cframe is not None else None),
            checkpoints=[dict(frame=i, ee_difference_mm=float(ep[i]), ball_difference_mm=float(op[i]),
                live_ball_minus_ee_mm=((lo[i]-lp[i])*1000).tolist(),
                teacher_ball_minus_ee_mm=((so[i]-sp[i])*1000).tolist()) for i in [0, 40, 80, 104, 106, 130, 144, 162, 268]],
            interpretation=('No recorded finger contact; inspect approach geometry and closure timing.' if cframe is None else 'Contact detected; compare its chronology with jaw transitions and approach deviation before assigning a failure cause.'),
            limits='Same-time expert comparisons after diverging states are descriptive, not oracle corrective actions. Hybrid image/state inputs are sensitivity probes and may be inconsistent; no causal success claim.')
        # Inspect camera calibration only at the exactly matched initial robot/object state.
        with h5py.File(meta['images']) as image:
            sn = json.loads(image.attrs['camera_names_json'])
            ln = json.loads(obs.attrs['camera_pose_order_json'])
            cameras = {}
            for name in meta['cameras']:
                j, k = sn.index(name), ln.index(name)
                x, y = obs[name][0].astype(float), image[name+'_rgb'][0].astype(float)
                p = obs['camera_pose_world'][0, k]
                q = image['camera_quaternion_world_wxyz'][0, j].astype(float)
                dot = abs(np.dot(p[3:]/np.linalg.norm(p[3:]), q/np.linalg.norm(q)))
                cameras[name] = dict(initial_position_difference_mm=float(np.linalg.norm(p[:3]-image['camera_position_world'][0,j])*1000),
                    initial_orientation_difference_deg=float(np.degrees(2*np.arccos(np.clip(dot,0,1)))),
                    initial_rgb_mae_255=float(abs(x-y).mean()), live_mean_255=float(x.mean()), training_mean_255=float(y.mean()))
            result['initial_camera_comparison'] = cameras
        plot = dict(t=t, ep=ep, op=op, jaw=jaw, teacher_jaw=teacher_jaw, lc=lc, sc=sc)
    result['source_sha256'] = {str(p): sha256(p) for p in [run/'report.json', run/'run_config.json', source,
        source.with_name('meta.json'), dataset/'arrays.npz', dataset/'dataset.json', Path(__file__)]}
    result['live_artifacts_sha256'] = report['artifacts']
    return result, plot, cfg, meta, arrays


def model_probe(run, cfg, meta, arrays, out):
    import torch
    from safetensors.torch import load_model
    from hgwind_policy_server import prepare_input
    from predict_hgwind_development import inspect_run, check_normalizers
    from finetune_hgwind_development import build_policy
    torch.set_num_threads(8); torch.manual_seed(0)
    training = Path(cfg['training_run'])
    ds, pcfg, stats, pin, weights, _ = inspect_run(training)
    for path, digest in cfg['checkpoint_hashes'].items():
        assert sha256(path) == digest
    policy = build_policy(pcfg, stats, pin, weights, ROOT/'data/models/smollm2_metadata'/pin['backbone_revision'])
    load_model(policy, str(training/'checkpoint_final/model.safetensors'), strict=True, device='cpu')
    check_normalizers(policy, stats); policy.to('cuda:0').eval()
    rows = []; predictions = []
    with h5py.File(run/'live_observations.hdf5') as live, h5py.File(meta['images']) as images:
        for frame in FRAMES:
            ids = [max(0, frame-2), frame]
            states = {'teacher': arrays['states'][ids], 'live': np.stack([live['state'][i] for i in ids])}
            rgb = {'teacher': {c: np.stack([images[c+'_rgb'][i] for i in ids]) for c in meta['cameras']},
                   'live': {c: np.stack([live[c][i] for i in ids]) for c in meta['cameras']}}
            # At frame0 both state histories match exactly: image contrast is unconfounded by robot state.
            if frame == 0: np.testing.assert_array_equal(states['teacher'], states['live'])
            with np.load(run/'ipc'/f'request_{frame:06d}.npz') as req:
                np.testing.assert_array_equal(states['live'], req['states'])
                for c in meta['cameras']: np.testing.assert_array_equal(rgb['live'][c], req[c])
            samples = {}
            for im, st in [('teacher','teacher'), ('live','teacher'), ('teacher','live'), ('live','live')]:
                inputs = prepare_input(states[st], rgb[im], meta['instruction'], 'cuda:0')
                noise = torch.randn((1,20,32), device='cuda:0', generator=torch.Generator(device='cuda:0').manual_seed(cfg['model_seed']+frame))
                policy.reset()
                with torch.inference_mode(): a = policy.predict_action_chunk(inputs, noise=noise).cpu().numpy()[0]
                assert a.shape == (20,29) and np.isfinite(a).all()
                name = im+'_images_'+st+'_state'; samples[name] = a
                predictions.append(a)
                rows.append(dict(frame=frame, condition=name,
                    first_target_position_difference_from_teacher_mm=float(np.linalg.norm(a[0,:3]-arrays['actions'][frame,:3])*1000),
                    first_gripper_raw=float(a[0,28]), first_four_gripper_closed=(a[:4,28]>=.5).tolist()))
            # Validate rerun fidelity before interpreting interventions. Do not silently continue with a different forward path.
            with np.load(run/'ipc'/f'response_{frame:06d}.npz') as saved:
                np.testing.assert_allclose(samples['live_images_live_state'], saved['absolute'], rtol=1e-4, atol=1e-5)
                rows[-1]['saved_live_chunk_max_abs_difference'] = float(np.max(abs(samples['live_images_live_state']-saved['absolute'])))
            base = samples['teacher_images_teacher_state']
            for row in rows[-4:]:
                a = samples[row['condition']]
                row['first_four_target_position_difference_from_teacher_input_mm'] = position_change_mm(a[:4,:3],base[:4,:3]).tolist()
            print(f'INPUT_PROBE frame={frame} conditions=4 saved_live_response=matched', flush=True)
    np.savez(out/'input_probe_chunks.npz', chunks=np.stack(predictions), frames=np.repeat(FRAMES,4))
    return dict(conditions=4, anchors=FRAMES, rows=rows,
        first_frame_identical_state_verified=True, saved_live_responses_matched=True,
        noise='same 90000+frame across four conditions; batch size1 as live server',
        checkpoint_hashes=cfg['checkpoint_hashes'])



def summarize_saved_probe(directory):
    """Recompute units from immutable chunks without rerunning model inference."""
    directory=Path(directory).resolve();run=directory.parents[1]
    report=json.loads((directory/'report.json').read_text())
    cfg=json.loads((run/'run_config.json').read_text())
    assert report['model_probe_executed'] and report['input_probe']['anchors']==FRAMES
    with np.load(directory/'input_probe_chunks.npz',allow_pickle=False) as saved:
        chunks=saved['chunks'].copy();np.testing.assert_array_equal(saved['frames'],np.repeat(FRAMES,4))
    assert chunks.shape==(32,20,29) and np.isfinite(chunks).all()
    values=chunks.reshape(8,4,20,29);conditions=['teacher_images_teacher_state','live_images_teacher_state','teacher_images_live_state','live_images_live_state']
    rows=[]
    for i,frame in enumerate(FRAMES):
        with np.load(run/'ipc'/f'response_{frame:06d}.npz') as saved:
            np.testing.assert_allclose(values[i,3],saved['absolute'],rtol=1e-4,atol=1e-5)
        for j,condition in enumerate(conditions):
            old=report['input_probe']['rows'][4*i+j]
            assert old['frame']==frame and old['condition']==condition
            row=dict(old)
            row['first_four_target_position_difference_from_teacher_input_mm']=position_change_mm(values[i,j,:4,:3],values[i,0,:4,:3]).tolist()
            rows.append(row)
    def stats(j):
        changes=position_change_mm(values[:,j,:4,:3],values[:,0,:4,:3])
        return dict(mean_mm=float(changes.mean()),max_mm=float(changes.max()),count=int(changes.size))
    result=dict(schema='physical-vla-lab.ball-input-probe-verified.v1',anchors=FRAMES,rows=rows,
        image_only_change=stats(1),state_only_change=stats(2),combined_change=stats(3),
        saved_live_responses_matched=True,grasp_success=False,heldout_evaluation=False,
        units_correction='Original report difference field named mm contained metres; recomputed here from saved arrays in millimetres. Original report retained. Other target-error fields were already mm.',
        limits=report['limits'],source_sha256={str(path):sha256(path) for path in [directory/'report.json',directory/'input_probe_chunks.npz',run/'run_config.json',Path(__file__)]})
    (directory/'verified_input_metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('SAVED_INPUT_METRICS_VERIFIED '+json.dumps({k:result[k] for k in ['image_only_change','state_only_change','combined_change']}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--live-run',type=Path,default=DEFAULT);p.add_argument('--run',action='store_true',help='Run GPU input sensitivity inference; no simulation/training')
    p.add_argument('--summarize-probe',type=Path,help='Recompute a completed probe from saved arrays on CPU')
    a=p.parse_args()
    if a.summarize_probe:
        if a.run:p.error('--summarize-probe cannot launch inference')
        summarize_saved_probe(a.summarize_probe);return
    run=a.live_run.resolve()
    result, plot, cfg, meta, arrays=inspect(run)
    out=run/'failure_diagnostics'/datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    try:
        import matplotlib
    except ImportError:
        print('PLOT_SKIPPED matplotlib unavailable; numerical audit unaffected', flush=True)
    else:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(3,1,figsize=(11,9),sharex=True)
        ax[0].plot(plot['t'],plot['ep'],label='EE position difference');ax[0].plot(plot['t'],plot['op'],label='Ball position difference')
        ax[0].set_ylabel('Difference from expert (mm)');ax[0].legend()
        ax[1].plot(plot['t'],plot['jaw'],label='Live gripper output');ax[1].step(plot['t'],plot['teacher_jaw'],where='post',label='Expert close request')
        ax[1].axhline(.5,color='gray',ls=':');ax[1].set_ylabel('Close output / threshold');ax[1].legend()
        ax[2].plot(plot['t'],plot['lc'][:,0],label='Live left');ax[2].plot(plot['t'],plot['lc'][:,1],label='Live right')
        ax[2].set_ylabel('Contact impulse (N s)');ax[2].set_xlabel('Simulation time (s)');ax[2].legend()
        for axis in ax:
            if result['chronology']['first_live_contact_frame'] is not None:
                axis.axvline(result['chronology']['first_live_contact_frame']/60,color='red',ls='--',alpha=.5)
            axis.grid(alpha=.2)
        fig.suptitle('Learned rollout: approach, gripper commands and recorded contacts');fig.tight_layout();fig.savefig(out/'contact_chronology.png',dpi=150);plt.close(fig)
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('CPU_FAILURE_AUDIT_PASS '+json.dumps(result['chronology']),flush=True)
    print('OUTPUT '+str(out),flush=True)
    if a.run:
        result['input_probe']=model_probe(run,cfg,meta,arrays,out)
        result['model_probe_executed']=True
        (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print('BALL_INPUT_DIAGNOSTIC_COMPLETED report='+str(out/'report.json'),flush=True)
    else:print('GPU_NOT_STARTED; --run enables matched-input model inference only',flush=True)


if __name__=='__main__':main()
