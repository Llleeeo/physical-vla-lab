#!/usr/bin/env python3
"""Paired saved-model XYZ smoothing diagnostic; CPU by default, --run uses GPU."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo
import h5py
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from capstone_vla.ball_smoothing import smooth_actions
from capstone_vla.ball_live_protocol import decoded_request
from capstone_vla.hf_archive import sha256
from capstone_vla.cartesian_target_replay import require
from run_hgwind_closed_loop import prepare,command
from analyze_hgwind_smoothing import analyze
DEFAULT=ROOT/'data/derived/hgwind_traj18_development_20261031/training_runs/20261031_133555_985430/closed_loop_runs/20261031_140441_130052'


def read_actions(source):
    report=json.loads((source/'report.json').read_text())
    cfg=json.loads((source/'run_config.json').read_text())
    require(report['full_requested_integration_passed'] and cfg['frames']==269,'Need complete source')
    for name,digest in report['artifacts'].items():
        require(sha256(source/name)==digest,'Source artifact changed: '+name)
    with h5py.File(source/'live_observations.hdf5') as f:
        key='model_action' if cfg.get('vertical_diagnostic') else 'action'
        actions=f[key][:]
    for i in range(len(actions)):
        origin=i//cfg['prefix']*cfg['prefix']
        with np.load(source/'ipc'/f'response_{origin:06d}.npz',allow_pickle=False) as f:
            require(int(f['frame'])==origin,'Response frame changed')
            np.testing.assert_array_equal(actions[i],f['absolute'][i-origin])
    return actions,cfg,report,key


def export_review(out):
    """Encode already captured RGB; no new simulation or renderer launch."""
    videos=out/'video_review';videos.mkdir(exist_ok=True)
    with h5py.File(out/'live_observations.hdf5') as f:
        n=min(len(f['opst_cam']),len(f['wrist_cam']))
        require(n>0,'No RGB frames for video')
        cmd=['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24',
             '-s','1280x480','-r','15','-i','pipe:0','-an','-c:v','libx264',
             '-crf','20','-pix_fmt','yuv420p',str(videos/'two_views_4x_slow.mp4')]
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
        try:
            for i in range(n):proc.stdin.write(np.concatenate([f['opst_cam'][i],f['wrist_cam'][i]],axis=1).tobytes())
            proc.stdin.close();require(proc.wait()==0,'Video encoding failed')
        finally:
            if proc.poll() is None:proc.terminate();proc.wait()
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=nb_read_frames,width,height','-of','json',str(videos/'two_views_4x_slow.mp4')],text=True))['streams'][0]
    require(int(probe['nb_read_frames'])==n and probe['width']==1280 and probe['height']==480,'Video frame verification failed')
    (videos/'manifest.json').write_text(json.dumps(dict(frames=n,playback_fps=15,control_hz=60,
        source_sha256=sha256(out/'live_observations.hdf5'),video_sha256=sha256(videos/'two_views_4x_slow.mp4')),indent=2)+'\n')
    print('VIDEO '+str(videos/'two_views_4x_slow.mp4'),flush=True)


def compare(pair,source,source_report):
    reports=[json.loads((pair/name/'report.json').read_text()) for name in ('raw','smooth')]
    result=dict(scope='Fixed recorded-model command diagnostic; no online learned policy or held-out evaluation',
        cases={name:rep for name,rep in zip(('raw','smooth'),reports)},comparison_evaluable=False)
    if all(rep.get('full_requested_integration_passed') for rep in reports):
        with h5py.File(Path(reports[0]['probe_episode'])/'replay.hdf5') as a,h5py.File(Path(reports[1]['probe_episode'])/'replay.hdf5') as b,h5py.File(Path(source_report['probe_episode'])/'replay.hdf5') as old:
            keys=['joint_pos','joint_vel','object_pos','object_quat','object_vel','object_ang_vel']
            initial={k:float(np.max(abs(a['obs/'+k][0]-b['obs/'+k][0]))) for k in keys}
            prefix={k:float(np.max(abs(a['obs/'+k][:120]-old['obs/'+k][:120]))) for k in keys}
            result.update(pair_initial_max_abs_differences=initial,
                raw_vs_original_first120_max_abs_differences=prefix,
                original_prefix_reproduced=all(v<=1e-5 for v in prefix.values()),
                comparison_evaluable=all(v<=1e-5 for v in initial.values()))
        result['limits']='Initial equality does not prove full simulator determinism. If raw prefix differs from original, do not claim reproduction of the original failure. Position-only smoothing leaves velocity/acceleration feedforward and gripper unchanged; failure cannot rule out their contributions.'
    (pair/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('SMOOTHING_COMPARISON report='+str(pair/'comparison.json'),flush=True)
    for name,rep in zip(('raw','smooth'),reports):
        print(name+' '+json.dumps(dict(integration=rep.get('full_requested_integration_passed'),grasp=(rep.get('independent_grasp') or {}).get('status'),contact=(rep.get('independent_grasp') or {}).get('measurements',{}).get('first_contact_frame'))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,default=DEFAULT)
    p.add_argument('--alpha',type=float,default=.25)
    p.add_argument('--run',action='store_true')
    a=p.parse_args();source=a.source_run.resolve()
    raw,source_cfg,source_report,key=read_actions(source)
    filtered=smooth_actions(raw,a.alpha)
    require(a.alpha<1,'Experiment alpha must be below 1')
    np.testing.assert_array_equal(filtered[:,3:],raw[:,3:])
    cfg,lab,repo=prepare(Path(source_cfg['training_run']),269,4,'legacy')
    for field in ['seed','attempt_offset','task_sha256','physics_sha256','robot_posture_configuration']:
        require(cfg[field]==source_cfg[field],'Source/setup mismatch: '+field)
    anchor=json.loads((source/'cartesian_packet_execution.json').read_text())['posture_initialization']['anchor_rad']
    for actions in (raw,filtered):
        for i,action in enumerate(actions):decoded_request(action,i,anchor)
    metrics={}
    for name,actions in [('raw',raw),('smooth',filtered)]:
        jumps=np.linalg.norm(np.diff(actions[:120,:3],axis=0),axis=1)*1000
        metrics[name]=dict(first120_step_mean_mm=float(jumps.mean()),first120_step_max_mm=float(jumps.max()))
        approach=np.linalg.norm(np.diff(actions[79:120,:3],axis=0),axis=1)*1000
        metrics[name]['frames80_119_step_mean_mm']=float(approach.mean())
        metrics[name]['frames80_119_step_max_mm']=float(approach.max())
    metrics['filter_displacement_first120_mm']=dict(mean=float(np.linalg.norm(filtered[:120,:3]-raw[:120,:3],axis=1).mean()*1000),maximum=float(np.linalg.norm(filtered[:120,:3]-raw[:120,:3],axis=1).max()*1000))
    print('CPU_SMOOTHING_PREFLIGHT_PASS cases=2 frames=269 source_model_responses=exact xyz_only=true native_decode=538/538',flush=True)
    print('FILTER '+json.dumps(dict(alpha=a.alpha,metrics=metrics)),flush=True)
    print('SCOPE recorded_model_actions=true live_inference=false raw_RGB_recorded=true source_height_overlay_excluded=true',flush=True)
    if not a.run:print('GPU_NOT_STARTED; add --run');return
    stamp=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')
    pair=ROOT/'data/derived/hgwind_smoothing_checks'/stamp;pair.mkdir(parents=True,exist_ok=False)
    actions_path=pair/'recorded_actions.npz';np.savez(actions_path,actions=raw)
    manifest=dict(source_run=str(source),source_action_dataset=key,source_had_height_intervention=bool(source_cfg.get('vertical_diagnostic')),
        source_limit='Raw model predictions only. Later predictions in a hybrid source depend on its intervened observations; these are fixed-command diagnostics, not an unassisted feedback trajectory.',
        source_hashes={str(source/n):sha256(source/n) for n in ['report.json','run_config.json','live_observations.hdf5']},metrics=metrics,alpha=a.alpha)
    (pair/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('PAIR_OUTPUT '+str(pair),flush=True)
    context=json.loads((ROOT/'data/derived/hgwind_40_camera_v2_20261028/render_manifest.json').read_text())['task_context']
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TERM='xterm',OMNI_KIT_ACCEPT_EULA='YES',
        ISAACLAB_PATH=str(lab),CAPSTONE_ISAACLAB_PATH=str(lab),ISAAC_SIM_ROOT=str((lab/'_isaac_sim').resolve()),
        ROBOPHYS_CAPSTONE_OFFLINE_TASK='1',ROBOPHYS_TASK_CONTEXT=Path(context).read_text())
    env['PYTHONPATH']=os.pathsep.join([str(ROOT/'src'),*[str(x) for x in (lab/'source').iterdir() if x.is_dir()]])
    files=['scripts/run_hgwind_smoothing.py','scripts/analyze_hgwind_smoothing.py','scripts/sim/run_hgwind_smoothing_worker.py','src/capstone_vla/ball_smoothing.py','src/capstone_vla/ball_recorded_runtime.py']
    for name,alpha in [('raw',1.),('smooth',a.alpha)]:
        out=pair/name;out.mkdir()
        current=dict(cfg,output=str(out),worker_output=str(out/'worker_output'),smoothing_alpha=alpha,
            recorded_actions_path=str(actions_path),recorded_actions_sha256=sha256(actions_path),
            source_actions_used_for_control=True,learned_policy_evaluated=False,model_inference_used=False,
            standalone_policy_evaluation=False)
        current['runtime_hashes']={**cfg['runtime_hashes'],**{str(ROOT/f):sha256(ROOT/f) for f in files},str(actions_path):sha256(actions_path)}
        path=out/'run_config.json';path.write_text(json.dumps(current,indent=2)+'\n')
        env['CAPSTONE_BALL_LIVE_CONFIG']=str(path)
        cmd=command(current,lab);cmd[2]=str(ROOT/'scripts/sim/run_hgwind_smoothing_worker.py')
        print('CASE_BEGIN '+name,flush=True)
        with (out/'simulation.log').open('x') as log:
            proc=subprocess.Popen(cmd,cwd=repo,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace')
            try:
                for line in proc.stdout:
                    log.write(line);log.flush()
                    if any(x in line for x in ['BALL_','COLLECT ATTEMPT','Traceback','Error','ABORT']):print(line.rstrip(),flush=True)
                code=proc.wait()
            except BaseException:
                proc.terminate();proc.wait();raise
        report=analyze(out,code)
        try:export_review(out)
        except Exception as exc:
            (out/'video_export_error.txt').write_text(str(exc)+'\n')
            print('VIDEO_EXPORT_FAILED '+str(exc),flush=True)
        if not report.get('full_requested_integration_passed'):
            print('PAIR_STOPPED incomplete case; inspect '+str(out/'report.json'),flush=True);return
    compare(pair,source,source_report)


if __name__=='__main__':main()
