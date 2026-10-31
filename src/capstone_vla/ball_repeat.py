"""Matched exposed-case repeats; no controller intervention."""
from pathlib import Path
import json


def matched_config(reference, candidate, seed):
    # The launcher now exposes an existing model-seed setting. All runtime
    # controller/model/data hashes must still agree with the original run.
    ignored = {'output', 'worker_output', 'model_seed', 'runtime_hashes'}
    if {k:v for k,v in reference.items() if k not in ignored} != {k:v for k,v in candidate.items() if k not in ignored}:
        raise ValueError('Repeated run changes more than output paths/model seed')
    def hashes(cfg):
        return {p:h for p,h in cfg['runtime_hashes'].items()
                if not p.endswith('/scripts/run_hgwind_closed_loop.py')}
    if hashes(reference) != hashes(candidate):raise ValueError('Frozen runtime dependencies changed')
    if candidate['model_seed'] != seed:raise ValueError('Wrong model seed')


def outcome(report):
    if report.get('abort_reason') or report.get('runtime_safety_passed') is False:
        return 'runtime_abort'
    if not (report.get('full_requested_integration_passed') is True
            and report.get('initial_state_match') is True
            and report.get('model_response_binding_verified') is True
            and report.get('approach_audit',{}).get('verified') is True):
        return 'incomplete_or_invalid'
    status=(report.get('independent_grasp') or {}).get('status')
    return {'pass':'grasp_pass','fail':'grasp_fail'}.get(status,'incomplete_or_invalid')


def export_video(out):
    import subprocess
    import h5py
    import numpy as np
    from .hf_archive import sha256
    out=Path(out);source=out/'live_observations.hdf5'
    dest=out/'video_review';dest.mkdir(exist_ok=True)
    video=dest/'two_views_realtime.mp4'
    with h5py.File(source,'r') as f:
        n=min(len(f[k]) for k in ['state','opst_cam','wrist_cam'])
        if n==0:raise ValueError('No saved RGB frames')
        process=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-n',
            '-f','rawvideo','-pix_fmt','rgb24','-s','1280x480','-r','60','-i','pipe:0',
            '-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
        try:
            for i in range(n):
                frame=np.concatenate([f['opst_cam'][i],f['wrist_cam'][i]],axis=1)
                if frame.shape!=(480,1280,3) or frame.dtype!=np.uint8:raise ValueError('RGB schema changed')
                process.stdin.write(frame.tobytes())
        finally:
            process.stdin.close()
            code=process.wait()
        if code:raise RuntimeError('Video encoder failed')
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames',
        '-select_streams','v:0','-show_entries','stream=nb_read_frames,r_frame_rate',
        '-of','json',str(video)]))['streams'][0]
    if int(info['nb_read_frames'])!=n or info['r_frame_rate']!='60/1':raise ValueError('Video verification failed')
    (dest/'manifest_realtime.json').write_text(json.dumps(dict(frames=n,playback_fps=60,
        left='opst_cam',right='wrist_cam',source_sha256=sha256(source),video_sha256=sha256(video)),indent=2)+'\n')
    return str(video)
