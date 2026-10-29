"""Bounded development-case selection and saved-state video review helpers."""
import copy
import json
from pathlib import Path
import re
import subprocess
from fractions import Fraction
import h5py
import numpy as np
from .cartesian_target_replay import require
from .hgwind_cohort import TASK_ID, CAMERAS


def classify_gripper_schema(episode):
    """Read actuator-label support only; never infer binary CLOSE from a width."""
    with h5py.File(Path(episode)/'replay.hdf5','r') as handle:
        grip=handle['actions'][:,7:]
    valid=(grip.ndim==2 and grip.shape[1]==2 and len(grip)>1 and np.isfinite(grip).all())
    supported=bool(valid and np.allclose(grip[:,0],grip[:,1],atol=1.e-6,rtol=0.)
        and np.all(np.isclose(grip,0.,atol=1.e-5,rtol=0.)|np.isclose(grip,.04,atol=1.e-5,rtol=0.)))
    return dict(gripper_schema_supported=supported,
        schema_reason='binary_0_or_0.04m' if supported else 'nonbinary_or_asymmetric_actuator_targets_not_silently_converted')


def select_cases(rows, count=3):
    require(3<=count<=5, 'Use 3-5 development cases')
    candidates=sorted((r for r in rows if r['host']=='local' and r['episode']!='traj_00000' and r.get('gripper_schema_supported',False)),
        key=lambda r:int(r['attempt_index']))
    require(len(candidates)>=count, 'Not enough local development episodes')
    selected=[copy.deepcopy(candidates[i*(len(candidates)-1)//(count-1)]) for i in range(count)]
    require(len({r['numeric_fingerprint'] for r in selected})==count
        and len({r['episode_seed'] for r in selected})==count, 'Duplicated selected development conditions')
    require(all(r['depth_bias_m']==.002 for r in selected), 'Controller depth differs across cases')
    return selected


def saved_episode(root, run):
    root,run=Path(root).resolve(),Path(run).resolve()
    require(run.is_relative_to(root/'data/derived') and run!=root/'data/derived', 'Probe outside derived workspace')
    plan=json.loads((run/'plan.json').read_text())
    worker=(run/'worker_output').resolve()
    require(Path(plan['worker_output']).resolve()==worker, 'Worker output escaped run directory')
    episodes=[p.parent for p in worker.glob('1s1p/'+TASK_ID+'/*/replay.hdf5')]
    require(len(episodes)==1, 'Expected exactly one saved episode, regardless of success label')
    episode=episodes[0].resolve()
    require(episode.is_relative_to(worker) and re.fullmatch(r'(traj|success|failed)_\d+',episode.name),
        'Unexpected episode location/name')
    meta=json.loads((episode/'meta.json').read_text())
    require(meta['task']['id']==TASK_ID and int(meta['num_steps'])>=1, 'Wrong task or empty saved episode')
    report_path=run/'report.json'
    if report_path.is_file():
        report=json.loads(report_path.read_text())
        require(Path(report['probe_episode']).resolve()==episode, 'Report points to another saved episode')
    return plan,episode,meta


def inspect_review_videos(root,run):
    _,episode,meta=saved_episode(root,run)
    video=Path(run)/'visualization/Trajectory/1s1p'/TASK_ID/episode.name/'video'
    paths={name:video/(name+'_rgb_vis.mp4') for name in CAMERAS}
    require(all(p.is_file() and p.stat().st_size>0 for p in paths.values()), 'Missing one or more RGB review videos')
    with h5py.File(episode/'replay.hdf5','r') as source,h5py.File(video/'rendered.hdf5','r') as visual:
        indices=visual['source_frame_index'][:]
        require(np.array_equal(indices,np.arange(0,int(meta['num_steps']),2)), 'Review render does not cover the 30 Hz source-state clock')
        require(float(visual.attrs['fps'])==30. and np.allclose(
            visual['timestamp_s'][:],source['timestamp'][:][indices],atol=1.e-7,rtol=0.), 'Review timestamps/FPS mismatch')
        for camera in CAMERAS:
            require(visual[camera+'_rgb'].shape==(len(indices),480,640,3), 'Review RGB shape differs')
    streams={}
    for camera,path in paths.items():
        data=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-count_frames','-show_entries','stream=width,height,nb_read_frames,r_frame_rate',
            '-of','json',str(path)],text=True))['streams']
        require(len(data)==1, 'Expected one video stream')
        stream=data[0]
        require(int(stream['nb_read_frames'])==len(indices) and
            (stream['width'],stream['height'])==(640,480) and
            float(Fraction(stream['r_frame_rate']))==30., 'Truncated/incorrect preview video')
        streams[camera]=dict(path=str(path.resolve()),frames=len(indices),fps=30,
            camera=camera,manual_reviewed=False)
    return dict(source_frames=int(meta['num_steps']),rendered_frames=len(indices),
        saved_state_visualization=True,new_physics_rollout=False,training_render=False,videos=streams)


def write_review_index(output,state):
    output=Path(output)
    rows=[]
    for record in state['existing']:
        rows.append((record['label'],record))
    for case in state['cases']:
        for stage in ['teacher','contract']:
            rows.append((case['episode']+'/'+stage,case.get(stage,{})))
    lines=['# Saved-state video review','',
        'Every saved success/failure is rendered. These are 30 Hz inspection videos, not training pixels.',
        'Independent grasp result and collector filename are separate; pending/failed rendering is listed explicitly.','',
        '| Run | Independent grasp | Video status | Front | Left | Right | Overhead | Wrist |',
        '|---|---|---|---|---|---|---|---|']
    errors=[]
    for label,record in rows:
        videos=record.get('video_check',{}).get('videos',{})
        links=['['+name+']('+videos[name]['path']+')' if name in videos else '—' for name in CAMERAS]
        lines.append('| '+label+' | '+str(record.get('grasp','not_evaluated'))+' | '+
            record.get('render_state',record.get('state','pending'))+' | '+' | '.join(links)+' |')
        if record.get('error'):
            errors.append('- '+label+': '+str(record['error']).replace('\n',' '))
    if errors:lines.extend(['']+errors)
    lines.extend(['','Inspect approach, moving-ball interception, closure depth, lift/retention, jitter, and ball visibility.',
        'Record human judgements in MANUAL_REVIEW.md; this index is regenerated on resume.',''])
    (output/'VIDEO_REVIEW.md').write_text('\n'.join(lines))
    manual=output/'MANUAL_REVIEW.md'
    if not manual.exists():
        manual.write_text('# Human review\n\n| Run | Reasonable motion? | Ball retained? | Views adequate? | Notes |\n|---|---|---|---|---|\n'+
            '\n'.join('| '+label+' | | | | |' for label,_ in rows)+'\n')
