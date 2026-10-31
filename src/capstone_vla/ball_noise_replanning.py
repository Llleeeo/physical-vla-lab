"""Noise schedule and full-stream diagnostics; no policy promotion."""
import numpy as np
from .ball_precision import metrics
from .ball_prediction_metrics import distribution

SEEDS=tuple(90000+100000*i for i in range(8))
MODES=('independent','shared')


def noise_seed(mode,base,frame):
    if mode not in MODES or type(frame) is not int or frame<0 or frame%4:
        raise ValueError('Expected known mode and nonnegative four-frame anchor')
    return int(base)+(frame if mode=='independent' else 0)


def assemble_prefix(chunks,frames=269,prefix=4):
    a=np.asarray(chunks)
    if type(frames) is not int or frames<1 or type(prefix) is not int or not 1<=prefix<=20:
        raise ValueError('Invalid episode length/prefix')
    if a.shape!=((frames+prefix-1)//prefix,20,29) or not np.isfinite(a).all():
        raise ValueError('Invalid chunk array')
    return a[:,:prefix].reshape(-1,29)[:frames].copy()


def stream_metrics(stream,targets,anchor):
    m=metrics(stream,targets,anchor)
    p=np.asarray(stream,dtype=float);t=np.asarray(targets,dtype=float)
    error=p[:,:3]-t[:,:3]
    m['startup_0_39_position_mm']=distribution(np.linalg.norm(error[:40],axis=1)*1000)
    m['first_four_position_mm']=distribution(np.linalg.norm(error[:4],axis=1)*1000)
    dp=np.diff(p[:,:3],axis=0);dt=np.diff(t[:,:3],axis=0);frame=np.arange(1,len(p))
    m['increments']={}
    for region,region_mask in [('full',np.ones(len(frame),bool)),('precontact_80_119',(frame>=80)&(frame<120))]:
        # Each increment ends at the named frame; frame80 includes transition79->80.
        for kind,mask in [('all',np.ones(len(frame),bool)),('replan_boundary',frame%4==0),('within_chunk',frame%4!=0)]:
            use=region_mask&mask
            m['increments'][region+'_'+kind]=dict(predicted_step_mm=distribution(np.linalg.norm(dp[use],axis=1)*1000),
                expert_step_mm=distribution(np.linalg.norm(dt[use],axis=1)*1000),
                expert_increment_error_mm=distribution(np.linalg.norm(dp[use]-dt[use],axis=1)*1000))
    predicted_closed=p[:,28]>=.5;teacher_closed=t[:,28]>=.5
    closed=np.flatnonzero(predicted_closed)
    m['first_predicted_close_frame']=int(closed[0]) if len(closed) else None
    m['false_closed_frames']=np.flatnonzero(predicted_closed&~teacher_closed).tolist()
    m['false_open_frames']=np.flatnonzero(~predicted_closed&teacher_closed).tolist()
    return m
