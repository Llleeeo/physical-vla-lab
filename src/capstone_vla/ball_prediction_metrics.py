"""Decoded29D ball action checks; no dynamics/task-success claims."""
import numpy as np
from .ball_action_contract import SLICES, decode_action
from .cartesian_target_replay import require

FLAGS=('acceleration_enabled','velocity_reference_enabled','relative_speed_enabled',
       'orientation_axis_enabled','preserve_orientation_during_translation',
       'operational_orientation_task','gripper_closed')


def prefix_stream(chunks, prefix):
    chunks=np.asarray(chunks)
    require(chunks.ndim==3 and chunks.shape[-1]==29 and len(chunks)>0,'Need NxHx29 chunks')
    require(type(prefix) is int and 1<=prefix<=chunks.shape[1],'Invalid prefix')
    frames=np.arange(len(chunks));anchors=(frames//prefix)*prefix;horizons=frames-anchors
    return chunks[anchors,horizons].copy(),anchors,horizons


def distribution(values):
    x=np.asarray(values,float)
    if not len(x):return {'count':0,'mean':None,'median':None,'p95':None}
    require(np.isfinite(x).all(),'Nonfinite metric values')
    return {'count':len(x),'mean':float(x.mean()),'median':float(np.median(x)),
        'p95':float(np.percentile(x,95))}


def summarize_stream(predicted, target):
    pred=np.asarray(predicted,float);goal=np.asarray(target,float)
    require(pred.shape==goal.shape and pred.ndim==2 and pred.shape[1]==29 and len(pred)>0,
        'Expected matched Nx29 action streams')
    require(np.isfinite(goal).all(),'Nonfinite target labels')
    finite=np.isfinite(pred).all(axis=1)
    q=pred[:,3:7];qt=goal[:,3:7];norm=np.linalg.norm(q,axis=1)
    goodq=finite & (norm>1.e-6)
    require(np.all(np.abs(np.linalg.norm(qt,axis=1)-1)<1.e-5),'Invalid target quaternion')
    dots=np.abs(np.sum(q[goodq]/norm[goodq,None]*qt[goodq]/np.linalg.norm(qt[goodq],axis=1)[:,None],axis=1))
    angle=np.rad2deg(2*np.arccos(np.clip(dots,0.,1.)))
    errors=[]
    for i,a in enumerate(pred):
        try:decode_action(a,i)
        except (ValueError,TypeError) as exc:errors.append({'frame':i,'reason':str(exc)})
    result={'frames':len(pred),'finite_frames':int(finite.sum()),'nonfinite_frames':int((~finite).sum()),
        'position_error_mm':distribution(np.linalg.norm(pred[finite,:3]-goal[finite,:3],axis=1)*1000),
        'orientation_error_deg':distribution(angle),'orientation_invalid_frames':int((~goodq).sum()),
        'raw_quaternion_norm':distribution(norm[finite]),
        'decoder_rejected_frames':len(errors),'decoder_rejections':errors,
        'decoder_acceptance_is_not_physical_feasibility':True,
        'mode_accuracy':{},'scalar_projection':{},'vector_error':{}}
    for key in FLAGS:
        p=pred[:,SLICES[key]][:,0];t=goal[:,SLICES[key]][:,0]
        result['mode_accuracy'][key]={'correct':int((finite & ((p>=.5)==(t>=.5))).sum()),
            'total':len(pred),'accuracy':float((finite & ((p>=.5)==(t>=.5))).mean()),
            'target_true_count':int((t>=.5).sum()),'predicted_true_finite_count':int((finite & (p>=.5)).sum())}
    for key,upper,flag in [('orientation_roll_weight',1.,None),('joint_limit_margin_rad',.08,None),
                          ('maximum_relative_linear_speed_mps',1.7,'relative_speed_enabled')]:
        active=finite.copy()
        if flag:active &= pred[:,SLICES[flag]][:,0]>=.5
        v=pred[:,SLICES[key]][:,0]
        result['scalar_projection'][key]={'active_finite_frames':int(active.sum()),
            'clipped_frames':int((active & ((v<0)|(v>upper))).sum())}
    for key,unit,flag in [('target_linear_velocity','m/s',None),('target_linear_acceleration','m/s^2','acceleration_enabled'),
                         ('linear_velocity_reference','m/s','velocity_reference_enabled')]:
        active=finite.copy()
        if flag:active &= goal[:,SLICES[flag]][:,0]>=.5
        result['vector_error'][key]={'unit':unit,'scope':'raw_prediction_on_target_enabled_frames',
            **distribution(np.linalg.norm(pred[active,SLICES[key]]-goal[active,SLICES[key]],axis=1))}
    result['raw_per_channel_mse']=(np.mean((pred[finite]-goal[finite])**2,axis=0).tolist() if finite.any() else None)
    p=pred[:,28];t=goal[:,28]
    result['target_gripper_transition_frames']=(np.flatnonzero(np.diff(t>=.5))+1).tolist()
    result['predicted_gripper_transition_frames']=((np.flatnonzero(np.diff(p>=.5))+1).tolist() if finite.all() else None)
    return result


def summarize_chunks(chunks, targets):
    require(np.shape(chunks)==(len(targets),20,29),'Unexpected action chunk shape')
    summaries={}
    for prefix in [1,4,20]:
        stream,_,_=prefix_stream(chunks,prefix)
        summaries[str(prefix)]=summarize_stream(stream,targets)
    return {'recorded_observation_prefixes':summaries,
        'scope':'Source observation at each chunk anchor remains expert-recorded; no policy feedback simulation',
        'absolute_actions':True,'physics_feasibility_verified':False,'task_success_verified':False}
