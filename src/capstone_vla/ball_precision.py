"""Bounded development objective and generated-action gates (not validation)."""
import numpy as np
from .ball_refinement import generated_metrics

CONFIG=dict(xyz_flow_weight=4.,gripper_flow_weight=4.,position_scale_m=.01,
            endpoint_weight=.1,temporal_weight=.1)


def precision_loss(velocity, actions, noise, times, padding, std, config=CONFIG):
    """Flow endpoint estimate x_t-t*v_t; never claim it is a sampled rollout.

    Temporal residual compares predicted and demonstrated increments. Thus a
    correct abrupt expert motion has zero penalty. Padding excludes both ends.
    Torch is lazy-imported so portable metric checks need no model environment.
    """
    import torch
    import torch.nn.functional as F
    if velocity.shape!=actions.shape or actions.shape!=noise.shape or actions.ndim!=3 or actions.shape[-1]!=32:
        raise ValueError('Expected identical BxTx32 flow arrays')
    if padding.dtype!=torch.bool or padding.shape!=actions.shape[:2] or times.shape!=(len(actions),):
        raise ValueError('Invalid padding/time shape')
    if not torch.isfinite(times).all() or not ((times>0)&(times<=1)).all():raise ValueError('Invalid flow times')
    if std.shape!=(29,) or not torch.isfinite(std).all() or not (std>0).all():raise ValueError('Invalid normalizer')
    valid=~padding
    if not valid.any():raise ValueError('Empty action supervision')
    residual=velocity-(noise-actions)
    weights=velocity.new_ones(29);weights[:3]=config['xyz_flow_weight'];weights[28]=config['gripper_flow_weight']
    flow=(residual[...,:29].square()*weights)[valid].sum()/(valid.sum()*weights.sum())
    # x_hat - a = -t*(v-u), normalized units -> physical units.
    error=-times[:,None,None]*residual[...,:29]*std
    xyz=error[...,:3]/config['position_scale_m']
    endpoint=F.smooth_l1_loss(xyz[valid],torch.zeros_like(xyz[valid]))
    pair=valid[:,1:]&valid[:,:-1]
    differences=xyz[:,1:]-xyz[:,:-1]
    temporal=F.smooth_l1_loss(differences[pair],torch.zeros_like(differences[pair])) if pair.any() else xyz.sum()*0
    total=flow+config['endpoint_weight']*endpoint+config['temporal_weight']*temporal
    return total,dict(weighted_flow=flow,endpoint_position=endpoint,expert_increment_residual=temporal)


def metrics(stream,targets,anchor):
    m=generated_metrics(stream,targets,anchor)
    error=np.asarray(stream[:,:3],float)-targets[:,:3]
    m['precontact_80_119_mean_mm']=float(np.linalg.norm(error[80:120],axis=1).mean()*1000)
    m['precontact_increment_error_mm']=float(np.linalg.norm(np.diff(error[79:120],axis=0),axis=1).mean()*1000)
    m['full_position_p95_mm']=m['full_stream_metrics']['position_error_mm']['p95']
    return m


def accept(candidate,baseline,best):
    if len(candidate)!=2 or len(baseline)!=2 or len(best)!=2:raise ValueError('Two fixed noise probes required')
    names=['approach_position_mean_mm','precontact_80_119_mean_mm','precontact_increment_error_mm',
           'full_position_p95_mm','orientation_p95_deg','gripper_error_frames','gripper_transitions']
    for c,b in zip(candidate,baseline):
        if any(c.get(k) is None or not np.isfinite(c[k]) for k in names):return False
        if c['native_rejected_frames'] or c['nonfinite_frames']:return False
        for k,ratio in [('approach_position_mean_mm',.9),('precontact_80_119_mean_mm',.9),('precontact_increment_error_mm',.8)]:
            if c[k]>ratio*b[k]:return False
        for k in ['full_position_p95_mm','orientation_p95_deg','gripper_error_frames','gripper_transitions']:
            if c[k]>b[k]:return False
        for k,v in b['full_stream_metrics']['mode_accuracy'].items():
            if c['full_stream_metrics']['mode_accuracy'][k]['correct']<v['correct']:return False
    return sum(c['precontact_80_119_mean_mm'] for c in candidate)<sum(c['precontact_80_119_mean_mm'] for c in best)
