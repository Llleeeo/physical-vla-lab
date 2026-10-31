"""Conditional pre-collision translation supervision; never a runtime controller.

The auxiliary endpoint term uses teacher cap/reference/masks, supervising only
XYZ and linear feedforward. Full 29D flow supervision remains separate. Future
expert states are offline supervision, never new model observations.
"""
import numpy as np
from .ball_action_contract import decode_action

SCALE_MPS = .1
AUX_WEIGHT = .1


def command_numpy(position, feedforward, current, reference=None, relative_limit=None):
    values=[np.asarray(x,dtype=float) for x in (position,feedforward,current)]
    if any(x.shape!=(3,) or not np.isfinite(x).all() for x in values):
        raise ValueError('Expected finite XYZ vectors')
    cmd=9.*(values[0]-values[2])+values[1]
    cmd*=min(1.,1.7/max(float(np.linalg.norm(cmd)),1.e-12))
    if relative_limit is not None:
        ref=np.asarray(reference,dtype=float)
        if ref.shape!=(3,) or not np.isfinite(ref).all() or np.linalg.norm(ref)>1.7+1.e-9:
            raise ValueError('Invalid native velocity reference')
        if not np.isfinite(relative_limit) or relative_limit<0:
            raise ValueError('Invalid relative speed limit')
        diff=cmd-ref
        cmd=ref+diff*min(1.,float(relative_limit)/max(float(np.linalg.norm(diff)),1.e-12))
    return cmd


def conditional_command(position,feedforward,current,teacher):
    """Torch equivalent of frozen native velocity mapping, with fixed teacher modes."""
    import torch
    def cap(v,limit):
        norm=torch.linalg.vector_norm(v,dim=-1,keepdim=True)
        return v*torch.clamp(limit/norm.clamp_min(1.e-12),max=1.)
    if position.shape!=feedforward.shape or position.shape!=current.shape or position.shape[-1]!=3 or teacher.shape!=(*position.shape[:-1],29):
        raise ValueError('Invalid command tensor shapes')
    enabled=teacher[...,19:20]>=.5
    if not torch.isfinite(teacher).all() or not torch.isfinite(current).all():
        raise ValueError('Nonfinite teacher/state')
    if torch.any(enabled & (teacher[...,17:18]<.5)) or torch.any(enabled & ((teacher[...,18:19]<-1.e-6)|(teacher[...,18:19]>1.7+1.e-6))):
        raise ValueError('Invalid teacher relative-speed contract')
    reference=teacher[...,14:17]
    if torch.any(enabled & (torch.linalg.vector_norm(reference,dim=-1,keepdim=True)>1.7+1.e-6)):
        raise ValueError('Teacher reference exceeds absolute cap')
    cmd=cap(9.*(position-current)+feedforward,1.7)
    relative=reference+cap(cmd-reference,teacher[...,18:19].clamp(0.,1.7))
    return torch.where(enabled,relative,cmd)


def auxiliary_loss(velocity,actions,noise,times,padding,mean,std,pinch):
    """First-four valid endpoint command Huber; output is dimensionless.

    x_hat=a-t*(v-(noise-a)), denormalized once. It is a training-time endpoint
    estimate, not a sample obtained by integrating the flow model at inference.
    """
    import torch
    import torch.nn.functional as F
    if velocity.shape!=actions.shape or noise.shape!=actions.shape or actions.ndim!=3 or actions.shape[1:]!=(20,32):
        raise ValueError('Expected Bx20x32 flow tensors')
    if padding.dtype!=torch.bool or padding.shape!=actions.shape[:2] or pinch.shape!=(*padding.shape,3):
        raise ValueError('Invalid padding or corresponding future pinch states')
    if times.shape!=(len(actions),) or not torch.isfinite(times).all() or not ((times>0)&(times<=1)).all():
        raise ValueError('Invalid flow time')
    if mean.shape!=(29,) or std.shape!=(29,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std>0).all():
        raise ValueError('Invalid normalizers')
    use=(~padding).clone();use[:,4:]=False
    if not use.any():raise ValueError('No valid executed-prefix supervision')
    # Select first: padded/nonconsumed values cannot contaminate auxiliary loss.
    a=actions[...,:29][use].detach();n=noise[...,:29][use]
    v=velocity[...,:29][use];t=times[:,None].expand_as(padding)[use].unsqueeze(-1)
    teacher=(a*std+mean).detach()
    estimate=(a-t*(v-(n-a)))*std+mean
    q=pinch[use].detach()
    predicted=conditional_command(estimate[:,:3],estimate[:,7:10],q,teacher)
    target=conditional_command(teacher[:,:3],teacher[:,7:10],q,teacher).detach()
    residual=(predicted-target)/SCALE_MPS
    return F.smooth_l1_loss(residual,torch.zeros_like(residual))


def future_pinch(states,indices,padding):
    states=np.asarray(states);ids=np.asarray(indices)
    pad=np.asarray(padding)
    if states.ndim!=2 or states.shape[1]!=30 or ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or np.any(ids<0) or np.any(ids>=len(states)):
        raise ValueError('Invalid supervision indices')
    positions=ids[:,None]+np.arange(20)
    if pad.dtype!=np.bool_ or pad.shape!=positions.shape or not np.array_equal(pad,positions>=len(states)):
        raise ValueError('Future state/action mask mismatch')
    return states[np.minimum(positions,len(states)-1),14:17].copy()


def command_metrics(stream,targets,states):
    """Full predicted runtime masks/caps, NOT teacher-conditioned training term.

    A rejected request invalidates whole-stream error aggregates; never hide
    failures by reporting an error mean on only the surviving frames.
    """
    if np.shape(stream)!=np.shape(targets) or np.shape(stream)!=(269,29) or np.shape(states)!=(269,30):
        raise ValueError('Expected complete development arrays')
    errors=[];rejected=[]
    for frame,(pred,target) in enumerate(zip(stream,targets)):
        commands=[]
        for action in (target,pred):
            try:
                i=decode_action(action,frame,states[0,21:28])['inputs']
                cmd=command_numpy(i['target_position'],i['target_linear_velocity'],states[frame,14:17],i['linear_velocity_reference'],i['maximum_relative_linear_speed_mps'])
            except ValueError:
                if len(commands)==0:raise
                rejected.append(frame);cmd=np.full(3,np.nan)
            commands.append(cmd)
        errors.append(np.linalg.norm(commands[1]-commands[0]))
    error=np.asarray(errors)
    metrics={name:None if rejected else float(error[lo:hi].mean()) for name,lo,hi in [
        ('startup_command_mean_mps',0,40),('near_command_mean_mps',80,120),('full_command_mean_mps',0,269)]}
    return dict(**metrics,command_rejected_frames=rejected)
