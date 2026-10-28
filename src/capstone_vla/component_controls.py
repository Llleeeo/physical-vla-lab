"""Explicit expert-channel diagnostic controls; never a deployable VLA policy."""
import json
import numpy as np
VARIANTS={
    'expert_all':{'pose':'expert','gripper':'expert'},
    'model_pose_expert_gripper':{'pose':'model','gripper':'expert'},
    'expert_pose_model_gripper':{'pose':'expert','gripper':'model'},
}

def compose_commands(expert,model,variant):
    if variant not in VARIANTS:raise ValueError('unknown component variant')
    a,b=np.asarray(expert),np.asarray(model)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=7 or not len(a):raise ValueError('expected matching Nx7 command arrays')
    for x in [a,b]:
        if not np.isfinite(x).all() or not np.all(np.abs(x[:,6])==1):raise ValueError('finite binary-gripper commands required')
    sources=VARIANTS[variant]
    out=np.empty_like(a,dtype=np.float64)
    out[:,:6]=(a if sources['pose']=='expert' else b)[:,:6]
    out[:,6]=(a if sources['gripper']=='expert' else b)[:,6]
    return out,dict(sources)

def validate_component_bundle(prior):
    if prior.get('schema')!='capstone_component_command_bundle_v1' or prior.get('training_eligible') is not False:
        raise ValueError('invalid component diagnostic bundle')
    variant=prior.get('component_variant')
    if variant not in VARIANTS or prior.get('component_sources')!=VARIANTS[variant]:raise ValueError('component provenance mismatch')
    return dict(VARIANTS[variant])

def annotate_component_worker(source,sources):
    if sources not in VARIANTS.values():raise ValueError('invalid reference component sources')
    anchor="                    pilot_trace.attrs['expert_actions_used_for_control'] = False"
    if source.count(anchor)!=1:raise ValueError('component audit anchor mismatch')
    replacement="                    pilot_trace.attrs['expert_actions_used_for_control'] = True\n"
    replacement+="                    pilot_trace.attrs['diagnostic_component_sources_json'] = "+repr(json.dumps(sources,sort_keys=True))
    return source.replace(anchor,replacement,1)
