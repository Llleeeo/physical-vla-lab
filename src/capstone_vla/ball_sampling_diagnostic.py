"""Fixed-input sampling decomposition; empirical development statistics only."""
from contextlib import contextmanager
import numpy as np


def decompose_xyz(predictions, targets):
    """D draws x N targets x XYZ in metres; population variance identity."""
    p=np.asarray(predictions,dtype=float);t=np.asarray(targets,dtype=float)
    if p.ndim!=3 or t.ndim!=2 or p.shape[0]<2 or len(t)==0 or p.shape[1:]!=t.shape or t.shape[-1]!=3:
        raise ValueError('Expected D>=2 x N>0 x 3 predictions and Nx3 targets')
    if not np.isfinite(p).all() or not np.isfinite(t).all():raise ValueError('Nonfinite inputs')
    mean=p.mean(axis=0)
    bias2=float(np.mean(np.sum((mean-t)**2,axis=-1)))
    spread2=float(np.mean(np.sum((p-mean)**2,axis=-1)))
    total2=float(np.mean(np.sum((p-t)**2,axis=-1)))
    if not np.isclose(total2,bias2+spread2,rtol=1e-10,atol=1e-15):raise ValueError('Decomposition mismatch')
    return dict(draws=len(p),targets=len(t),empirical_mean_bias_rms_mm=1000*float(np.sqrt(bias2)),
        sampling_spread_rms_mm=1000*float(np.sqrt(spread2)),total_error_rms_mm=1000*float(np.sqrt(total2)),
        identity_residual_mm2=1e6*(total2-bias2-spread2))


@contextmanager
def integration_steps(policy, steps):
    """Change runtime integration count temporarily and count actual solver calls."""
    if type(steps) is not int or steps<1:raise ValueError('Positive integer steps required')
    configs=[]
    for config in (policy.config,policy.model.config):
        if not any(config is c for c,_ in configs):configs.append((config,config.num_steps))
    model=policy.model;original=model.denoise_step
    had_local='denoise_step' in vars(model);local=vars(model).get('denoise_step')
    count={'calls':0}
    def counted(*args,**kwargs):
        count['calls']+=1
        return original(*args,**kwargs)
    try:
        for config,_ in configs:config.num_steps=steps
        model.denoise_step=counted
        yield count
        if count['calls']!=steps:raise ValueError(f"Expected {steps} denoise calls, got {count['calls']}")
    finally:
        for config,old in configs:config.num_steps=old
        if had_local:model.denoise_step=local
        else:delattr(model,'denoise_step')


def predict_preserving_noise(policy, inputs, noise, steps):
    # Upstream sample_actions aliases noise and updates x_t in-place.
    independent=noise.clone() if hasattr(noise,'clone') else noise.copy()
    policy.reset()
    with integration_steps(policy,steps):
        return policy.predict_action_chunk(inputs,noise=independent)
