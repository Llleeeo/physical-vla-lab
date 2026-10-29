"""Episode-latched robot posture configuration; no task-object queries."""
import copy
import numpy as np
from .cartesian_target_replay import require


def sampled_posture_configuration(variation):
    switches = variation['switches']
    config = dict(schema='physical-vla-lab.robot-posture-configuration.v1',
        ik_offset_rad=list(variation['ik_seed_offset_rad']) if switches['ik_nullspace'] else [0.]*7,
        homotopy_enabled=bool(switches['path_homotopy']),
        homotopy_rank=int(variation['path_homotopy_rank']) % 8,
        strength=float(variation['strength']))
    validate_configuration(config)
    return config


def validate_configuration(config):
    require(isinstance(config,dict) and set(config)=={'schema','ik_offset_rad','homotopy_enabled','homotopy_rank','strength'}, 'Unexpected robot posture configuration fields')
    require(config['schema']=='physical-vla-lab.robot-posture-configuration.v1', 'Unknown robot posture configuration')
    offset = np.asarray(config['ik_offset_rad'], float)
    require(offset.shape==(7,) and np.isfinite(offset).all(), 'Invalid sampled robot posture offset')
    require(type(config['homotopy_enabled']) is bool and type(config['homotopy_rank']) is int
        and 0<=config['homotopy_rank']<8 and np.isfinite(config['strength'])
        and 0<=config['strength']<=1, 'Invalid sampled robot posture sector/strength')


def compute_anchor(initial_joints, config):
    validate_configuration(config)
    q = np.asarray(initial_joints, float)
    require(q.shape==(7,) and np.isfinite(q).all(), 'Invalid initial robot joints')
    # Preserve native arithmetic order: q + (IK delta + homotopy delta).
    delta = np.zeros(7)
    delta += np.asarray(config['ik_offset_rad'], float)
    if config['homotopy_enabled']:
        angle = 2*np.pi*(config['homotopy_rank']+.5)/8
        pattern = np.array([0.,np.sin(angle),0.,np.cos(angle),0.,-np.sin(angle),.5*np.cos(angle)])
        delta += .12*config['strength']*pattern
    return q + delta


def validate_anchor_bounds(anchor, lower, upper):
    anchor,lower,upper = [np.asarray(x,float) for x in (anchor,lower,upper)]
    require(all(x.shape==(7,) and np.isfinite(x).all() for x in (anchor,lower,upper))
        and np.all(lower+.02<=anchor) and np.all(anchor<=upper-.02), 'Latched robot posture outside joint-limit margin')


def initialize_anchor(expert, config):
    state = expert.arm_motion_state()
    initial = np.asarray(state.position,float).copy()
    anchor = compute_anchor(initial,config)
    validate_anchor_bounds(anchor,state.lower,state.upper)
    return dict(frame=0, configuration=copy.deepcopy(config), initial_joint_position_rad=initial.tolist(),
        lower_joint_limit_rad=np.asarray(state.lower,float).tolist(), upper_joint_limit_rad=np.asarray(state.upper,float).tolist(),
        anchor_rad=anchor.tolist())


def execution_context(expert, anchor):
    diagnostic = getattr(expert,'_last_differential_ik',{})
    q = np.asarray(expert.arm_motion_state().position,float)
    target = np.asarray(diagnostic.get('posture_target_rad'),float)
    source = diagnostic.get('posture_source')
    expected = q if source=='collision_projection' else np.asarray(anchor,float)
    require(source in ('preflight_endpoint','collision_projection') and q.shape==(7,)
        and np.isfinite(q).all() and target.shape==(7,) and np.array_equal(target,expected),
        'Native posture differs from latched robot configuration')
    return dict(measured_joint_position_rad=q.tolist(), native_posture_target_rad=target.tolist(), native_posture_source=source)


def audit_latched_posture(events, initialization):
    try:
        if not events or initialization['frame']!=0:
            return False
        anchor = compute_anchor(initialization['initial_joint_position_rad'],initialization['configuration'])
        validate_anchor_bounds(anchor,initialization['lower_joint_limit_rad'],initialization['upper_joint_limit_rad'])
        if not np.array_equal(anchor,np.asarray(initialization['anchor_rad'],float)):
            return False
        if not np.array_equal(np.asarray(initialization['initial_joint_position_rad'],float),
                np.asarray(events[0]['latched_posture_context']['measured_joint_position_rad'],float)):
            return False
        for event in events:
            context = event['latched_posture_context']
            q = np.asarray(context['measured_joint_position_rad'],float)
            source = context['native_posture_source']
            expected = q if source=='collision_projection' else anchor
            if (q.shape!=(7,) or not np.isfinite(q).all()
                    or source not in ('preflight_endpoint','collision_projection')
                    or not np.array_equal(np.asarray(event['inputs']['posture_target'],float),anchor)
                    or not np.array_equal(np.asarray(context['native_posture_target_rad'],float),expected)):
                return False
        return True
    except (KeyError,TypeError,ValueError):
        return False
