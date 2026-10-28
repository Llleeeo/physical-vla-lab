#!/usr/bin/env python3
"""CPU diagnostic: endpoint/velocity semantics, not a physical simulation."""
import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import h5py
import numpy as np
from scipy.interpolate import CubicHermiteSpline

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from capstone_vla.hgwind_cohort import require,verify_files
from capstone_vla.hf_archive import sha256
from capstone_vla.recorded_franka import RecordedFrankaAdapter


def position_velocity_consistency(position,velocity,dt):
    q,v=np.asarray(position,float),np.asarray(velocity,float)
    require(q.ndim==2 and q.shape==v.shape and len(q)>=2 and np.isfinite(q).all()
            and np.isfinite(v).all() and np.isfinite(dt) and dt>0,'Invalid trajectory consistency arrays')
    residual=np.diff(q,axis=0)/dt-(v[:-1]+v[1:])/2
    return {'maximum_abs_rad_s':float(np.max(np.abs(residual))),
            'rms_rad_s':float(np.sqrt(np.mean(residual**2))),
            'per_joint_abs_p95_rad_s':np.percentile(np.abs(residual),95,axis=0).tolist()}


def diagnose(run,output):
    run=Path(run).resolve();plan=json.loads((run/'plan.json').read_text());report=json.loads((run/'report.json').read_text())
    require(plan.get('reference')=='recorded-motion','Requires a recorded-motion run')
    source=Path(plan['source']);verify_files(source.parent,plan['source_files'])
    actual_meta=json.loads((Path(report['probe_episode'])/'meta.json').read_text());d=actual_meta['arm_execution']['dynamic_drive']
    native=Path(plan['worker']).parents[1]/'sim_common/motion_safety.py'
    spec=importlib.util.spec_from_file_location('hgwind_native_motion_safety',native)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    cfg=json.loads(Path(plan['controller_config']).read_text());source_meta=json.loads((source.parent/'meta.json').read_text())
    adapter=RecordedFrankaAdapter(cfg['urdf'],source_meta['end_effector'])
    with h5py.File(source) as f:q=f['obs/joint_pos'][:,:7];v=f['obs/joint_vel'][:,:7];t=f['timestamp'][:]
    require(np.allclose(np.diff(t),1/60,atol=1e-7,rtol=0),'Source clock changed')
    # This curve is an explicitly assumed ideal measured state. It is not a
    # reconstructed 180 Hz physical trace and is never sent to the simulator.
    curve=CubicHermiteSpline(t,q,v);dt=1/180
    velocity=np.asarray(d['reference_max_velocity_rad_s']);acceleration=np.asarray(d['reference_max_acceleration_rad_s2']);jerk=np.asarray(d['reference_max_jerk_rad_s3'])
    margin=np.asarray(d['maximum_drive_position_error_rad'])
    tracker=module.DynamicDriveReferenceTracker(max_velocity=velocity,max_acceleration=acceleration,max_jerk=jerk,position_gain=12,velocity_gain=30)
    tracker.reset(q[0],v[0]);out_q=[q[0]];out_v=[v[0]];tick_times=[float(t[0])];counter=0;held_position_updates=0
    for k in range(len(q)-1):
        for j in range(3):
            now=t[k]+j*dt
            qr,vr=tracker.advance(q[k+1],v[k+1],dt,lower=adapter.lower,upper=adapter.upper,
                measured_position=curve(now),measured_velocity=curve(now,1),braking_acceleration=acceleration,
                joint_limit_margin=margin,maximum_position_error=margin)
            out_q.append(qr.copy());out_v.append(vr.copy());tick_times.append(float(now+dt));counter+=1
            if j>0 and np.max(np.abs(out_q[-1]-out_q[-2]))<1e-9 and np.max(np.abs(vr))>.01:held_position_updates+=1
    out_q,out_v,tick_times=map(np.asarray,(out_q,out_v,tick_times))
    candidate_q,candidate_v=curve(tick_times),curve(tick_times,1)
    candidate_a,candidate_j=curve(tick_times,2),curve(tick_times,3)
    violations={}
    for name,array,limit in [('velocity',candidate_v,velocity),('acceleration',candidate_a,acceleration),('within_interval_jerk',candidate_j,jerk)]:
        ratio=np.max(np.abs(array)/limit,axis=1);violations[name]={'maximum_limit_ratio':float(ratio.max()),'samples_above_limit':int(np.sum(ratio>1+1e-6)),'samples':len(ratio)}
    # Cubic Hermite interpolation is C1; acceleration may jump at each knot.
    knot_jumps=np.abs(curve(t[1:-1]+1e-9,2)-curve(t[1:-1]-1e-9,2))
    result={'schema':'physical-vla-lab.hgwind-reference-clock.v1','status':'CPU_ENDPOINT_SEMANTICS_AUDIT',
        'actual_run':{'tracking_branch':plan.get('tracking','joint'),'completed_frames':report['completed_frames'],
            'grasp_status':report['independent_grasp_evidence']['status'],'tracking':report['source_state_comparison']},
        'source_sha256':plan['source_reference_sha256'],'native_generator_sha256':sha256(native),'source_report_sha256':sha256(run/'report.json'),
        'counterfactual_model':'native generator with ideal measured motion assumed from source CubicHermiteSpline; not a physical rollout',
        'physics_rate_calls':counter,'held_position_updates_with_nonzero_velocity':held_position_updates,
        'native_endpoint_consistency':position_velocity_consistency(out_q,out_v,dt),
        'candidate_hermite_consistency':position_velocity_consistency(candidate_q,candidate_v,dt),
        'unexecuted_hermite_candidate_limits':violations,'candidate_acceleration_knot_jump_max_rad_s2':float(np.max(knot_jumps)),
        'simulation_started':False,'runtime_controls_modified':False,
        'conclusions':['The native API supplies controller endpoints plus a separately filtered velocity; it does not promise a differentiable position trajectory.',
            'The previous constructed OSC reference treated these fields as one differentiable path; that interpretation is not established.',
            'Naive interpolation is only a candidate: any limit violations and acceleration jumps must be resolved before a new physical run.',
            'This audit does not establish that reference inconsistency is the sole cause of grasp failure or a defect in the benchmark.']}
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    np.savez(output/'clock_arrays.npz',time=tick_times,endpoint_position=out_q,filtered_velocity=out_v,assumed_position=candidate_q,assumed_velocity=candidate_v)
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    verify_files(source.parent,plan['source_files'])
    print('CPU_CLOCK_AUDIT '+json.dumps({k:result[k] for k in ['physics_rate_calls','held_position_updates_with_nonzero_velocity','native_endpoint_consistency','unexecuted_hermite_candidate_limits']}))
    print('REPORT '+str(output/'report.json'));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'data/derived/hgwind_reference_clock'/datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f'))
    args=parser.parse_args();diagnose(args.run,args.output)
