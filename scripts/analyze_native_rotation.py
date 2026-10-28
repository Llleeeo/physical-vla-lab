#!/usr/bin/env python3
"""Compare synchronized native angular velocities with pose-derived rotation."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial.transform import Rotation


def summarize_native(data):
    dt=np.asarray(data['physics_dt_s'],dtype=float)
    if dt.ndim!=1 or len(dt)<2 or not np.isfinite(dt).all() or not np.allclose(dt,1/180,atol=1e-8,rtol=0):
        raise ValueError('native sampling cadence invalid')
    n=len(dt);step=float(dt[0]);window=min(n-1,int(round(.5/step)))
    arrays={k:np.asarray(data[k],dtype=float) for k in ['object_pose_wxyz','hand_pose_wxyz','object_velocity_world','hand_velocity_world']}
    for key,values in arrays.items():
        if values.shape!=(n,7 if 'pose' in key else 6) or not np.isfinite(values).all():raise ValueError('invalid '+key)
    op,hp=arrays['object_pose_wxyz'],arrays['hand_pose_wxyz']
    if not np.allclose(np.linalg.norm(np.r_[op[:,3:],hp[:,3:]],axis=1),1,atol=1e-4):raise ValueError('invalid quaternion norms')
    obj=Rotation.from_quat(op[:,[4,5,6,3]]);hand=Rotation.from_quat(hp[:,[4,5,6,3]])
    relative=hand.inv()*obj
    ow,hw=arrays['object_velocity_world'][:,3:],arrays['hand_velocity_world'][:,3:]
    fd_obj=(obj[1:]*obj[:-1].inv()).as_rotvec()/step
    fd_relative=(relative[1:]*relative[:-1].inv()).as_rotvec()/step
    def p95(values):return float(np.percentile(np.asarray(values)[-window:],95))
    raw=p95(np.linalg.norm(ow-hw,axis=1))
    relative_tail=relative[-window:]
    deviation=np.rad2deg((relative_tail*relative_tail.mean().inv()).magnitude())
    return {'schema':'capstone_native_rotation_diagnosis_v1','physics_samples':n,'physics_hz':1/step,
        'final_window_s':window*step,
        'final_native_relative_speed_p95_rad_s':raw,
        'unchanged_0p5_rad_s_instantaneous_criterion_pass':bool(raw<=.5),
        'final_relative_pose_derivative_p95_rad_s':p95(np.linalg.norm(fd_relative,axis=1)),
        'final_pose_net_relative_rotation_deg':float(np.rad2deg((relative[-1]*relative[-window].inv()).magnitude())),
        'final_relative_attitude_deviation_from_mean_max_deg':float(np.max(deviation)),
        'final_object_velocity_vs_backward_pose_derivative_error_p95_rad_s':p95(np.linalg.norm(ow[1:]-fd_obj,axis=1)),
        'final_midpoint_velocity_vs_pose_derivative_error_p95_rad_s':p95(np.linalg.norm((ow[1:]+ow[:-1])/2-fd_obj,axis=1)),
        'notes':['Native velocities are instantaneous; pose differences average a full step, so they are not identical measurements.',
                 'No averaging result replaces the predeclared instantaneous-speed criterion.',
                 'This probe diagnoses sampling/contact dynamics; it does not certify official benchmark success.']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args()
    with h5py.File(a.run/'cartesian_trace.hdf5') as f:
        group=f['native_rotation_probe']
        result=summarize_native({key:group[key][:] for key in group})
        if result['physics_samples']!=3*len(f['actions']):raise ValueError('incomplete native physics sampling')
        result['runtime_finger_material']=json.loads(f.attrs['runtime_finger_material_json'])
        result['actual_grasp_mode']=str(f.attrs['actual_grasp_mode'])
    output=a.run/'native_rotation_diagnosis.json'
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
