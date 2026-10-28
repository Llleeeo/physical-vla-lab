"""Independent capstone replay checks, not the official RoboPhys task verifier.

Privileged object/contact states are evaluator inputs only. Numerical defaults
are explicit engineering checks for the pilot, not a benchmark success label.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.spatial.transform import Rotation

@dataclass(frozen=True)
class GraspEvidenceCriteria:
    minimum_bilateral_duration_s: float = .5
    final_hold_duration_s: float = .5
    minimum_lift_m: float = .1
    maximum_relative_position_jitter_m: float = .005
    minimum_precontact_speed_m_s: float = .01
    motion_requirement: str = 'linear'
    minimum_precontact_angular_speed_rad_s: float = .1
    maximum_final_relative_spin_rad_s: float = .5
    require_lift: bool = True


def evaluate_grasp(data, criteria=None):
    c=criteria or GraspEvidenceCriteria()
    if c.motion_requirement not in ('linear','angular'):raise ValueError('unknown motion requirement')
    t=np.asarray(data['timestamp_s'],dtype=float)
    if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or not np.all(np.diff(t)>0):
        raise ValueError('invalid or unordered timestamps')
    n=len(t);dt=float(np.median(np.diff(t)))
    if not np.allclose(np.diff(t),dt,atol=1e-8,rtol=0):raise ValueError('nonuniform sample timing')
    shapes={'ee_pos':(n,3),'ee_quat':(n,4),'object_pos':(n,1,3),'object_vel':(n,1,3),
            'contact_L':(n,),'contact_R':(n,),'impulse_L':(n,),'impulse_R':(n,)}
    if c.motion_requirement=='angular':
        shapes.update(object_quat=(n,1,4),object_ang_vel=(n,1,3))
    a={}
    for key,shape in shapes.items():
        a[key]=np.asarray(data[key])
        if a[key].shape!=shape or not np.isfinite(a[key]).all():raise ValueError('invalid field '+key)
    quat=a['ee_quat']
    if not np.allclose(np.linalg.norm(quat,axis=1),1.,atol=1e-4):raise ValueError('invalid EE quaternion norms')
    left=a['contact_L'].astype(bool);right=a['contact_R'].astype(bool)
    both=left & right;any_contact=left|right;ids=np.flatnonzero(both)
    first_any=np.flatnonzero(any_contact)
    first=int(first_any[0]) if len(first_any) else None
    edges=np.diff(np.r_[False,both,False].astype(int))
    starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
    longest=int(np.max(ends-starts)) if len(starts) else 0
    window=max(1,int(np.ceil(c.final_hold_duration_s/dt)))
    enough_window=n>=window
    rel=Rotation.from_quat(quat[:,[1,2,3,0]]).inv().apply(a['object_pos'][:,0]-a['ee_pos'])
    tail=rel[-window:];jitter=float(np.max(np.linalg.norm(tail-np.median(tail,axis=0),axis=1)))
    if len(ids):
        z0=float(a['object_pos'][ids[0],0,2]);rise=float(np.min(a['object_pos'][-window:,0,2])-z0)
    else:rise=None
    speed=float(np.linalg.norm(a['object_vel'][first-1,0])) if first is not None and first>0 else None
    checks={
        'continuous_bilateral_contact':longest*dt>=c.minimum_bilateral_duration_s,
        'final_bilateral_hold':bool(enough_window and np.all(both[-window:])),
        'sustained_lift':bool(rise is not None and rise>=c.minimum_lift_m),
        'relative_position_retention':bool(enough_window and jitter<=c.maximum_relative_position_jitter_m),
        'bilateral_nonzero_impulses':bool(np.any((a['impulse_L']>0)&both) and np.any((a['impulse_R']>0)&both)),
        'object_moving_before_first_contact':None if speed is None else speed>=c.minimum_precontact_speed_m_s,
    }
    angular_speed=None;relative_spin_p95=None
    if not c.require_lift:del checks['sustained_lift']
    if c.motion_requirement=='angular':
        obj_quat=a['object_quat'][:,0]
        if not np.allclose(np.linalg.norm(obj_quat,axis=1),1.,atol=1e-4):raise ValueError('invalid object quaternion norms')
        angular_speed=float(np.linalg.norm(a['object_ang_vel'][first-1,0])) if first is not None and first>0 else None
        checks['object_moving_before_first_contact']=None if angular_speed is None else angular_speed>=c.minimum_precontact_angular_speed_rad_s
        ee_rot=Rotation.from_quat(quat[:,[1,2,3,0]])
        ee_omega=np.zeros((n,3));ee_omega[1:]=(ee_rot[1:]*ee_rot[:-1].inv()).as_rotvec()/dt
        relative_spin=np.linalg.norm(a['object_ang_vel'][:,0]-ee_omega,axis=1)
        relative_spin_p95=float(np.percentile(relative_spin[-window:],95))
        checks['final_rotation_arrest_relative_to_hand']=relative_spin_p95<=c.maximum_final_relative_spin_rad_s
    audit=data.get('grasp_assistance_audit')
    if audit is None:checks['no_grasp_assistance']=None
    else:
        audit=np.asarray(audit)
        if audit.shape!=(n,3) or not np.isfinite(audit).all():raise ValueError('invalid grasp assistance audit')
        checks['no_grasp_assistance']=bool(not np.any(audit))
    checks={key:None if value is None else bool(value) for key,value in checks.items()}
    status='fail' if any(value is False for value in checks.values()) else ('undetermined' if any(value is None for value in checks.values()) else 'pass')
    return {'schema':'capstone_grasp_evidence_v1','status':status,'official_benchmark_success':None,
            'criteria':asdict(c),'checks':checks,
            'measurements':{'frames':n,'sample_hz':1/dt,'bilateral_frames':int(both.sum()),
                'longest_bilateral_duration_s':longest*dt,'first_contact_frame':first,
                'first_bilateral_frame':int(ids[0]) if len(ids) else None,
                'minimum_final_lift_from_first_bilateral_m':rise,
                'final_relative_position_jitter_m':jitter,'object_speed_before_first_contact_m_s':speed,'object_angular_speed_before_first_contact_rad_s':angular_speed,'final_relative_spin_p95_rad_s':relative_spin_p95},
            'scope':'independent engineering check on recorded states; not full geometry/physics-generalization certification'}
