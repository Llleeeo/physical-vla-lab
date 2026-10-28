"""Robot-only live kinematics; no trajectory, task or object-state inputs."""
import numpy as np


def live_pinch_jacobian(expert):
    """World spatial Jacobian at the current calibrated pinch center.

    RoboPhys exposes the tool Jacobian and the tool-to-pinch conversion. Shift
    the linear rows using omega cross offset; angular rows are unchanged.
    Calls use only the simulated robot's current state and gripper geometry.
    """
    position,quaternion=expert.ee_pose()
    position=np.asarray(position,float);quaternion=np.asarray(quaternion,float)
    jacobian=np.asarray(expert.measured_ee_jacobian(),float)
    if (position.shape!=(3,) or quaternion.shape!=(4,) or jacobian.shape!=(6,7)
            or not np.isfinite(position).all() or not np.isfinite(quaternion).all()
            or not np.isclose(np.linalg.norm(quaternion),1.,atol=1e-5) or not np.isfinite(jacobian).all()):
        raise ValueError('Invalid current robot pose/Jacobian')
    pinch=np.asarray(expert.ee_target_to_pinch(position.copy(),quaternion.copy()),float)
    if pinch.shape!=(3,) or not np.isfinite(pinch).all():raise ValueError('Invalid current pinch calibration')
    shifted=jacobian.copy()
    shifted[:3]+=np.cross(jacobian[3:].T,pinch-position).T
    return shifted


def operational_task_jacobian(physical_jacobian, orientation_length_scale_m):
    """Match RoboPhys's coupled SE(3) metric: [J_v; length_scale * J_w].

    The same metric already applies to the logged angular task velocity.
    The scale is explicit controller configuration, not inferred online from
    an object pose or a stored reference Jacobian.
    """
    jacobian=np.asarray(physical_jacobian,float)
    scale=float(orientation_length_scale_m)
    if (jacobian.shape!=(6,7) or not np.isfinite(jacobian).all()
            or not np.isfinite(scale) or scale<=0):
        raise ValueError('Invalid coupled task Jacobian/length scale')
    result=jacobian.copy();result[3:]*=scale
    return result


def validate_logged_task_metric(rows, orientation_length_scale_m):
    """Franka revolute axes have unit angular norm before task weighting.

    Check all logged coupled references against the explicit metric before
    launching the simulator. This is a source-specific CPU compatibility
    check, not a runtime policy input or a generic Jacobian identity.
    """
    scale=float(orientation_length_scale_m)
    if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid task length scale')
    count=0
    for row in rows:
        ref=row['operational_space_reference']
        if ref is None:continue
        j=np.asarray(ref['jacobian'],float)
        if (j.shape!=(6,7) or not np.isfinite(j).all()
                or not np.allclose(np.linalg.norm(j[3:],axis=0),scale,atol=1e-7,rtol=0)):
            raise ValueError('Logged coupled task metric differs from configured angular length scale')
        count+=1
    if not count:raise ValueError('No coupled task references to validate')
    return count
