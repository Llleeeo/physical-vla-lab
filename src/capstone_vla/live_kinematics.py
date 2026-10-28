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
