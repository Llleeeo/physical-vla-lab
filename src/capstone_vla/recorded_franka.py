"""Metadata-defined Franka tool kinematics; keeps existing action encoding intact."""
import numpy as np
from scipy.spatial.transform import Rotation
from .franka import FrankaCartesianAdapter


class RecordedFrankaAdapter(FrankaCartesianAdapter):
    def __init__(self, urdf, end_effector, contract=None):
        super().__init__(urdf, contract)
        last = max(i for i, j in enumerate(self.chain) if j.attrib['type'] == 'revolute')
        reference = self.chain[last].find('child').attrib['link']
        if str(end_effector.get('link_path', '')).rsplit('/', 1)[-1] != reference:
            raise ValueError('Recorded tool must reference the last arm link: ' + reference)
        self.tool_position = np.asarray(end_effector.get('offset_position_m'), float)
        q = np.asarray(end_effector.get('offset_quat_wxyz'), float)
        if (self.tool_position.shape != (3,) or q.shape != (4,) or not np.isfinite(self.tool_position).all()
                or not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q), 1., atol=1e-6)):
            raise ValueError('Invalid recorded tool transform')
        self.tool_rotation = Rotation.from_quat(q[[1, 2, 3, 0]])
        fixed_p, fixed_r = np.zeros(3), Rotation.identity()
        for joint in self.chain[last+1:]:
            if joint.attrib['type'] != 'fixed':
                raise ValueError('Unexpected movable joint after the arm')
            origin = joint.find('origin')
            attrs = {} if origin is None else origin.attrib
            fixed_p += fixed_r.apply(np.fromstring(attrs.get('xyz', '0 0 0'), sep=' '))
            fixed_r = fixed_r * Rotation.from_euler('xyz', np.fromstring(attrs.get('rpy', '0 0 0'), sep=' '))
        self.hand_rotation_from_reference = fixed_r
        self.legacy_tip_from_reference = fixed_p + fixed_r.apply([0, 0, self.contract.ee_offset_m])
        self.frame_name = end_effector.get('frame', reference + '_recorded_tool')

    def pose_jacobian(self, joints):
        old_p, old_r, jac = super().pose_jacobian(joints)
        reference_r = old_r * self.hand_rotation_from_reference.inv()
        displacement = reference_r.apply(self.tool_position - self.legacy_tip_from_reference)
        new_jac = jac.copy()
        new_jac[:3] += np.cross(jac[3:].T, displacement).T
        return old_p + displacement, reference_r * self.tool_rotation, new_jac
