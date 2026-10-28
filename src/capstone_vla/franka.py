"""Absolute Franka Cartesian actions with one explicit end-effector frame.

Action: base-frame xyz metres, scipy Euler xyz radians,
and binary gripper (+1 open, -1 close). Euler xyz uses extrinsic fixed axes,
matching scipy Rotation.from_euler('xyz'). Internally orientations use SO(3).
The controller consumes robot joint measurements only, with no object state.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class FrankaActionContract:
    ee_offset_m: float = 0.1034
    damping: float = 0.01
    max_joint_target_delta_rad: float = 0.15
    finger_open_m: float = 0.04

    def __post_init__(self):
        values = [self.ee_offset_m, self.damping, self.max_joint_target_delta_rad, self.finger_open_m]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('contract constants must be finite and positive')

    def decode(self, action):
        a = np.asarray(action, dtype=float)
        if a.shape != (7,) or not np.isfinite(a).all():
            raise ValueError('action must contain seven finite values')
        if abs(a[6]) != 1.0:
            raise ValueError('binary gripper must be +1 (open) or -1 (close)')
        return a[:3].copy(), Rotation.from_euler('xyz', a[3:6]), self.finger_open_m if a[6] > 0 else 0.0

    @staticmethod
    def encode(position, rotation, gripper):
        euler = rotation.as_euler('xyz')
        euler[[0, 2]] = np.mod(euler[[0, 2]], 2*np.pi)
        result = np.r_[position, euler, float(gripper)]
        FrankaActionContract().decode(result)
        return result


class FrankaCartesianAdapter:
    def __init__(self, urdf: str | Path, contract: FrankaActionContract | None = None):
        self.contract = contract or FrankaActionContract()
        path = Path(urdf)
        self.urdf_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        root = ET.parse(path).getroot()
        by_child = {j.find('child').attrib['link']: j for j in root.findall('joint')}
        chain, link = [], 'panda_hand'
        while link != 'panda_link0':
            joint = by_child[link]
            chain.append(joint)
            link = joint.find('parent').attrib['link']
        self.chain = tuple(reversed(chain))
        self.joint_names = [j.attrib['name'] for j in self.chain if j.attrib['type'] == 'revolute']
        if self.joint_names != [f'panda_joint{i}' for i in range(1, 8)]:
            raise ValueError(f'unexpected arm chain: {self.joint_names}')
        joints = {j.attrib['name']: j for j in root.findall('joint')}
        self.lower = np.array([float(joints[n].find('limit').attrib['lower']) for n in self.joint_names])
        self.upper = np.array([float(joints[n].find('limit').attrib['upper']) for n in self.joint_names])

    def pose_jacobian(self, joints):
        q = np.asarray(joints, dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError('expected seven finite measured arm joints')
        pos, rot = np.zeros(3), Rotation.identity()
        origins, axes = [], []
        for joint in self.chain:
            origin = joint.find('origin')
            pos += rot.apply(np.fromstring(origin.attrib.get('xyz', '0 0 0'), sep=' '))
            rot = rot * Rotation.from_euler('xyz', np.fromstring(origin.attrib.get('rpy', '0 0 0'), sep=' '))
            if joint.attrib['type'] == 'revolute':
                axis = np.fromstring(joint.find('axis').attrib['xyz'], sep=' ')
                origins.append(pos.copy()); axes.append(rot.apply(axis))
                index = self.joint_names.index(joint.attrib['name'])
                rot = rot * Rotation.from_rotvec(axis*q[index])
        pos += rot.apply([0, 0, self.contract.ee_offset_m])
        jac = np.stack([np.r_[np.cross(axis, pos-origin), axis] for origin, axis in zip(origins, axes)], axis=1)
        return pos, rot, jac

    def state(self, measured_joints):
        pos, rot, _ = self.pose_jacobian(measured_joints)
        return self.contract.encode(pos, rot, 1.0)[:6]

    def joint_target(self, measured_joints, action):
        q = np.asarray(measured_joints, dtype=float)
        goal_pos, goal_rot, finger = self.contract.decode(action)
        pos, rot, jac = self.pose_jacobian(q)
        error = np.r_[goal_pos-pos, (goal_rot*rot.inv()).as_rotvec()]
        delta = jac.T @ np.linalg.solve(jac@jac.T + self.contract.damping**2*np.eye(6), error)
        # This bounds target jumps, not measured speed. PhysX actuator velocity
        # and effort caps are configured separately by the simulation adapter.
        delta = np.clip(delta, -self.contract.max_joint_target_delta_rad, self.contract.max_joint_target_delta_rad)
        target = np.clip(q+delta, self.lower, self.upper)
        return target.astype(np.float32), float(finger)
