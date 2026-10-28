"""Narrow, fail-closed bridge to the isolated RoboPhys replay worker.

Worker source is patched in memory only. Its scene/scripts/contact recorder are
reused; robot targets are controlled by the capstone action contract.
"""
import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import numpy as np
from .franka import FrankaCartesianAdapter, FrankaActionContract


def configuration():
    return json.loads(Path(os.environ['CAPSTONE_FRANKA_CONFIG']).read_text())


def configure_actuators(robot, arm_indices, metadata):
    from tools.pilot_recorded_actuators import restore_recorded_franka_pd
    cfg = configuration()
    motor = cfg['actuators']
    meta = copy.deepcopy(metadata)
    profile=os.environ.get('CAPSTONE_ACTUATOR_PROFILE','capstone')
    if profile not in ('capstone','recorded-diagnostic'):raise ValueError('unknown actuator profile')
    if profile=='capstone':meta['actuator_safety'].update({
        'stiffness': motor['stiffness'], 'damping': motor['damping'],
        'max_effort_nm': motor['effort_limit_nm'],
        'max_velocity_rad_s': motor['velocity_limit_rad_s'],
        'disable_gravity': motor['disable_gravity'],
    })
    result = restore_recorded_franka_pd(robot, arm_indices, meta)
    fingers = np.array([robot.dof_names.index(f'panda_finger_joint{i}') for i in (1,2)], dtype=np.int32)
    controller = robot.get_articulation_controller()
    kp,kd = (np.asarray(v).reshape(-1).copy() for v in controller.get_gains())
    kp[fingers] = motor['finger_stiffness']; kd[fingers] = motor['finger_damping']
    controller.set_gains(kp,kd)
    view = robot._articulation_view
    view.set_max_efforts(np.full((1,2),motor['finger_effort_limit_n'],dtype=np.float32),joint_indices=fingers)
    actual_kp,actual_kd = (np.asarray(v).reshape(-1) for v in controller.get_gains())
    effort = np.asarray(view.get_max_efforts(joint_indices=fingers)).reshape(-1)
    if not (np.allclose(actual_kp[fingers],motor['finger_stiffness'])
            and np.allclose(actual_kd[fingers],motor['finger_damping'])
            and np.allclose(effort,motor['finger_effort_limit_n'])):
        raise RuntimeError('capstone finger actuator readback mismatch')
    result.update(profile=('capstone_franka_action_interface_v1' if profile=='capstone' else 'recorded_arm_pd_diagnostic'),source=('capstone config' if profile=='capstone' else 'episode metadata arm parameters'),
                  finger_settings_readback_passed=True,velocity_feedforward='zero',
                  source_180hz_reference_reconstructed=False)
    return result


def bounded_dls(jac, desired, damping, lower, upper):
    """Bounded least squares; limits participate in solving, not post-clipping."""
    from scipy.optimize import lsq_linear
    matrix=np.vstack([jac,damping*np.eye(jac.shape[1])])
    result=lsq_linear(matrix,np.r_[desired,np.zeros(jac.shape[1])],bounds=(lower,upper),
                      tol=1e-8,max_iter=100)
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError('bounded IK failed: '+result.message)
    return result.x


@dataclass
class ConstraintRecovery:
    """Pilot-calibrated tracking recovery, independent of time and task phase."""
    position_enter_m: float = .025
    position_exit_m: float = .0125
    orientation_enter_rad: float = np.deg2rad(5.)
    orientation_exit_rad: float = np.deg2rad(2.5)
    active: bool = False

    def update(self, position_error, orientation_error, constrained):
        if not np.isfinite([position_error,orientation_error]).all():
            raise ValueError('nonfinite recovery input')
        if self.active:
            if position_error < self.position_exit_m and orientation_error < self.orientation_exit_rad:
                self.active=False
        elif constrained and (position_error > self.position_enter_m or orientation_error > self.orientation_enter_rad):
            self.active=True
        return self.active


def grasp_joint_candidates(stage, robot_path, object_paths):
    """List direct robot/object constraint joints, including disabled candidates."""
    from pxr import UsdPhysics
    def under(path,root):
        return str(path)==str(root) or str(path).startswith(str(root).rstrip('/')+'/')
    found=[]
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):continue
        joint=UsdPhysics.Joint(prim)
        bodies=[str(x) for x in [*joint.GetBody0Rel().GetTargets(),*joint.GetBody1Rel().GetTargets()]]
        if any(under(body,robot_path) for body in bodies) and any(
                under(body,obj) or under(obj,body) for body in bodies for obj in object_paths):
            found.append(str(prim.GetPath()))
    return found


class SceneController:
    def __init__(self, robot, arm_indices, mode):
        cfg=configuration()
        keys=('ee_offset_m','damping','max_joint_target_delta_rad','finger_open_m')
        self.adapter=FrankaCartesianAdapter(cfg['urdf'],FrankaActionContract(**{k:cfg['action'][k] for k in keys}))
        self.ik_solver=os.environ.get('CAPSTONE_IK_SOLVER','clipped')
        self.bounded_from_command=int(os.environ.get('CAPSTONE_BOUNDED_FROM_COMMAND','0'))
        if self.bounded_from_command<0:raise ValueError('negative bounded solver switch index')
        if self.ik_solver not in ('clipped','bounded','adaptive'):raise ValueError('unknown IK solver')
        if self.ik_solver=='adaptive' and self.bounded_from_command!=0:
            raise ValueError('adaptive solver forbids a scheduled frame switch')
        self.recovery=ConstraintRecovery()
        self.robot=robot; self.arm=np.asarray(arm_indices,dtype=int); self.mode=mode
        self.fingers=np.array([robot.dof_names.index(f'panda_finger_joint{i}') for i in (1,2)])
        self.source=None; self.action=None; self.servo_calls=0
        self.physics_dt_samples=[]; self.callback_error=None
        self.velocity_reference=os.environ.get('CAPSTONE_VELOCITY_REFERENCE','zero')
        if self.velocity_reference not in ('zero','causal','recorded-diagnostic'):
            raise ValueError('unknown velocity reference')
        self.velocity_limits=np.asarray(cfg['actuators']['velocity_limit_rad_s'])
        self.command_hz=float(cfg['timing']['command_hz'])
        self.previous_source=None; self.previous_pose=None
        self.reference_joint_velocity=np.zeros(7); self.reference_twist=np.zeros(6)
        self.velocity_commands=[]
        self.diagnostics={name:[] for name in ('command_index','q','qd','position_target','velocity_target',
            'solver_projected_effort_nm','nominal_pd_effort_nm','unclipped_ik_delta_rad',
            'ik_delta_clipped','target_limit_clipped','velocity_clipped','jacobian_min_singular_value',
            'goal_position_error_m','goal_orientation_error_rad','bounded_active')}
        self.command_index=-1
        gains=self.robot.get_articulation_controller().get_gains() if hasattr(self.robot,'get_articulation_controller') else None
        self.kp=np.asarray(gains[0]).reshape(-1)[self.arm] if gains is not None else np.asarray(cfg['actuators']['stiffness'])
        self.kd=np.asarray(gains[1]).reshape(-1)[self.arm] if gains is not None else np.asarray(cfg['actuators']['damping'])

    def set_source(self, target, recorded_velocity=None):
        self.command_index+=1
        self.source=np.asarray(target,dtype=np.float32).copy()
        # Convert once at command cadence, then hold the Cartesian action.
        p,r,_=self.adapter.pose_jacobian(self.source[self.arm])
        values=self.source[self.fingers]
        grip=1. if values.mean()>.02 else -1.
        if not (np.all(np.isclose(values,0,atol=1e-5)) or np.all(np.isclose(values,.04,atol=1e-5))):
            raise ValueError('source has nonbinary/asymmetric gripper; explicit conversion required')
        if self.previous_source is not None:
            self.reference_joint_velocity=(self.source[self.arm]-self.previous_source[self.arm])*self.command_hz
            previous_p,previous_r=self.previous_pose
            self.reference_twist=np.r_[p-previous_p,(r*previous_r.inv()).as_rotvec()]*self.command_hz
        if self.velocity_reference=='recorded-diagnostic':
            if self.mode!='joint':raise ValueError('recorded velocity probe is joint-only')
            values=np.asarray(recorded_velocity,dtype=float)
            if values.shape!=self.source.shape or not np.isfinite(values).all():
                raise ValueError('recorded velocity probe requires complete finite drive reference')
            self.reference_joint_velocity=values[self.arm].copy()
        self.previous_source=self.source.copy(); self.previous_pose=(p.copy(),r)
        self.action=self.adapter.contract.encode(p,r,grip)
        return self.target()

    def use_bounded(self, measured):
        if self.ik_solver=='bounded':return self.command_index>=self.bounded_from_command
        if self.ik_solver!='adaptive':return False
        pos,rot,jac=self.adapter.pose_jacobian(measured)
        goal,goal_rot,_=self.adapter.contract.decode(self.action)
        error=np.r_[goal-pos,(goal_rot*rot.inv()).as_rotvec()]
        raw=jac.T@np.linalg.solve(jac@jac.T+self.adapter.contract.damping**2*np.eye(6),error)
        bound=self.adapter.contract.max_joint_target_delta_rad
        constrained=bool(np.any(np.abs(raw)>bound) or np.any(measured+raw<self.adapter.lower)
                         or np.any(measured+raw>self.adapter.upper))
        return self.recovery.update(np.linalg.norm(error[:3]),np.linalg.norm(error[3:]),constrained)

    def target(self):
        result=self.source.copy()
        if self.mode=='cartesian':
            measured=np.asarray(self.robot.get_joint_positions())[self.arm]
            result[self.arm],finger=self.adapter.joint_target(measured,self.action)
            if self.use_bounded(measured):
                p,rot,jac=self.adapter.pose_jacobian(measured)
                goal,goal_rot,_=self.adapter.contract.decode(self.action)
                error=np.r_[goal-p,(goal_rot*rot.inv()).as_rotvec()]
                bound=self.adapter.contract.max_joint_target_delta_rad
                lower=np.maximum(-bound,self.adapter.lower-measured)
                upper=np.minimum(bound,self.adapter.upper-measured)
                result[self.arm]=measured+bounded_dls(jac,error,self.adapter.contract.damping,lower,upper)
            result[self.fingers]=finger
        return result

    def velocity_target(self):
        result=np.zeros_like(self.source)
        if self.velocity_reference in ('causal','recorded-diagnostic'):
            velocity=self.reference_joint_velocity.copy()
            if self.mode=='cartesian':
                q=np.asarray(self.robot.get_joint_positions())[self.arm]
                _,_,jac=self.adapter.pose_jacobian(q)
                velocity=jac.T @ np.linalg.solve(
                    jac@jac.T+self.adapter.contract.damping**2*np.eye(6),self.reference_twist)
            if self.mode=='cartesian' and self.use_bounded(q):
                velocity=bounded_dls(jac,self.reference_twist,self.adapter.contract.damping,
                                     -self.velocity_limits,self.velocity_limits)
            result[self.arm]=np.clip(velocity,-self.velocity_limits,self.velocity_limits)
        # Fingers keep zero velocity references; their opening/closing is PD.
        return result

    def physics_step(self, dt):
        # A rendered update can integrate several physics steps. Only this
        # callback, rather than the render loop, owns servo timing.
        try:
            if abs(float(dt)-1.0/180)>1e-8:
                raise RuntimeError(f'unexpected servo physics dt: {dt}')
            self.servo()
            self.physics_dt_samples.append(float(dt))
        except Exception as exc:
            # Kit may log and swallow callback exceptions. Retain the failure
            # and explicitly reject the run before writing its trace.
            self.callback_error=repr(exc)
            raise

    def servo(self):
        from isaacsim.core.utils.types import ArticulationAction
        velocity=self.velocity_target()
        target=self.target()
        q=np.asarray(self.robot.get_joint_positions())[self.arm]
        qd=np.asarray(self.robot.get_joint_velocities())[self.arm]
        p,rot,jac=self.adapter.pose_jacobian(q)
        goal,goal_rot,_=self.adapter.contract.decode(self.action)
        error=np.r_[goal-p,(goal_rot*rot.inv()).as_rotvec()]
        raw=jac.T@np.linalg.solve(jac@jac.T+self.adapter.contract.damping**2*np.eye(6),error)
        delta=np.clip(raw,-self.adapter.contract.max_joint_target_delta_rad,self.adapter.contract.max_joint_target_delta_rad)
        data=dict(command_index=self.command_index,q=q.copy(),qd=qd.copy(),position_target=target[self.arm].copy(),
            velocity_target=velocity[self.arm].copy(),
            solver_projected_effort_nm=np.asarray(self.robot.get_measured_joint_efforts())[self.arm].copy(),
            nominal_pd_effort_nm=self.kp*(target[self.arm]-q)+self.kd*(velocity[self.arm]-qd),
            unclipped_ik_delta_rad=raw,ik_delta_clipped=np.abs(raw)>self.adapter.contract.max_joint_target_delta_rad,
            target_limit_clipped=(q+delta<self.adapter.lower)|(q+delta>self.adapter.upper),
            velocity_clipped=np.abs(velocity[self.arm])>=self.velocity_limits-1e-5,
            jacobian_min_singular_value=np.linalg.svd(jac,compute_uv=False)[-1],
            goal_position_error_m=np.linalg.norm(error[:3]),goal_orientation_error_rad=np.linalg.norm(error[3:]),
            bounded_active=self.use_bounded(q) if self.mode=='cartesian' else False)
        for key,value in data.items():self.diagnostics[key].append(value)
        self.robot.apply_action(ArticulationAction(joint_positions=target, joint_velocities=velocity))
        self.velocity_commands.append(velocity.copy())
        self.servo_calls+=1


def patch_worker(source):
    """Exact anchors deliberately fail after incompatible upstream changes."""
    replacements=[
      ('        pilot_ik_controller = None',
       '        pilot_ik_controller = None\n        capstone_hold_audit = []\n        capstone_joint_candidates = None'),
      ('            rec["ee_pos"].append(ee_p); rec["ee_quat"].append(np.asarray(ee_q, np.float32))',
       '            rec["ee_pos"].append(ee_p); rec["ee_quat"].append(np.asarray(ee_q, np.float32))\n            if capstone_joint_candidates is None:\n                from capstone_vla.task2_bridge import grasp_joint_candidates\n                capstone_joint_candidates = grasp_joint_candidates(stage, robot_path, objects)\n            enabled = [path for path in capstone_joint_candidates\n                       if bool(stage.GetPrimAtPath(path).GetAttribute(\'physics:jointEnabled\').Get())]\n            capstone_hold_audit.append([bool(attach_state.get(\'attached\')), len(enabled),\n                                       bool(attach_state.get(\'held_tool\'))])\n            if args.grasp_mode == \'physical\' and (enabled or attach_state.get(\'attached\')):\n                raise RuntimeError(\'physical replay unexpectedly enabled grasp assistance\')'),
      ('pilot_velocity_target = np.zeros_like(tgt)',
       'pilot_velocity_target = pilot_ik_controller.velocity_target()'),
      ('from tools.pilot_recorded_actuators import restore_recorded_franka_pd',
       'from capstone_vla.task2_bridge import configure_actuators as restore_recorded_franka_pd'),
      ('''                if args.pilot_cartesian_ik:
                    if pilot_ik_controller is None:''', '''                if True:
                    from capstone_vla.task2_bridge import SceneController
                    if pilot_ik_controller is None:
                        pilot_ik_controller = SceneController(robot, arm_dof, os.environ['CAPSTONE_TASK2_MODE'])
                        sim.add_physics_callback('capstone_task2_servo', pilot_ik_controller.physics_step)
                    recorded_velocity = None
                    if pilot_ik_controller.velocity_reference == 'recorded-diagnostic':
                        recorded_velocity = (replay_state_trace or {})['joint_drive_velocity'][k]
                    tgt = pilot_ik_controller.set_source(tgt, recorded_velocity)
                if False:
                    if pilot_ik_controller is None:'''),
      ('''                    _step_control_period(sim, script_runtime, dt_control, physics_dt)
            if recorder is not None:''', '''                    expected_dt = 1.0 / 180
                    if abs(physics_dt - expected_dt) > 1e-9 or abs(dt_control - 1.0/60) > 1e-9:
                        raise RuntimeError('capstone task2 requires 60 Hz command / 180 Hz physics')
                    _step_control_period(sim, script_runtime, dt_control, physics_dt)
                    if pilot_ik_controller.callback_error is not None:
                        raise RuntimeError(pilot_ik_controller.callback_error)
            if recorder is not None:'''),
      ('pilot_trace.attrs["controller_ee_offset_m"] = 0.107',
       '''pilot_trace.attrs["controller_ee_offset_m"] = 0.1034
                    pilot_trace.attrs["capstone_mode"] = os.environ['CAPSTONE_TASK2_MODE']
                    pilot_trace.attrs["mode"] = 'capstone_' + os.environ['CAPSTONE_TASK2_MODE']
                    pilot_trace.attrs["servo_calls"] = pilot_ik_controller.servo_calls
                    pilot_trace.create_dataset("servo_physics_dt_s", data=pilot_ik_controller.physics_dt_samples)
                    pilot_trace.attrs["command_hz"] = 60
                    pilot_trace.attrs["servo_hz"] = 180
                    pilot_trace.attrs["ik_solver"] = pilot_ik_controller.ik_solver
                    pilot_trace.attrs["recovery_thresholds_json"] = json.dumps(vars(pilot_ik_controller.recovery))
                    pilot_trace.attrs["bounded_from_command"] = pilot_ik_controller.bounded_from_command
                    pilot_trace.attrs["velocity_feedforward"] = pilot_ik_controller.velocity_reference
                    pilot_trace.create_dataset("servo_joint_velocity_target", data=pilot_ik_controller.velocity_commands)
                    pilot_trace.attrs["diagnostic_only"] = True
                    pilot_trace.attrs["state_written_each_frame"] = False
                    pilot_trace.create_dataset('grasp_assistance_audit', data=capstone_hold_audit)
                    pilot_trace.attrs['grasp_assistance_columns'] = json.dumps(['attachment_state_active','enabled_direct_robot_object_joints','held_tool_mode'])
                    pilot_trace.attrs['grasp_joint_candidates_json'] = json.dumps(capstone_joint_candidates)
                    pilot_trace.attrs['actual_grasp_mode'] = args.grasp_mode
                    diag = pilot_trace.create_group('servo_diagnostics')
                    diag.attrs['sampling'] = 'pre-step state; projected effort from preceding solve'
                    diag.attrs['nominal_pd_effort'] = 'estimate, not measured motor torque; implicit solver and contact response differ'
                    for name,values in pilot_ik_controller.diagnostics.items():
                        diag.create_dataset(name,data=values)'''),
    ]
    for old,new in replacements:
        if source.count(old)!=1:
            raise RuntimeError('worker bridge anchor mismatch: '+old[:100])
        source=source.replace(old,new,1)
    compile(source,'<capstone-task2-worker>','exec')
    return source
