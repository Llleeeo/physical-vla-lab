#!/usr/bin/env python3
"""Local Franka action-interface checks; no collection/task-service imports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--smoke', action='store_true')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
cfg = json.loads(args.config.read_text())
if args.output.exists() and any(p.name != 'run.log' for p in args.output.iterdir()):
    parser.error('output directory must be new or empty')
args.output.mkdir(parents=True, exist_ok=True)
app = AppLauncher(args).app

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from capstone_vla.franka import FrankaActionContract, FrankaCartesianAdapter


def array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def quat_rotation(quat_wxyz):
    quat = np.asarray(quat_wxyz)
    return Rotation.from_quat(quat[[1,2,3,0]])


def main():
    timing = cfg['timing']
    hz = int(timing['physics_hz'])
    if hz != timing['servo_hz'] or hz % int(timing['command_hz']):
        raise ValueError('servo must equal physics rate; command rate must divide it')
    command_every = hz//int(timing['command_hz'])
    dt = 1/hz
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device))
    sim.set_camera_view(eye=[1.6,1.6,1.2], target=[.35,0,.45])
    floor = sim_utils.CuboidCfg(size=(2.,2.,.05), collision_props=sim_utils.CollisionPropertiesCfg(),
                               visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.35,.35,.35)))
    floor.func('/World/Floor', floor, translation=(0,0,-.05))
    light = sim_utils.DomeLightCfg(intensity=2500., color=(.8,.8,.8))
    light.func('/World/Light', light)
    arm_names = [f'panda_joint{i}' for i in range(1,8)]
    motor = cfg['actuators']
    actuators = {
        f'arm_{i}': ImplicitActuatorCfg(
            joint_names_expr=[name], effort_limit_sim=motor['effort_limit_nm'][i],
            velocity_limit_sim=motor['velocity_limit_rad_s'][i],
            stiffness=motor['stiffness'][i], damping=motor['damping'][i],
        ) for i,name in enumerate(arm_names)
    }
    actuators['fingers'] = ImplicitActuatorCfg(
        joint_names_expr=['panda_finger_joint.*'], effort_limit_sim=motor['finger_effort_limit_n'],
        stiffness=motor['finger_stiffness'], damping=motor['finger_damping'],
    )
    robot_cfg = ArticulationCfg(
        prim_path='/World/Robot',
        spawn=sim_utils.UsdFileCfg(
            usd_path=cfg['robot_usd'],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=motor['disable_gravity']),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={
            **dict(zip(arm_names,cfg['initial_arm_joints_rad'])), 'panda_finger_joint.*':.04,
        }),
        actuators=actuators,
    )
    robot = Articulation(robot_cfg)
    # All changes are in this newly created stage, never in the source USD.
    import omni.usd
    from pxr import Usd
    stage = omni.usd.get_context().get_stage()
    removed_scripts = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Robot')):
        attr = prim.GetAttribute('omni:scripting:scripts')
        if attr and attr.Get():
            removed_scripts.append(str(prim.GetPath()));attr.Set([])
    sim.reset()
    robot.update(dt)
    joint_ids,names = robot.find_joints(arm_names, preserve_order=True)
    finger_ids,finger_names = robot.find_joints(['panda_finger_joint1','panda_finger_joint2'], preserve_order=True)
    body_ids,_ = robot.find_bodies('panda_hand')
    hand_id = body_ids[0]
    if names != arm_names or len(finger_ids)!=2 or not robot.is_fixed_base:
        raise ValueError('unexpected articulation joint/body/base contract')
    contract_keys = ['ee_offset_m','damping','max_joint_target_delta_rad','finger_open_m']
    adapter = FrankaCartesianAdapter(cfg['urdf'],FrankaActionContract(**{k:cfg['action'][k] for k in contract_keys}))
    caps = cfg['acceptance']
    view = robot.root_physx_view
    readback = {
        'stiffness':array(view.get_dof_stiffnesses())[0,joint_ids],
        'damping':array(view.get_dof_dampings())[0,joint_ids],
        'effort_limit_nm':array(view.get_dof_max_forces())[0,joint_ids],
        'velocity_limit_rad_s':array(view.get_dof_max_velocities())[0,joint_ids],
    }
    for key,values in readback.items():
        if not np.allclose(values,motor[key],atol=1e-4,rtol=1e-6):
            raise RuntimeError(f'PhysX actuator {key} readback mismatch: {values}')
    gravity_readback=array(view.get_disable_gravities()).astype(bool)
    if not np.all(gravity_readback==bool(motor['disable_gravity'])):
        raise RuntimeError('PhysX robot gravity mode readback mismatch')
    trace = {key:[] for key in ['case','time_s','action','joint_pos','joint_target',
                                'measured_ee_position_base','measured_ee_quaternion_xyzw',
                                'position_error_m','rotation_error_rad','finger_error_m',
                                'fk_live_position_error_m','fk_live_rotation_error_rad']}
    def measured():
        root = array(robot.data.root_pose_w)[0]
        hand = array(robot.data.body_pose_w)[0,hand_id]
        root_r,hand_r = quat_rotation(root[3:]),quat_rotation(hand[3:])
        point_w = hand[:3]+hand_r.apply([0,0,adapter.contract.ee_offset_m])
        return root_r.inv().apply(point_w-root[:3]),root_r.inv()*hand_r

    case_names = ['move_x'] if args.smoke else ['move_x','move_y','move_z','rotate_z','close','open','circle']
    results=[]
    for name in case_names:
        q0=robot.data.default_joint_pos.clone();v0=torch.zeros_like(q0)
        q0[:,finger_ids]=0. if name=='open' else .04
        robot.write_joint_state_to_sim(q0,v0);robot.reset()
        robot.set_joint_position_target(q0);robot.set_joint_velocity_target(v0)
        for _ in range(hz//4):
            robot.write_data_to_sim();sim.step(render=not args.headless);robot.update(dt)
        home_p,home_r = measured()
        duration = 2. if args.smoke or name in ('close','open') else (6. if name=='circle' else 3.)
        rows=[]
        for tick in range(round(duration*hz)):
            if not app.is_running():raise RuntimeError('simulation closed before completion')
            time_s=tick*dt
            if tick % command_every==0:
                u=min(time_s/(.5 if args.smoke else 1.),1.)
                smooth=10*u**3-15*u**4+6*u**5
                p,r,grip=home_p.copy(),home_r,1.
                if name.startswith('move_'):
                    axis={'move_x':0,'move_y':1,'move_z':2}[name]
                    p[axis]+=(.01 if args.smoke else .03)*smooth
                elif name=='rotate_z':r=Rotation.from_rotvec([0,0,np.deg2rad(10)*smooth])*home_r
                elif name=='close':grip=-1.
                elif name=='circle':
                    angle=2*np.pi*time_s/duration
                    p+=np.array([.01*np.sin(angle),.01*(1-np.cos(angle)),0])
                action=adapter.contract.encode(p,r,grip)
            q=array(robot.data.joint_pos)[0,joint_ids]
            target,finger=adapter.joint_target(q,action)
            full=robot.data.joint_pos.clone()
            full[:,joint_ids]=torch.as_tensor(target,device=robot.device)
            full[:,finger_ids]=finger
            robot.set_joint_position_target(full)
            robot.set_joint_velocity_target(torch.zeros_like(full))
            robot.write_data_to_sim();sim.step(render=(not args.headless and tick%command_every==0));robot.update(dt)
            actual_p,actual_r=measured()
            current=array(robot.data.joint_pos)[0]
            fk_p,fk_r,_=adapter.pose_jacobian(current[joint_ids])
            model_p=float(np.linalg.norm(fk_p-actual_p));model_r=float((fk_r*actual_r.inv()).magnitude())
            if model_p>caps['fk_live_position_error_m'] or model_r>caps['fk_live_rotation_error_rad']:
                raise RuntimeError(f'URDF/live end-effector mismatch: {model_p} m, {model_r} rad')
            goal_p,goal_r,_=adapter.contract.decode(action)
            row = {
                'case':name,'time_s':(tick+1)*dt,'action':action.copy(),
                'joint_pos':current.copy(),'joint_target':array(full)[0].copy(),
                'measured_ee_position_base':actual_p,'measured_ee_quaternion_xyzw':actual_r.as_quat(),
                'position_error_m':float(np.linalg.norm(actual_p-goal_p)),
                'rotation_error_rad':float((goal_r*actual_r.inv()).magnitude()),
                'finger_error_m':float(np.max(np.abs(current[finger_ids]-finger))),
                'fk_live_position_error_m':model_p,'fk_live_rotation_error_rad':model_r,
            }
            for key,value in row.items():trace[key].append(value)
            rows.append(row)
        final=rows[-max(1,hz//5):]
        pos=float(np.mean([x['position_error_m'] for x in final]))
        rot=float(np.mean([x['rotation_error_rad'] for x in final]))
        fingers=float(np.mean([x['finger_error_m'] for x in final]))
        p95=float(np.percentile([x['position_error_m'] for x in rows],95))
        passed=(pos<caps['static_position_error_m'] and rot<caps['static_rotation_error_rad']
                and fingers<caps['finger_position_error_m'])
        if name=='circle':passed=p95<caps['circle_position_p95_m'] and rot<caps['static_rotation_error_rad']
        case={'name':name,'passed':bool(passed),'duration_s':duration,
              'final_position_error_m':pos,'final_rotation_error_rad':rot,
              'final_finger_error_m':fingers,'path_position_p95_m':p95,
              'measured_displacement_base_m':(measured()[0]-home_p).tolist()}
        results.append(case);print('INTERFACE_CASE '+json.dumps(case),flush=True)
    np.savez_compressed(args.output/'trace.npz',**{key:np.asarray(value) for key,value in trace.items()})
    report={'schema':'capstone_franka_interface_check_v1','passed':all(x['passed'] for x in results),
            'smoke_only':args.smoke,'config':cfg,
            'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),
            'robot_usd_sha256':hashlib.sha256(Path(cfg['robot_usd']).read_bytes()).hexdigest(),
            'urdf_sha256':adapter.urdf_sha256,'arm_joint_names':names,'finger_joint_names':finger_names,
            'disabled_robot_scripts':removed_scripts,'actuator_readback':{k:v.tolist() for k,v in readback.items()},
            'robot_gravity_disabled_readback':bool(np.all(gravity_readback)),
            'max_fk_live_position_error_m':float(max(trace['fk_live_position_error_m'])),
            'max_fk_live_rotation_error_rad':float(max(trace['fk_live_rotation_error_rad'])),
            'cases':results,'benchmark_task_success_evaluated':False}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('INTERFACE_RESULT '+str(args.output/'report.json')+' passed='+str(report['passed']),flush=True)
    return 0 if report['passed'] else 2


if __name__=='__main__':
    status=1
    try:
        status=main()
    except Exception as exc:
        import traceback
        (args.output/'error.json').write_text(json.dumps({'error_type':type(exc).__name__,'message':str(exc)},indent=2)+'\n')
        traceback.print_exc()
    finally:
        if args.headless:
            # Isaac 4.5 may block while shutting down an unused Replicator
            # scheduler. All report/NPZ handles have closed before this point.
            # Preserve the real pass/fail status and let the OS release Kit.
            import os
            sys.stdout.flush();sys.stderr.flush()
            os._exit(status)
        app.close(wait_for_replicator=False)
    raise SystemExit(status)
