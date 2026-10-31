"""Live sensor/robot observations and learned requests through native packets.

The world object is passed only to the existing target-blind geometry adapter.
Default execution uses no recorded actions. An explicit diagnostic can replace
one target-height channel; raw model actions and hybrid provenance are retained.
"""
import json
import time
from pathlib import Path
import h5py
import numpy as np
from .ball_live_protocol import ActionQueue, validate_observation, decoded_request
from .cartesian_packets import PacketStreamExecutor
from .cartesian_target_replay import require
from .hgwind_development_data import states_from_robot
from .live_policy_protocol import atomic_npz, wait_file, history_pair
from .pilot_dataset import CAMERAS
from .posture_anchor import initialize_anchor
from .ball_vertical_diagnostic import apply_height, validate_plan
from .ball_approach_guard import ApproachGuard, robot_corners
from .ball_visual_geometry import locate_ball


class LiveExecutor(PacketStreamExecutor):
    def __init__(self, config):
        super().__init__([], 'fixed-9-8', 'latched', config['robot_posture_configuration'])
        self.config=config; self.root=Path(config['output']); self.ipc=self.root/'ipc'
        self.queue=ActionQueue(config['prefix'],config['frames']); self.history=[]
        self.camera=None; self.archive=None; self.timing=[]; self.closed=False
        self.vertical=config.get("vertical_diagnostic")
        self.approach_plan=config.get('approach_guard')
        require(not (self.vertical and self.approach_plan),'Cannot combine approach and recorded-height interventions')
        self.approach=ApproachGuard(self.approach_plan) if self.approach_plan is not None else None
        self.approach_events=[]
        if self.vertical is not None:validate_plan(self.vertical,config["frames"])

    def _open_camera(self, stage, sim, robot):
        from .closed_loop import LiveRGB, publish_gui_physics_state
        from camera.config import config_digest
        require(config_digest()==self.config['camera_config_digest'], 'Live/training camera profile mismatch')
        self.camera=LiveRGB(stage,sim,robot,publish_gui_physics_state)
        protocol=self.config.get('camera_protocol','legacy')
        require(protocol in ('legacy','formal-temporal'), 'Unknown camera protocol')
        if protocol=='formal-temporal':
            import carb
            from .ball_camera_protocol import formal_capture,validated_clamps
            calibration=self.config['camera_render_calibration']
            validated_clamps(calibration)
            self.camera.render_boundary=lambda rep,quality: formal_capture(rep,quality,carb.settings.get_settings(),calibration)
        self.archive=h5py.File(self.root/'live_observations.hdf5','x')
        self.archive.attrs.update(schema='physical-vla-lab.ball-live-observations.v1',
            timing='pre-action RGB/state; synchronous pause during inference',complete=False,
            camera_records_json=json.dumps(self.camera.records),camera_pose_order_json=json.dumps(self.camera_order),
            camera_protocol=protocol,camera_render_calibration_json=json.dumps(self.config.get('camera_render_calibration')))
        for name in CAMERAS:
            self.archive.create_dataset(name,shape=(0,480,640,3),maxshape=(None,480,640,3),
                dtype='u1',compression='gzip',compression_opts=1,chunks=(1,480,640,3))
        for name,shape in [('state',(30,)),('action',(29,)),('camera_pose_world',(2,7))]:
            self.archive.create_dataset(name,shape=(0,*shape),maxshape=(None,*shape),dtype='f8')
        self.archive.create_dataset('timestamp_s',shape=(0,),maxshape=(None,),dtype='f8')
        if self.approach is not None:
            self.archive.create_dataset('robot_finger_points',shape=(0,2,3),maxshape=(None,2,3),dtype='f8')
            self.archive.create_dataset('robot_finger_corners',shape=(0,2,8,3),maxshape=(None,2,8,3),dtype='f8')
            self.archive.attrs['approach_guard_json']=json.dumps(self.approach_plan)
        if self.vertical is not None or self.approach is not None:
            self.archive.create_dataset('model_action',shape=(0,29),maxshape=(None,29),dtype='f8')
            self.archive.attrs['vertical_diagnostic_json']=json.dumps(self.vertical)

    @property
    def camera_order(self):
        # LiveRGB uses the older sensor helper's order; policy keys are named.
        from .live_policy_protocol import CAMERAS as sensor_order
        return sensor_order

    def live_step(self, policy, world, index, dt, stage, sim, robot, q, qdot, ee_pos, ee_quat):
        require(index==self.queue.next_frame and abs(dt-1/60)<1e-12, 'Live control clock changed')
        if self.posture_initialization is None:
            require(index==0, 'Posture must initialize before policy frame zero')
            self.posture_initialization=initialize_anchor(policy.expert,self.posture_configuration)
            np.testing.assert_array_equal(self.posture_initialization['initial_joint_position_rad'],np.asarray(q[:7],float))
            np.testing.assert_allclose(policy.expert._pinch_offset_local,self.config['pinch_offset_local_m'],rtol=0,atol=1e-8)
        anchor=self.posture_initialization['anchor_rad']
        state=states_from_robot(np.asarray(q)[None],np.asarray(qdot)[None],np.asarray(ee_pos)[None],
            np.asarray(ee_quat)[None],anchor,np.asarray(self.config['pinch_offset_local_m']))[0]
        before=float(sim.current_time)
        if self.camera is None:self._open_camera(stage,sim,robot)
        images=self.camera.capture()
        require(abs(float(sim.current_time)-before)<1e-9,'Sensor setup/capture advanced physics')
        np.testing.assert_allclose(robot.get_joint_positions(),q,rtol=0,atol=1e-7)
        np.testing.assert_allclose(robot.get_joint_velocities(),qdot,rtol=0,atol=1e-7)
        self.history.append(dict(state=state,images=images));self.history=self.history[-3:]
        waiting=inference=0.
        if not self.queue.values:
            past,current=history_pair(self.history)
            states=np.stack([past['state'],current['state']])
            rgb={name:np.stack([past['images'][name],current['images'][name]]) for name in CAMERAS}
            validate_observation(states,rgb)
            start=time.monotonic()
            atomic_npz(self.ipc/f'request_{index:06d}.npz',frame=np.asarray(index,dtype=np.int64),states=states,**rgb)
            response=self.ipc/f'response_{index:06d}.npz';wait_file(response,self.ipc,timeout=180)
            waiting=time.monotonic()-start
            with np.load(response,allow_pickle=False) as f:
                require(set(f.files)=={'frame','absolute','inference_s'},'Unexpected response fields')
                require(int(f['frame'])==index,'Response belongs to another frame')
                self.queue.accept(index,f['absolute']);inference=float(f['inference_s'])
            require(abs(float(sim.current_time)-before)<1e-9,'Physics advanced during policy inference')
            print(f'BALL_LIVE_POLICY frame={index} prefix={self.queue.prefix} wait_s={waiting:.3f}',flush=True)
        action,origin=self.queue.pop(index)
        model_action=action.copy()
        if self.vertical is not None:action=apply_height(action,index,self.vertical,self.config["frames"])
        poses=np.asarray(self.camera.last_poses,float)
        require(poses.shape==(2,7) and np.isfinite(poses).all(),'Invalid live camera transforms')
        guard_values={}
        if self.approach is not None:
            optics={c['name']:c for c in self.approach_plan['cameras']}
            require(set(optics)==set(self.camera_order),'Approach camera mismatch')
            detection=locate_ball([images[n] for n in self.camera_order],poses,
                [optics[n]['intrinsics'] for n in self.camera_order])
            points=np.asarray(policy.expert.finger_segment_world(),float)
            corners=robot_corners(stage,self.approach_plan)
            action,event=self.approach.apply(action,index,state[14:21],points,corners,detection)
            self.approach_events.append(dict(event,detection=detection))
            guard_values=dict(robot_finger_points=points,robot_finger_corners=corners)
        values=dict(state=state,action=action,camera_pose_world=poses,timestamp_s=index/60,**images,**guard_values)
        if self.vertical is not None or self.approach is not None:values["model_action"]=model_action
        for name,value in values.items():
            ds=self.archive[name];ds.resize(index+1,axis=0);ds[index]=value
        self.archive.flush()
        self.timing.append(dict(frame=index,anchor_frame=origin,simulation_time_s=before,
            wait_s=waiting,inference_s=inference,motor_boundary_completed=False))
        try:
            packet=decoded_request(action,index,anchor)
            self.packets.append(packet)
            decision=super().step(policy,world,index,dt)
            self.timing[-1]['motor_boundary_completed']=True
            return decision
        except Exception as exc:
            self.timing[-1]['error']=type(exc).__name__+': '+str(exc)
            raise
        finally:self.save_progress()

    def payload(self):
        payload=super().payload()
        payload.update(learned_policy_evaluated=True,source_motion_and_gripper_stream_retained=False,
            source_numeric_controller_settings_retained=False,source_actions_used_for_control=False,
            synchronous_inference=True,real_time_evaluated=False)
        if self.approach is not None:
            payload.update(approach_guard=self.approach_plan,policy_postprocessing=True,
                unmodified_model_actions=False,visual_guidance=True,simulator_target_state_used=False,
                standalone_policy_evaluation=False)
        if self.vertical is not None:
            payload.update(learned_policy_evaluated=False,model_inference_used=True,
                source_actions_used_for_control=True,vertical_diagnostic=self.vertical,
                standalone_policy_evaluation=False)
        return payload

    def save_progress(self):
        if self.approach is not None:
            path=self.root/'approach_guard_events.json';tmp=path.with_suffix('.tmp')
            tmp.write_text(json.dumps(self.approach_events,indent=2,allow_nan=False)+'\n');tmp.replace(path)
        for name,value in [('policy_timing.json',self.timing),('packet_requests.json',self.packets),
                           ('cartesian_packet_execution.json',self.payload())]:
            path=self.root/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

    def finish(self):
        if self.closed:return
        self.save_progress()
        if self.archive is not None:
            self.archive.attrs['complete']=len(self.events)==self.config['frames'] and all(e['completed'] for e in self.events)
            self.archive.close()
        self.closed=True
