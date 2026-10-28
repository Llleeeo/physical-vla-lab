"""Read-only synchronized native PhysX rotation evidence; never writes state."""
import numpy as np


class NativeRotationProbe:
    def __init__(self,robot,target_view,hand_path):
        self.arm=robot._articulation_view._physics_view
        self.body=target_view._physics_view
        if self.arm is None or self.body is None:raise RuntimeError('native physics views unavailable')
        self.hand=list(self.arm.shared_metatype.link_names).index(str(hand_path).rsplit('/',1)[-1])
        self.samples={name:[] for name in ['command_index','physics_dt_s','object_pose_wxyz','hand_pose_wxyz',
                                          'object_velocity_world','hand_velocity_world']}
        self.error=None

    def sample(self,dt,command_index):
        try:
            if not np.isfinite(dt) or abs(float(dt)-1/180)>1e-8:raise ValueError('unexpected native probe physics dt')
            obj=np.asarray(self.body.get_transforms(),dtype=float).reshape(-1,7)[0]
            hand=np.asarray(self.arm.get_link_transforms(),dtype=float)[0,self.hand]
            ov=np.asarray(self.body.get_velocities(),dtype=float).reshape(-1,6)[0]
            hv=np.asarray(self.arm.get_link_velocities(),dtype=float)[0,self.hand]
            # Native tensors: xyz + xyzw. Archive: xyz + wxyz, both world frames.
            op=obj[[0,1,2,6,3,4,5]].copy();hp=hand[[0,1,2,6,3,4,5]].copy()
            if not np.isfinite(np.r_[op,hp,ov,hv]).all():raise ValueError('nonfinite native probe state')
            if not np.allclose([np.linalg.norm(op[3:]),np.linalg.norm(hp[3:])],1.,atol=1e-4):
                raise ValueError('native probe quaternion norms invalid')
            values=dict(command_index=int(command_index),physics_dt_s=float(dt),object_pose_wxyz=op,
                        hand_pose_wxyz=hp,object_velocity_world=ov.copy(),hand_velocity_world=hv.copy())
            for key,value in values.items():self.samples[key].append(value)
        except Exception as exc:
            self.error=repr(exc);raise

    def write_hdf5(self,group):
        if self.error is not None:raise RuntimeError(self.error)
        group.attrs['sampling']='pre-physics-callback snapshots from native PhysX views; initial absolute clock offset omitted'
        group.attrs['pose_layout']='world xyz metres, quaternion wxyz'
        group.attrs['velocity_layout']='world linear m/s, angular rad/s'
        group.attrs['scope']='observer only; no control or object-state writes'
        for key,values in self.samples.items():group.create_dataset(key,data=np.asarray(values))


def instrument_worker_source(source):
    replacements=[
      ("                        sim.add_physics_callback('capstone_task2_servo', pilot_ik_controller.physics_step)",
       """                        from capstone_vla.native_rotation_probe import NativeRotationProbe
                        native_rotation_probe = NativeRotationProbe(robot, views[objects[0]], hand_contact_path)
                        def capstone_native_tick(dt):
                            native_rotation_probe.sample(dt, pilot_ik_controller.command_index)
                            pilot_ik_controller.physics_step(dt)
                        sim.add_physics_callback('capstone_task2_servo', capstone_native_tick)"""),
      ('                    if pilot_ik_controller.callback_error is not None:',
       """                    if native_rotation_probe.error is not None:
                        raise RuntimeError(native_rotation_probe.error)
                    if pilot_ik_controller.callback_error is not None:"""),
      ('                    pilot_trace.attrs["schema"] = "capstone_franka_actuator_replay_probe_v1"',
       """                    native_rotation_probe.write_hdf5(pilot_trace.create_group('native_rotation_probe'))
                    pilot_trace.attrs['runtime_finger_material_json'] = json.dumps(finger_material)
                    pilot_trace.attrs['schema'] = 'capstone_franka_actuator_replay_probe_v1'"""),
    ]
    for old,new in replacements:
        if source.count(old)!=1:raise RuntimeError('native probe anchor mismatch: '+old[:100])
        source=source.replace(old,new,1)
    compile(source,'<native-rotation-worker>','exec')
    return source
