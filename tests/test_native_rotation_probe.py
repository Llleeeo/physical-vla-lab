import unittest
from types import SimpleNamespace
import numpy as np
from capstone_vla.native_rotation_probe import NativeRotationProbe, instrument_worker_source

class NativeRotationProbeTests(unittest.TestCase):
    def fixture(self):
        p=np.array([[1,2,3,0,0,np.sqrt(.5),np.sqrt(.5)]])
        v=np.array([[.1,.2,.3,1,2,3]])
        arm=SimpleNamespace(shared_metatype=SimpleNamespace(link_names=['panda_hand']),
                            get_link_transforms=lambda:p[:,None,:],get_link_velocities=lambda:v[:,None,:])
        body=SimpleNamespace(get_transforms=lambda:p,get_velocities=lambda:v)
        robot=SimpleNamespace(_articulation_view=SimpleNamespace(_physics_view=arm))
        target=SimpleNamespace(_physics_view=body)
        return NativeRotationProbe(robot,target,'/Robot/panda_hand'),p,v
    def test_native_units_layout_and_copy(self):
        probe,p,v=self.fixture();probe.sample(1/180,12)
        np.testing.assert_allclose(probe.samples['object_pose_wxyz'][0],[1,2,3,np.sqrt(.5),0,0,np.sqrt(.5)])
        np.testing.assert_allclose(probe.samples['object_velocity_world'][0],[.1,.2,.3,1,2,3])
        p[:]=0;v[:]=0
        self.assertEqual(probe.samples['object_pose_wxyz'][0][0],1)
        self.assertEqual(probe.samples['hand_velocity_world'][0][-1],3)
    def test_observer_patch_compiles_and_rejects_missing_anchors(self):
        source='def worker():\n' + ''.join('    '*i+'if True:\n' for i in range(1,6))
        source+="                        sim.add_physics_callback('capstone_task2_servo', pilot_ik_controller.physics_step)\n"
        source+='                    if pilot_ik_controller.callback_error is not None:\n                        pass\n'
        source+='                    pilot_trace.attrs["schema"] = "capstone_franka_actuator_replay_probe_v1"\n'
        compile(instrument_worker_source(source),'<test>','exec')
        with self.assertRaises(RuntimeError):instrument_worker_source('pass')

    def test_invalid_timestep_retains_error(self):
        probe,_,_=self.fixture()
        with self.assertRaises(ValueError):probe.sample(1/60,0)
        self.assertIsNotNone(probe.error)

if __name__=='__main__':unittest.main()
