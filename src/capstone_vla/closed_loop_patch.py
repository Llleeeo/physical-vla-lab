"""Exact in-memory patch around the already validated isolated worker bridge."""
from .task2_bridge import patch_worker

def patch_policy_worker(source):
 source=patch_worker(source)
 replacements=[
 ('        pilot_ik_controller = None\n        capstone_hold_audit',"        n_control = min(n_control, int(os.environ['CAPSTONE_POLICY_FRAMES']))\n        pilot_ik_controller = None\n        capstone_hold_audit"),
 ('tgt = np.array(replay_actions[k], dtype=np.float32, copy=True)','tgt = np.asarray(robot.get_joint_positions(), dtype=np.float32).copy()'),
 ('from capstone_vla.task2_bridge import SceneController','from capstone_vla.closed_loop import PolicySceneController as SceneController, publish_gui_physics_state\n                    if os.environ.get("CAPSTONE_POLICY_COMMANDS"):\n                        from capstone_vla.fixed_policy_replay import RecordedPolicySceneController as SceneController'),
 ("SceneController(robot, arm_dof, os.environ['CAPSTONE_TASK2_MODE'])","SceneController(robot, arm_dof, os.environ['CAPSTONE_TASK2_MODE'], stage, sim, publish_gui_physics_state)"),
 ('tgt = pilot_ik_controller.set_source(tgt, recorded_velocity)','tgt = pilot_ik_controller.set_policy_frame(k)'),
 ('                    pilot_trace.create_dataset("expert_actions", data=np.asarray(replay_actions[:len(rec["actions"])], dtype=np.float32))',"                    pilot_trace.attrs['expert_actions_used_for_control'] = False\n                    pilot_trace.attrs['schema'] = ('capstone_fixed_policy_command_replay_v1' if os.environ.get('CAPSTONE_POLICY_COMMANDS') else 'capstone_vla_closed_loop_v1')"),
 ('                    "online_cartesian_dls" if args.pilot_cartesian_ik else "original_joint_targets",','                    ("fixed_policy_commands" if os.environ.get("CAPSTONE_POLICY_COMMANDS") else "vla_closed_loop_cartesian"),' ),
 ('                print("PILOT COMPARISON trace=%s frames=%d mode=%s" % (','                pilot_ik_controller.finish()\n                print("PILOT COMPARISON trace=%s frames=%d mode=%s" % ('),
 ]
 for old,new in replacements:
  if source.count(old)!=1:raise RuntimeError('policy worker anchor mismatch: '+old[:100])
  source=source.replace(old,new,1)
 compile(source,'<capstone-live-policy-worker>','exec')
 return source
