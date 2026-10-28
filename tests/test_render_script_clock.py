import ast,unittest
from pathlib import Path
from types import SimpleNamespace
from capstone_vla.live_policy_protocol import freeze_render_script_dispatch

class RenderClockTests(unittest.TestCase):
 def setUp(self):
  path=Path('/home/robophys/RoboPhys/isaacsim/extscache/omni.kit.scripting-106.5.2+106.5.0/omni/kit/scripting/scripts/script_manager.py')
  tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ScriptManager')
  method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_on_stage_update')
  scope={'carb':SimpleNamespace(profiler=SimpleNamespace(begin=lambda *a:None,end=lambda *a:None)),'traceback_format_exception':lambda:None}
  exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),scope)
  self.calls=[];instance=SimpleNamespace(_has_on_update_error=False,on_update=lambda t,dt:self.calls.append((t,dt)))
  self.manager=SimpleNamespace(_stage=True,_timeline_playing=True,_current_time=3.,_prim_to_scripts={'object':{'script':instance}})
  self.update=lambda:scope['_on_stage_update'](self.manager,0.,1/60)
 def test_real_script_manager_dispatch_is_frozen_only_for_render(self):
  with freeze_render_script_dispatch(self.manager):self.update();self.update()
  self.assertEqual(self.manager._current_time,3.);self.assertEqual(self.calls,[]);self.assertTrue(self.manager._timeline_playing)
  self.update();self.assertAlmostEqual(self.manager._current_time,3+1/60);self.assertEqual(len(self.calls),1)
 def test_original_play_state_restored_after_render_error(self):
  with self.assertRaisesRegex(ValueError,'render failure'):
   with freeze_render_script_dispatch(self.manager):raise ValueError('render failure')
  self.assertTrue(self.manager._timeline_playing);self.assertEqual(self.manager._current_time,3.)
 def test_unexpected_clock_change_is_rejected_without_resetting_it(self):
  with self.assertRaisesRegex(RuntimeError,'advanced BehaviorScript clock'):
   with freeze_render_script_dispatch(self.manager):self.manager._current_time+=1
  self.assertEqual(self.manager._current_time,4.);self.assertTrue(self.manager._timeline_playing)
