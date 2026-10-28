"""Small local-file protocol with explicit observation/action schemas."""
import os,time
from pathlib import Path
import numpy as np
CAMERAS=('wrist_cam','opst_cam')

def atomic_npz(path,**values):
 path=Path(path);temporary=path.with_suffix('.tmp')
 with temporary.open('wb') as f:np.savez_compressed(f,**values)
 temporary.replace(path)

def validate_observation(states,images):
 states=np.asarray(states)
 if states.shape!=(2,6) or not np.isfinite(states).all():raise ValueError('expected two finite measured 6D states')
 if set(images)!=set(CAMERAS):raise ValueError('policy needs exactly wrist/opst RGB')
 for name,value in images.items():
  if value.shape!=(2,480,640,3) or value.dtype!=np.uint8:raise ValueError('expected two640x480 RGB uint8 images: '+name)

def history_pair(history):
 if not history:raise ValueError('empty measured history')
 return history[max(0,len(history)-3)],history[-1]

def binary_action(value):
 action=np.asarray(value,dtype=np.float64).copy()
 if action.shape!=(7,) or not np.isfinite(action).all():raise ValueError('expected finite absolute 7D policy action')
 action[6]=1. if action[6]>0 else -1.
 return action

def wait_file(path,root,timeout=180):
 deadline=time.monotonic()+timeout
 while not Path(path).exists():
  if (Path(root)/'server_error.txt').exists():raise RuntimeError((Path(root)/'server_error.txt').read_text())
  if time.monotonic()>deadline:raise TimeoutError(str(path))
  time.sleep(.02)


from contextlib import contextmanager

@contextmanager
def freeze_render_script_dispatch(manager):
 """Suspend script on_update only for sensor rendering; never reset its clock."""
 if manager is None or not hasattr(manager,'_timeline_playing') or not hasattr(manager,'_current_time'):
  raise RuntimeError('unrecognized ScriptManager; cannot isolate render-only updates')
 playing=manager._timeline_playing;clock=float(manager._current_time)
 if not playing or not np.isfinite(clock):raise RuntimeError('ScriptManager not ready for live render capture')
 manager._timeline_playing=False
 try:
  yield
 finally:
  manager._timeline_playing=playing
  if abs(float(manager._current_time)-clock)>1e-9:
   raise RuntimeError('sensor rendering advanced BehaviorScript clock')
