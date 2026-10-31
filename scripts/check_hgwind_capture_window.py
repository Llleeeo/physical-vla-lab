import sys,json,copy,datetime,argparse
from pathlib import Path
import h5py,numpy as np
root=Path(__file__).resolve().parents[1];sys.path[:0]=[str(root/'src'),str(root/'scripts')]
from capstone_vla.ball_capture_window import SETTINGS
from capstone_vla.ball_approach_follow import make_approach_guard
from capstone_vla.ball_near_capture import locate_for_plan
from capstone_vla.ball_live_protocol import decoded_request
from capstone_vla.hf_archive import sha256
p=argparse.ArgumentParser();p.add_argument('--source-run',type=Path,required=True);args=p.parse_args();out=args.source_run.resolve()
cfg=json.loads((out/'run_config.json').read_text());report=json.loads((out/'report.json').read_text())
assert 'analysis_error' not in report
assert all(sha256(out/k)==v for k,v in report['artifacts'].items())
old=make_approach_guard(cfg['approach_guard']);plan=copy.deepcopy(cfg['approach_guard']);plan['near_capture']['extra_depth_m']=.005;plan['capture_window']=dict(SETTINGS);new=make_approach_guard(plan)
events=[];actions=[];checkframes={}
with h5py.File(out/'live_observations.hdf5') as f:
 names=json.loads(f.attrs['camera_pose_order_json']);ks={c['name']:c['intrinsics'] for c in plan['cameras']}
 for i in range(len(f['state'])):
  raw=f['model_action'][i]
  with np.load(out/'ipc'/f'response_{i:06d}.npz',allow_pickle=False) as response:np.testing.assert_array_equal(raw,response['absolute'][0])
  detection=locate_for_plan([f[n][i] for n in names],f['camera_pose_world'][i],[ks[n] for n in names],names,plan)
  values=(raw,i,f['state'][i,14:21],f['robot_finger_points'][i],f['robot_finger_corners'][i],detection)
  original,_=old.apply(*values);np.testing.assert_allclose(original,f['action'][i],rtol=0,atol=1e-10)
  action,event=new.apply(*values);assert action[28]==raw[28];decoded_request(action,i,f['state'][i,21:28])
  events.append(event);actions.append(action)
  if i in [132,135,140,144,145,150,160,170,174,175]:checkframes[str(i)]=dict(original_goal_z_m=float(original[2]),candidate_goal_z_m=float(action[2]),measured_z_m=float(f['state'][i,16]),window=event['capture_window'])
active=[i for i,e in enumerate(events) if e['capture_window']['active']]
result=dict(status='CPU_CAPTURE_WINDOW_PREVIEW_PASS',frames=len(events),plan=plan,active_frames=active,selected_frames=checkframes,jaw_scores_unchanged=True,native_decode_rejections=0,original_commands_reconstructed=True,physics_launched=False,grasp_result=None,scope='Frozen observed inputs only; no future-motion or grasp claim; no object truth used',source_hashes={k:sha256(out/k) for k in ['report.json','run_config.json','live_observations.hdf5','approach_guard_events.json']},implementation_hashes={k:sha256(root/k) for k in ['src/capstone_vla/ball_capture_window.py','src/capstone_vla/ball_near_capture.py','scripts/check_hgwind_capture_window.py']})
destination=out/'capture_window_previews'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');destination.mkdir(parents=True);(destination/'report.json').write_text(json.dumps(result,indent=2)+'\n');np.savez(destination/'actions.npz',actions=np.array(actions));(destination/'decisions.json').write_text(json.dumps(events,indent=2)+'\n')
print('CPU_CAPTURE_WINDOW_PREVIEW_PASS active='+str(len(active)));print(json.dumps(checkframes,indent=2));print('REPORT '+str(destination/'report.json'))
