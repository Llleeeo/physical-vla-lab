import sys,json,copy,collections,argparse,datetime
from pathlib import Path
import h5py,numpy as np
root=Path(__file__).resolve().parents[1];sys.path[:0]=[str(root/'src'),str(root/'scripts')]
from capstone_vla.ball_near_capture import SETTINGS,locate_for_plan,NearCaptureGuard
from capstone_vla.ball_visual_geometry import locate_ball
from capstone_vla.ball_approach_follow import make_approach_guard
from capstone_vla.ball_live_protocol import decoded_request
from capstone_vla.hf_archive import sha256
parser=argparse.ArgumentParser(description='CPU frozen-observation recovery/depth audit; no simulation')
parser.add_argument('--source-run',type=Path,required=True)
args=parser.parse_args();out=args.source_run.resolve()
cfg=json.loads((out/'run_config.json').read_text());report=json.loads((out/'report.json').read_text())
assert all(sha256(out/k)==v for k,v in report['artifacts'].items())
old=make_approach_guard(cfg['approach_guard']);plans=[]
for depth in [0,.005]:
 p=copy.deepcopy(cfg['approach_guard']);p['near_capture']=dict(settings=dict(SETTINGS),extra_depth_m=depth);plans.append(p)
gs=[NearCaptureGuard(p) for p in plans];continuity=NearCaptureGuard(plans[0]);events=[[],[]];continuity_actions=[];actions=[[],[]];errors=[];fallback=[]
ep=Path(report['probe_episode']);meta=json.loads((ep/'meta.json').read_text())
with h5py.File(out/'live_observations.hdf5') as f,h5py.File(ep/'replay.hdf5') as h:
 names=json.loads(f.attrs['camera_pose_order_json']);optics={c['name']:c['intrinsics'] for c in plans[0]['cameras']};ks=[optics[n] for n in names]
 # Object truth is isolated to scoring the completed estimator, never given to it.
 truth=h['obs/object_pos'][:,meta['object_paths'].index(meta['policy']['roles']['A'])]
 for i in range(len(f['state'])):
  imgs=[f[n][i] for n in names];poses=f['camera_pose_world'][i];d=locate_ball(imgs,poses,ks);nd=locate_for_plan(imgs,poses,ks,names,plans[0]);raw=f['model_action'][i]
  with np.load(out/'ipc'/f'response_{i:06d}.npz',allow_pickle=False) as response:np.testing.assert_array_equal(raw,response['absolute'][0])
  args=(raw,i,f['state'][i,14:21],f['robot_finger_points'][i],f['robot_finger_corners'][i])
  previous,_=old.apply(*args,d);np.testing.assert_allclose(previous,f['action'][i],rtol=0,atol=1e-12)
  cont,_=continuity.apply(*args,d);continuity_actions.append(cont)
  for j,g in enumerate(gs):
   a,e=g.apply(*args,nd);assert a[28]==raw[28];decoded_request(a,i,f['state'][i,21:28]);actions[j].append(a);events[j].append(e)
  if nd.get('method'):
   errors.append(float(np.linalg.norm(np.array(nd['position_world_m'])-truth[i])*1000));fallback.append(i)
metrics=dict(status='CPU_NEAR_CAPTURE_PREVIEW_PASS',source_frames=len(events[0]),source_incomplete=True,fallback_frames=fallback,
 fallback_position_error_mm=dict(mean=float(np.mean(errors)),maximum=float(max(errors))),
 original_commands_reconstructed=True,raw_predictions_bound=True,jaw_unchanged=True,native_decode_rejections=0,
 continuity_only_recovered_goal_step_mm={str(i):float(np.linalg.norm(continuity_actions[i][:3]-continuity_actions[i-1][:3])*1000) for i in [162,167]},
 candidates=[dict(extra_depth_m=p['near_capture']['extra_depth_m'],active_frames=sum(e['follow']['active'] for e in ev),hold_frames=sum(e['near_capture']['holding'] for e in ev),modes=dict(collections.Counter(e['mode'] for e in ev))) for p,ev in zip(plans,events)],
 physics_launched=False,grasp_result=None,scope='Frozen-observation preview; truth used only for offline localization scoring, not control',
 source_hashes={**{k:sha256(out/k) for k in ['report.json','run_config.json','live_observations.hdf5','approach_guard_events.json']},**{'probe/'+k:sha256(ep/k) for k in ['replay.hdf5','meta.json']}})
destination=out/'near_capture_previews'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
destination.mkdir(parents=True)
metrics['implementation_hashes']={name:sha256(root/name) for name in ['src/capstone_vla/ball_near_capture.py','scripts/check_hgwind_near_capture.py']}
(destination/'report.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps(metrics,indent=2));print('REPORT '+str(destination/'report.json'))
