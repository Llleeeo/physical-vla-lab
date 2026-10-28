#!/usr/bin/env python3
"""Read-only camera pose and rendered-position audit for this green-object Task2."""
import argparse,json
from pathlib import Path
import h5py,numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'data/derived/task2_multiepisode_20261026/training_runs/20261026_185108_323227/closed_loop_runs/20261026_194730_178870'

def main():
 p=argparse.ArgumentParser();p.add_argument('run_dir',type=Path);a=p.parse_args();base=a.run_dir;out=base/'camera_alignment_audit';out.mkdir(exist_ok=False)
 row=next(r for r in json.loads((ROOT/'data/derived/task2_expansion_inventory_20261025/render_sources.json').read_text())['sources'] if r['episode']=='traj_00042');result={'scope':'pose and green-object projection diagnostic on development Task2 only','camera_poses':{},'object_projection':[]};tiles=[]
 with h5py.File(row['expected_camera_archive']) as source,h5py.File(base/'live_observations.hdf5') as live,h5py.File(base/'physics_trace.hdf5') as physics,h5py.File(OLD/'live_observations.hdf5') as old:
  names=json.loads(source.attrs['camera_names_json']);order=json.loads(live.attrs['camera_pose_order_json']);calibrations=json.loads(source.attrs['camera_output_calibration_json'])
  for name in ['opst_cam','wrist_cam']:
   pose=live['camera_pose_world'][0,order.index(name)];idx=names.index(name);position=source['rgb_camera_position_world'][0,idx];quat=source['rgb_camera_quaternion_world_wxyz'][0,idx]
   error=float(np.linalg.norm(pose[:3]-position)*1000);angle=float(np.rad2deg((Rotation.from_quat(pose[[4,5,6,3]]).inv()*Rotation.from_quat(quat[[1,2,3,0]])).magnitude()))
   if error>.1 or angle>.01:raise RuntimeError('camera extrinsics mismatch '+name)
   result['camera_poses'][name]={'initial_position_error_mm':error,'initial_angle_error_deg':angle,'rgb_mean_absolute_difference':float(np.abs(live[name][0].astype(float)-source[name+'_rgb'][0].astype(float)).mean())}
   tiles.append(np.concatenate([source[name+'_rgb'][0],old[name][0],live[name][0]],axis=1))
  K=np.asarray(calibrations['opst_cam']['stream_calibration']['rgb']['intrinsic_matrix_px'])
  for k in range(min(8,len(live['state']))):
   pose=live['camera_pose_world'][k,order.index('opst_cam')];point=Rotation.from_quat(pose[[4,5,6,3]]).inv().apply(physics['object_pos'][k,0]-pose[:3]);expected=np.array([K[0,2]+K[0,0]*point[0]/(-point[2]),K[1,2]-K[1,1]*point[1]/(-point[2])])
   r,g,b=live['opst_cam'][k].astype(float).transpose(2,0,1);y,x=np.where((g>r*1.5)&(g>b*1.05)&(g>55));center=np.array([np.median(x),np.median(y)]) if len(x) else None
   result['object_projection'].append({'frame':k,'expected_pixel_xy':expected.tolist(),'green_pixel_median_xy':center.tolist() if center is not None else None,'pixel_error':float(np.linalg.norm(center-expected)) if center is not None else None})
  result['first_frame_pixel_check_pass']=result['object_projection'][0]['pixel_error'] is not None and result['object_projection'][0]['pixel_error']<3
 canvas=Image.fromarray(np.concatenate(tiles,axis=0));draw=ImageDraw.Draw(canvas)
 for x,label in [(0,'TRAINING EXPERT'),(640,'OLD LIVE DEFAULT CAMERA'),(1280,'CURRENT LIVE')]:draw.rectangle([x,0,x+500,24],fill='black');draw.text((x+6,5),label,fill='white')
 canvas.save(out/'first_frame_comparison.jpg',quality=90)
 (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2));print('CAMERA_AUDIT_REPORT',out/'report.json')
if __name__=='__main__':main()
