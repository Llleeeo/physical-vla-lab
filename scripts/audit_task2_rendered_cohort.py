#!/usr/bin/env python3
"""Read-only 56-episode camera alignment and sampled RGB audit."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import h5py
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'data/derived/task2_expansion_inventory_20261025'
NAMES=['opst_cam','wrist_cam','side_left_cam','side_right_cam','overhead_cam']

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
 return h.hexdigest()

def main():
 manifest=json.loads((PLAN/'split_manifest.json').read_text())
 render=json.loads((PLAN/'render_sources.json').read_text())
 if render['manifest_sha256']!=digest(PLAN/'split_manifest.json'):raise ValueError('split binding changed')
 if render.get('visuals_ready_count')!=56:raise ValueError('renderer has not marked all56 complete')
 out=PLAN/('visual_qa_'+datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S'))
 out.mkdir(exist_ok=False)
 declared={r['episode']:r for rows in manifest['selected'].values() for r in rows}
 rows=[]
 for row in render['sources']:
  source=Path(row['source_replay']);ep=Path(row['render_episode']);n=row['frames']
  if digest(source)!=declared[row['episode']]['replay_sha256']:raise ValueError('raw replay changed')
  if digest(ep/'replay.hdf5')!=declared[row['episode']]['replay_sha256']:raise ValueError('render replay differs from frozen raw')
  sampled=[]
  with h5py.File(source,'r') as replay,h5py.File(ep/'video/rendered.hdf5','r') as camera:
   if not np.array_equal(camera['source_frame_index'][:],np.arange(n)):raise ValueError('incomplete frame index '+row['episode'])
   if not np.allclose(camera['timestamp_s'][:],replay['timestamp'][:],rtol=0,atol=1e-7):raise ValueError('timestamp mismatch '+row['episode'])
   if float(camera.attrs['fps'])!=60.:raise ValueError('camera clock mismatch')
   actual=json.loads(camera.attrs['camera_names_json'])
   if set(actual)!=set(NAMES):raise ValueError('camera inventory mismatch')
   for name in NAMES:
    data=camera[name+'_rgb']
    if data.shape!=(n,480,640,3) or data.dtype!=np.uint8:raise ValueError('RGB shape/dtype mismatch')
   # Full indexing/schema checked; pixel statistics sample3 frames x2 policy views.
   for name in ['opst_cam','wrist_cam']:
    for i in sorted({0,n//2,n-1}):
     image=camera[name+'_rgb'][i]
     rec={'camera':name,'frame':i,'mean_rgb':float(image.mean()),'std_rgb':float(image.std()),
          'near_black_pixel_fraction':float(np.mean(np.max(image,axis=-1)<3))}
     rec['suspicious_blank_sample']=rec['std_rgb']<1 or rec['near_black_pixel_fraction']>.99
     sampled.append(rec)
  rows.append({'episode':row['episode'],'split':row['split'],'frames':n,
      'aligned_all_frame_indices':True,'aligned_all_timestamps':True,'camera_count':5,'actual_fps':60,
      'archive_bytes':(ep/'video/rendered.hdf5').stat().st_size,'sampled_rgb_statistics':sampled})
 report={'status':'FULL_SCHEMA_ALIGNMENT_PASS','episodes':rows,'episode_count':len(rows),
     'total_frames':sum(r['frames'] for r in rows),'total_archives_bytes':sum(r['archive_bytes'] for r in rows),
     'suspicious_blank_sample_count':sum(s['suspicious_blank_sample'] for r in rows for s in r['sampled_rgb_statistics']),
     'pixel_review_scope':'3 sampled frames per episode in each of2 policy views, not full image-content certification',
     'heldout_model_evaluation':False,'training_ready':False}
 # Training episode27 was flagged geometrically; inspect without modifying it.
 ep=Path(next(r['render_episode'] for r in render['sources'] if r['episode']=='traj_00027'))
 frames=[0,140,145,273,320,356];views=['opst_cam','wrist_cam','side_left_cam']
 sheet=Image.new('RGB',(6*320,3*265+30),'white');draw=ImageDraw.Draw(sheet)
 draw.text((10,8),'traj_00027 / recorded-state rendering / sampled camera QA',fill='black')
 with h5py.File(ep/'video/rendered.hdf5') as camera:
  for j,name in enumerate(views):
   for k,i in enumerate(frames):
    picture=Image.fromarray(camera[name+'_rgb'][i]).resize((320,240))
    sheet.paste(picture,(k*320,30+j*265))
    draw.text((k*320+5,30+j*265+242),name+' frame='+str(i),fill='black')
 sheet.save(out/'traj_00027_camera_contact_sheet.jpg',quality=92)
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print('COHORT_VISUAL_QA_SCHEMA_PASS episodes='+str(len(rows))+' frames='+str(report['total_frames'])+
       ' blank_sample_flags='+str(report['suspicious_blank_sample_count'])+' report='+str(out/'report.json'),flush=True)
 print('CONTACT_SHEET',out/'traj_00027_camera_contact_sheet.jpg',flush=True)

if __name__=='__main__':main()
