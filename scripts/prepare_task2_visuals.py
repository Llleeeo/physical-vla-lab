#!/usr/bin/env python3
"""Build rendering-only pose records from the completed physical comparison.

All moving poses and velocities come from measured trace states. These are ineligible
for training, physics replay, or task-success certification.
"""
from pathlib import Path
import copy
import json
import sys

import h5py
import numpy as np

REPO = Path('/home/robophys/Capstone_Project_Yuan_Haoming/data/derived/hf_pilot_render_repo')
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('run',type=Path)
parser.add_argument('--source', type=Path, help='Explicit source episode for another single-object task')
args=parser.parse_args()
ROOT=args.run.resolve()
SOURCE = args.source.resolve() if args.source else Path('/home/robophys/Capstone_Project_Yuan_Haoming/data/derived/hf_pilot_render_60hz_work/Trajectory/1s1p/grasp_sliding_motion_on_flat_surface_1s1p_g9wvkk1/traj_00001')
sys.path.insert(0,str(REPO/'Scripts'))
from dataset.replay_io import read_replay_scene,write_replay_scene,validate_replay_scene_consistency

meta=json.loads((SOURCE/'meta.json').read_text())
with h5py.File(SOURCE/'replay.hdf5','r') as original:
    template=read_replay_scene(original)
assert len(template['rigid_bodies']['paths'])==1
assert not template['visual_xforms']['paths']
assert not template['particle_sets'] and not template.get('mesh_sequences')
root_state=template['robot_root']['state']
assert np.allclose(root_state,root_state[0],atol=1e-7,rtol=0)
manifest={}
for index,mode in enumerate(['cartesian']):
    trace_path=ROOT/f'{mode}_trace.hdf5'
    with h5py.File(trace_path,'r') as trace:
        n=len(trace['timestamp_s'])
        observations={k:trace[k][:] for k in ['joint_pos','joint_vel','object_pos','object_quat','object_vel','object_ang_vel']}
        scene=copy.deepcopy(template)
        scene['rigid_bodies']['state']=np.concatenate([
            observations['object_pos'],observations['object_quat'],observations['object_vel'],
            observations['object_ang_vel']],axis=2).astype(np.float32)
        scene['rigid_bodies']['state_sources']=['collector_observation']
        scene['robot_root']['state']=np.repeat(root_state[:1],n,axis=0)
        scene['robot_root']['source']='usd_fixed_joint_backfill'
        scene['verification']=validate_replay_scene_consistency(scene,observations)
        target=ROOT/'visualization_replays/Trajectory/1s1p'/meta['task']['id']/f'traj_{90000+index:05d}'
        target.mkdir(parents=True,exist_ok=True)
        if (target/'replay.hdf5').exists():
            raise FileExistsError(target/'replay.hdf5')
        with h5py.File(target/'replay.hdf5','x') as out:
            out.attrs.update({'schema_version':3,'control_hz':60.,'num_samples':n,
                              'visualization_only':True,'training_eligible':False,
                              'unrecorded_object_angular_velocity':'measured from PhysX trace; visualization only'})
            out.create_dataset('actions',data=trace['actions'][:])
            out.create_dataset('timestamp',data=trace['timestamp_s'][:])
            for k,v in observations.items():out.create_dataset('obs/'+k,data=v)
            write_replay_scene(out,scene,n)
        m={k:copy.deepcopy(meta[k]) for k in ['task','source_task','robot','joint_names','objects','object_paths','action_space','grasp_mode','control_hz']}
        m.update({'success':None,'num_steps':n,'replay_has_scene':True,
                  'visualization_only':True,'trajectory_outcome':'diagnostic_visualization',
                  'trajectory_diversity':{'dataset_eligible_success':False},
                  'diagnostic_mode':mode,'diagnostic_source_trace':str(trace_path),
                  'unrecorded_channels':{'object_ang_vel':'present in source trace'},
                  'physics_replay_eligible':False,'training_eligible':False})
        (target/'meta.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
        with h5py.File(target/'replay.hdf5','r') as out:
            s=read_replay_scene(out)
            validate_replay_scene_consistency(s,{k:out['obs/'+k][:] for k in observations},require_verified=True)
            assert np.array_equal(out['obs/joint_pos'][:],trace['joint_pos'][:])
            assert np.array_equal(out['obs/object_pos'][:],trace['object_pos'][:])
        manifest[mode]={'render_episode':str(target),'source_trace':str(trace_path),'frames':n}
(ROOT/'visualization_replays/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
