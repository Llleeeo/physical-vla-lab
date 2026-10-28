#!/usr/bin/env python3
"""Export all synchronized Task2 frames for a development-only reader check."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from capstone_vla.franka import FrankaCartesianAdapter
from capstone_vla.pilot_dataset import CAMERAS, PilotH5Dataset, action_chunk, temporal_indices
TASK='grasp_sliding_motion_on_flat_surface_1s1p_g9wvkk1'
DEFAULT=ROOT/'data/derived/hf_pilot_render_60hz_work/Trajectory/1s1p'/TASK/'traj_00001'
UPSTREAM='434dc7105cf6d559018d15af5fecbcdad44f036e'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',type=Path,default=DEFAULT)
    parser.add_argument('--output',type=Path,default=ROOT/'data/derived/task2_dynamicvla_development_20261025')
    args=parser.parse_args()
    episode=args.episode.resolve();out=args.output.resolve()
    if out.exists():raise FileExistsError('Refusing to overwrite: '+str(out))
    meta=json.loads((episode/'meta.json').read_text())
    cfg_path=ROOT/'configs/franka_action_interface.json'
    cfg=json.loads(cfg_path.read_text());adapter=FrankaCartesianAdapter(cfg['urdf'])
    if meta['task']['id']!=TASK or meta['robot']['type']!='franka' or meta['action_space']!='joint_position_target':
        raise ValueError('this exporter is deliberately Task2-only')
    if meta['joint_names']!=[f'panda_joint{i}' for i in range(1,8)]+['panda_finger_joint1','panda_finger_joint2']:
        raise ValueError('unexpected joint order')
    instruction=meta['task'].get('instruction_cn') or meta['task']['instruction']
    source_paths=[episode/'meta.json',episode/'replay.hdf5',episode/'video/rendered.hdf5']
    source_stats={str(p):{'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in source_paths}
    with h5py.File(source_paths[1],'r') as replay,h5py.File(source_paths[2],'r') as camera:
        q=replay['obs/joint_pos'][:];targets=replay['actions'][:];ts=replay['timestamp'][:]
        n=len(ts);indices=camera['source_frame_index'][:];camera_ts=camera['timestamp_s'][:]
        if n!=232 or q.shape!=(n,9) or targets.shape!=(n,9):raise ValueError('incomplete episode')
        if not np.isfinite(q).all() or not np.isfinite(targets).all():raise ValueError('nonfinite robot data')
        if not np.array_equal(indices,np.arange(n)):raise ValueError('need exactly one image per action; no silent resampling')
        if not np.allclose(ts,np.arange(n)/60.,rtol=0,atol=1e-7):raise ValueError('control clock mismatch')
        if not np.allclose(camera_ts,ts,rtol=0,atol=1e-7):raise ValueError('image timestamps mismatch')
        if not np.isclose(float(camera.attrs['fps']),60):raise ValueError('camera FPS mismatch')
        if not np.allclose(targets[:,7],targets[:,8],atol=1e-6):raise ValueError('asymmetric gripper')
        if not np.all(np.isclose(targets[:,7],0,atol=1e-5)|np.isclose(targets[:,7],.04,atol=1e-5)):
            raise ValueError('unsupported gripper command')
        state=np.stack([adapter.state(x[:7]) for x in q]).astype(np.float32)
        action=np.stack([np.r_[adapter.state(x[:7]),1. if x[7]>.02 else -1.] for x in targets]).astype(np.float32)
        # Record the source, rather than labelling source images as new-controller images.
        out.mkdir(parents=True)
        payload=out/'episode_000000.hdf5'
        with h5py.File(payload,'x') as f:
            f.attrs.update(schema='capstone_dynamicvla_development_payload_v1',fps=60.,
                instruction=instruction,task_id=TASK,upstream_commit=UPSTREAM,
                split='development_overfit_only',action_representation='absolute_xyz_m_euler_xyz_rad_binary_gripper',
                action_frame='robot_base:panda_link0/panda_hand_plus_local_z_0.1034m',
                training_dataset_ready=False,source_grasp_assistance_declared=bool(meta.get('attach_grasp',{}).get('attached')))
            f.create_dataset('observation.state',data=state)
            f.create_dataset('action',data=action)
            f.create_dataset('timestamp',data=ts)
            f.create_dataset('source_control_frame_index',data=indices)
            for name in CAMERAS:
                src=camera[name+'_rgb']
                if src.shape!=(n,480,640,3) or src.dtype!=np.uint8:raise ValueError('RGB layout mismatch')
                dst=f.create_dataset('observation.images/'+name,shape=src.shape,dtype=src.dtype,
                                     chunks=(1,480,640,3),compression='lzf')
                for i in range(n):dst[i]=src[i]
        with h5py.File(payload,'r') as f:
            for name in CAMERAS:
                for i in range(n):
                    if not np.array_equal(f['observation.images/'+name][i],camera[name+'_rgb'][i]):
                        raise ValueError('RGB readback mismatch')
            if not np.array_equal(f['action'][:,6],np.where(targets[:,7]>.02,1.,-1.)):
                raise ValueError('gripper event was lost')
    dataset=PilotH5Dataset(payload)
    samples={}
    expected_keys={'observation.state','observation.state_is_pad','action','action_is_pad','task'}
    expected_keys.update('observation.images.'+name+suffix for name in CAMERAS for suffix in ['', '_is_pad'])
    for i in [0,1,116,n-1]:
        item=dataset[i]
        if set(item)!=expected_keys:raise ValueError('unexpected model-facing feature')
        samples[str(i)]={k:list(v.shape) for k,v in item.items() if hasattr(v,'shape')}
    max_delta=0.
    valid_action_targets=0
    for i in range(n):
        delta,pad=action_chunk(action,state,i)
        obs,_,a_idx,_=temporal_indices(i,n)
        if np.any(obs>i) or np.any(a_idx[~pad]<i):raise ValueError('causal indexing violation')
        reconstructed=delta[:,:6]+state[i]
        np.testing.assert_allclose(reconstructed,action[a_idx,:6],rtol=0,atol=5e-7)
        max_delta=max(max_delta,float(np.abs(delta[~pad,3:6]).max()))
        valid_action_targets+=int((~pad).sum())
    if source_stats!={str(p):{'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in source_paths}:
        raise RuntimeError('source files changed during export')
    grasp_report=ROOT/'data/derived/task2_interface_checks/20261025_162000/independent_grasp_evidence.json'
    report={'schema':'capstone_task2_dynamicvla_development_report_v1','status':'CPU_PAYLOAD_AND_NUMPY_READER_PASS',
      'payload':str(payload),'source_episode':str(episode),'source_files_stat_unchanged':True,
      'source_metadata_sha256':sha(source_paths[0]),'controller_config_sha256':sha(cfg_path),
      'urdf_sha256':adapter.urdf_sha256,'upstream_commit':UPSTREAM,'frames':n,'fps':60,
      'image_action_pairing':'RGB[k], measured robot state[k] -> command[k] (pre-action observation)',
      'observation_history_frames':[-2,0],'observation_history_seconds':[-2/60,0],
      'action_chunk_frames':20,'action_chunk_duration_s':20/60,
      'action_indices':'k..k+19, clamp at episode end and mask padded values',
      'delta_action':'subtract current observed state from all 20 absolute xyz/Euler targets; gripper unchanged',
      'max_valid_chunk_euler_component_delta_rad':max_delta,'valid_chunk_action_targets':valid_action_targets,
      'gripper_transition_frames':(np.flatnonzero(np.diff(action[:,6])!=0)+1).tolist(),
      'all_source_gripper_commands_preserved':True,'all_exported_RGB_frames_verified_exact':True,
      'policy_input_has_privileged_object_state':False,'reader_samples':samples,
      'source_attach_grasp':meta.get('attach_grasp'),
      'separate_physical_interface_verification':str(grasp_report),
      'source_images_are_new_controller_rollout':False,'development_episode_indices':[0],
      'held_out_evaluation':False,'lerobot_exported':False,'upstream_loader_executed':False,
      'model_forward_executed':False,'training_started':False,
      'remaining':['isolated PyTorch/LeRobot 0.3.3 training environment','LeRobot v2.1 export and actual upstream loader smoke',
                   'explicit single-episode split and verbatim instruction adapter','pretrained checkpoint and model forward/backward smoke']}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (out/'split.json').write_text(json.dumps({'purpose':'development_overfit_only','development':[0],
        'validation':[],'test':[],'never_report_as_heldout_evaluation':True},indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','frames','fps','gripper_transition_frames','max_valid_chunk_euler_component_delta_rad','payload','upstream_loader_executed']},indent=2))

if __name__=='__main__':main()
