"""One exposed ball demonstration, aligned to the existing 29D action contract.

Reads original 60 Hz RGB directly; never substitutes surrogate images or object
state. This small development adapter is not a LeRobot export or evaluation set.
"""
import json
from pathlib import Path
import h5py
import numpy as np
from .ball_action_contract import robot_state, encode_stream
from .cartesian_packets import build_packets
from .cartesian_target_replay import require
from .hf_archive import sha256
from .pilot_dataset import CAMERAS, temporal_indices
from .posture_anchor import compute_anchor

SCHEMA = 'physical-vla-lab.hgwind-development-data.v1'
BATCH = 'data/derived/hgwind_contract_batch/20261029_215343_401088/batch.json'
SUMMARY = 'results/summaries/2026-10-29-bounded-contract-batch-result.json'
RENDER = 'data/derived/hgwind_40_camera_v2_20261028/render_manifest.json'
DESTINATION = 'data/derived/hgwind_traj18_development_20261031'
CORE = ['actions', 'timestamp', 'is_first', 'is_last', 'is_terminal'] + [
    'obs/'+k for k in ('ee_pos','ee_quat','ee_vel','finger_segment_world',
    'joint_pos','joint_vel','object_ang_vel','object_pos','object_quat','object_vel')]


def aligned_indices(timestamps, indices, image_timestamps, fps):
    n = len(timestamps)
    require(n > 1 and np.isclose(fps, 60), 'Need 60 Hz source images')
    require(np.array_equal(indices, np.arange(n)), 'Missing/duplicated/reordered image frames')
    require(np.shape(image_timestamps) == (n,) and
        np.allclose(timestamps, np.arange(n)/60., atol=1.e-7, rtol=0) and
        np.allclose(image_timestamps, timestamps, atol=1.e-7, rtol=0), 'Image/control clock mismatch')


def states_from_robot(q, qdot, ee_pos, ee_quat, anchor, offset):
    n = len(q)
    require(q.shape == qdot.shape == (n,9) and ee_pos.shape == (n,3)
        and ee_quat.shape == (n,4), 'Unexpected measured robot schema')
    require(np.shape(offset) == (3,) and np.isfinite(offset).all(), 'Invalid robot pinch calibration')
    # Same fixed EE->pinch transform as Expert.ee_target_to_pinch; wxyz.
    norm = np.linalg.norm(ee_quat, axis=1, keepdims=True)
    require(np.isfinite(norm).all() and np.all(np.abs(norm-1) < 1.e-5), 'Invalid measured quaternion')
    quat = ee_quat.astype(float)/norm
    vec = quat[:,1:]
    rotated = offset + 2*np.cross(vec, np.cross(vec,offset)+quat[:,:1]*offset)
    return np.stack([robot_state(q[i,:7],qdot[i,:7],ee_pos[i]+rotated[i],
        quat[i],anchor,q[i,7:]) for i in range(n)])


def normalization(states, actions, chunk=20):
    require(states.shape == (len(actions),30) and actions.shape[1:] == (29,)
        and np.isfinite(states).all() and np.isfinite(actions).all(), 'Invalid training arrays')
    indices = np.arange(len(actions))[:,None] + np.arange(chunk)
    valid = indices < len(actions)
    targets = actions[np.minimum(indices,len(actions)-1)][valid]
    result = {}
    for name, values in [('observation.state',states),('action',targets)]:
        values = values.astype(np.float64)
        std = values.std(0); std[std < 1.e-6] = 1.
        result[name] = {k:v.astype(np.float32) for k,v in dict(
            mean=values.mean(0),std=std,min=values.min(0),max=values.max(0)).items()}
    return result


def inspect_source(root):
    root = Path(root).resolve()
    summary_path = root/SUMMARY; summary = json.loads(summary_path.read_text())
    batch_path = root/BATCH; require(sha256(batch_path) == summary['evidence_sha256']['batch.json'], 'Batch evidence changed')
    case = next(c for c in json.loads(batch_path.read_text())['cases'] if c['episode']=='traj_00018')
    teacher = Path(case['teacher']['run']); surrogate = Path(case['contract']['run'])
    paths = [summary_path,batch_path,root/RENDER,teacher/'plan.json',surrogate/'plan.json']
    reports = {}
    for kind, folder in [('teacher',teacher),('surrogate',surrogate)]:
        path = folder/'report.json'
        require(sha256(path)==summary['evidence_sha256']['traj_00018_'+kind+'_report.json'], 'Verified report changed')
        report = json.loads(path.read_text()); reports[kind]=report; paths.append(path)
        require(report['runtime_safety_passed'] is True and report['initial_state_match'] is True
            and report['independent_grasp_evidence']['status']=='pass'
            and report['completed_frames']==269, 'Need complete verified development case')
    require(reports['teacher']['cartesian_target_capture_verified'] is True
        and reports['surrogate']['ball_action_contract_passed'] is True, 'Action interface/capture not verified')
    plan = json.loads((surrogate/'plan.json').read_text())
    source = Path(plan['source']); replay = Path(plan['teacher_replay'])
    require(source.parent == Path(case['source']), 'Wrong source episode')
    manifest = json.loads((root/RENDER).read_text())
    row = next(r for r in manifest['rows'] if r['host']=='local' and r['episode']=='traj_00018')
    require(Path(row['source']) == source.parent and manifest['camera_fps']==60, 'Wrong render source')
    require(manifest['task_sha256']==plan['task_sha256'], 'Render and control task versions differ')
    for file in row['files']:
        p=source.parent/file['path'];require(sha256(p)==file['sha256'], 'Raw source changed: '+p.name)
        paths.append(p)
    staged=Path(row['staged']); image=staged/'video/rendered.hdf5'
    paths += [replay,replay.with_name('meta.json'),staged/'replay.hdf5',image,
        Path(plan['action_vectors']),Path(plan['cartesian_target_capture'])]
    for key in ['teacher_replay','cartesian_target_capture']:
        require(sha256(plan[key])==plan[key+'_sha256'], 'Captured teacher source changed')
    meta = json.loads(replay.with_name('meta.json').read_text())
    instruction = meta['task'].get('instruction_cn') or meta['task']['instruction']
    require(isinstance(instruction,str) and bool(instruction.strip()), 'Missing instruction')
    calibration=meta['validation']['pinch_frame']
    require(calibration['valid'] is True, 'Unverified EE->pinch calibration')
    offset=np.asarray(calibration['offset_local_m'],float)
    with h5py.File(source,'r') as original, h5py.File(replay,'r') as demo, h5py.File(staged/'replay.hdf5','r') as rendered:
        for key in CORE:
            require(np.array_equal(original[key][:],demo[key][:]) and
                np.array_equal(original[key][:],rendered[key][:]), 'Teacher/render/source states differ: '+key)
        q=demo['obs/joint_pos'][:]; qdot=demo['obs/joint_vel'][:]
        ts=demo['timestamp'][:]; anchor=compute_anchor(q[0,:7],plan['robot_posture_configuration'])
        capture=json.loads(Path(plan['cartesian_target_capture']).read_text())
        actions=encode_stream(build_packets(capture,demo['actions'][:,7:],'fixed-9-8','latched',anchor))
        require(np.array_equal(actions,np.load(plan['action_vectors'],allow_pickle=False)), 'Labels differ from verified float32 interface')
        states=states_from_robot(q,qdot,demo['obs/ee_pos'][:],demo['obs/ee_quat'][:],anchor,offset)
    with h5py.File(image,'r') as f:
        aligned_indices(ts,f['source_frame_index'][:],f['timestamp_s'][:],float(f.attrs['fps']))
        for name in CAMERAS:
            ds=f[name+'_rgb']
            require(ds.shape == (269,480,640,3) and ds.dtype==np.uint8, 'Unexpected RGB schema')
            for i in range(269):
                require(ds[i].shape==(480,640,3), 'RGB decode failure')
        camera_digest=str(f.attrs['camera_config_digest'])
    provenance = dict(schema=SCHEMA, episode='local/traj_00018',purpose='development_overfit_only',
        frames=269,fps=60,action_dim=29,state_dim=30,delta_action=False,
        history=[-2,0],chunk_size=20,instruction=instruction,images=str(image),
        camera_config_digest=camera_digest,cameras=list(CAMERAS),
        image_action_pairing='pre-action RGB[k], measured robot state[k] -> upper command[k]',
        pinch_offset_local_m=offset.tolist(),robot_posture_configuration=plan['robot_posture_configuration'],
        source_core_arrays_exact_to_teacher_and_render=True,all_RGB_frames_decoded=True,
        action_vectors_exact_to_passing_float32_surrogate=True,
        policy_inputs='RGB + measured robot state/latched robot posture + instruction',
        privileged_object_observations=False,source_images_are_surrogate_rollout=False,
        heldout_evaluation=False,production_training_eligible=False,lerobot_export=False,
        source_hashes={str(p):sha256(p) for p in dict.fromkeys(paths)})
    return states,actions,ts,provenance


class BallDevelopmentDataset:
    """Numpy dataset with pinned model-facing keys; absolute 29D actions only."""
    def __init__(self, directory, verify=True):
        self.directory=Path(directory)
        self.meta=json.loads((self.directory/'dataset.json').read_text())
        require(self.meta['schema']==SCHEMA and self.meta['purpose']=='development_overfit_only'
            and self.meta['delta_action'] is False, 'Wrong development payload')
        payload=self.directory/'arrays.npz'
        require(sha256(payload)==self.meta['payload_sha256'], 'Development arrays changed')
        if verify:
            for name,digest in self.meta['source_hashes'].items():
                require(sha256(name)==digest, 'Source changed: '+Path(name).name)
        with np.load(payload,allow_pickle=False) as a:
            self.states=a['states'].copy();self.actions=a['actions'].copy()
        require(self.states.shape==(269,30) and self.actions.shape==(269,29), 'Unexpected payload dimensions')
        normalization(self.states,self.actions)

    def __len__(self):return len(self.actions)

    def __getitem__(self,frame):
        obs,opad,act,apad=temporal_indices(frame,len(self))
        result={'observation.state':self.states[obs].copy(),'observation.state_is_pad':opad,
            'action':self.actions[act].copy(),'action_is_pad':apad,'task':self.meta['instruction']}
        with h5py.File(self.meta['images'],'r') as f:
            for name in CAMERAS:
                key='observation.images.'+name
                images=np.stack([f[name+'_rgb'][int(i)] for i in obs])
                result[key]=np.ascontiguousarray(images.transpose(0,3,1,2),dtype=np.float32)/255.
                result[key+'_is_pad']=opad.copy()
        return result
