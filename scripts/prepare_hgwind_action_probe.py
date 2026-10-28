#!/usr/bin/env python3
"""Audit recorded-tool FK for the ball cohort and prepare one CPU action probe."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from capstone_vla.franka import FrankaCartesianAdapter
from capstone_vla.recorded_franka import RecordedFrankaAdapter
from capstone_vla.hf_archive import sha256
from capstone_vla.hgwind_cohort import require, verify_files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'data/derived/hgwind_40_camera_v2_20261028/render_manifest.json')
    parser.add_argument('--out-root',type=Path,default=ROOT/'data/derived/hgwind_action_probe_20261028')
    parser.add_argument('--host',choices=('local','hongtao'),default='local')
    parser.add_argument('--episode',default='traj_00000')
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text());config=json.loads((ROOT/'configs/franka_action_interface.json').read_text())
    require(sha256(manifest['task_usd'])==manifest['task_sha256'],'Task USD changed')
    legacy=FrankaCartesianAdapter(config['urdf'])
    from pxr import Usd,UsdGeom
    stage=Usd.Stage.Open(manifest['task_usd']);cache=UsdGeom.XformCache()
    stats=[];selected=None
    for row in manifest['rows']:
        ep=Path(row['source']);verify_files(ep,row['files']);meta=json.loads((ep/'meta.json').read_text())
        require(meta['robot']['type']=='franka' and meta['control_hz']==60,'Unsupported embodiment/clock')
        require(meta['arm_execution']['mode']=='operational_space_inverse_dynamics_effort','Unexpected execution mode')
        require(meta['joint_names']==[f'panda_joint{i}' for i in range(1,8)]+['panda_finger_joint1','panda_finger_joint2'],'Joint order mismatch')
        require(meta['end_effector']['link_path'].startswith(meta['robot']['prim_path']+'/'),'Tool belongs to another robot')
        adapter=RecordedFrankaAdapter(config['urdf'],meta['end_effector'])
        prim=stage.GetPrimAtPath(meta['robot']['prim_path']+'/panda_link0');require(bool(prim),'Base link missing')
        matrix=cache.GetLocalToWorldTransform(prim);basis=np.asarray(matrix)[:3,:3].T
        require(np.allclose(basis.T@basis,np.eye(3),atol=1e-6) and np.linalg.det(basis)>0,'Nonrigid robot base transform')
        base_r=Rotation.from_matrix(basis);base_p=np.asarray(matrix.ExtractTranslation())
        with h5py.File(ep/'replay.hdf5','r') as f:
            time=f['timestamp'][:];q=f['obs/joint_pos'][:];targets=f['actions'][:];truth_p=f['obs/ee_pos'][:];truth_r=Rotation.from_quat(f['obs/ee_quat'][:][:,[1,2,3,0]])
        require(len(q)==row['frames'] and np.isfinite(q).all() and np.isfinite(targets).all(),'Invalid episode')
        require(np.allclose(np.diff(time),1/60,rtol=0,atol=1e-7),'Unexpected command timing')
        corrected=[adapter.pose_jacobian(x[:7])[:2] for x in q];old=[legacy.pose_jacobian(x[:7])[:2] for x in q]
        def errors(poses):
            position=np.stack([base_p+base_r.apply(p) for p,r in poses])
            orientation=Rotation.concatenate([base_r*r for p,r in poses])
            return np.linalg.norm(position-truth_p,axis=1),(orientation*truth_r.inv()).magnitude()
        pe,re=errors(corrected);old_pe,old_re=errors(old)
        require(pe.max()<config['acceptance']['fk_live_position_error_m'] and re.max()<config['acceptance']['fk_live_rotation_error_rad'],'Recorded-frame FK did not match physical observations')
        stats.append({'host':row['host'],'episode':row['episode'],'frames':len(q),'corrected_position_max_mm':float(pe.max()*1000),'corrected_rotation_max_deg':float(np.rad2deg(re.max())),
                      'legacy_frame_position_median_mm':float(np.median(old_pe)*1000),'legacy_frame_rotation_median_deg':float(np.rad2deg(np.median(old_re))),
                      'source_sha256':{x['path']:x['sha256'] for x in row['files']}})
        if row['host']==args.host and row['episode']==args.episode:
            grip=targets[:,7:]
            require(np.allclose(grip[:,0],grip[:,1],atol=1e-6) and np.all(np.isclose(grip,0,atol=1e-5)|np.isclose(grip,.04,atol=1e-5)),'Unsupported gripper encoding')
            state=np.stack([adapter.contract.encode(p,r,1)[:6] for p,r in corrected])
            action=np.stack([adapter.contract.encode(*adapter.pose_jacobian(x[:7])[:2],1 if x[7]>.02 else -1) for x in targets])
            selected=(row,meta,time,state,action,targets)
        verify_files(ep,row['files']);print('RECORDED_TOOL_FK_OK '+row['host']+'/'+row['episode'],flush=True)
    require(selected is not None,'Selected episode missing')
    out=args.out_root/datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True,exist_ok=False)
    row,meta,time,state,action,targets=selected
    with h5py.File(out/'action_probe.hdf5','x') as f:
        f.attrs.update(schema='physical-vla-lab.hgwind-action-probe.v1',fps=60.,training_ready=False,physics_replay_verified=False,
                       action_frame='robot_base:panda_link0 -> recorded right_gripper',representation='absolute_xyz_m_extrinsic_euler_xyz_rad_binary_gripper',
                       source_execution_mode=meta['arm_execution']['mode'],host=row['host'],episode=row['episode'],tool_transform_json=json.dumps(meta['end_effector']),
                       causality='source pre-physics measured state[k] with current command[k]; no resampling',
                       policy_input_has_privileged_object_state=False)
        for key,data in [('observation.state',state),('action.absolute',action),('source.joint_targets',targets),('timestamp',time)]:f.create_dataset(key,data=data)
    report={'schema':'physical-vla-lab.hgwind-fk-audit.v1','status':'CPU_RECORDED_TOOL_FK_PASS','episodes':len(stats),'frames':sum(x['frames'] for x in stats),
            'maximum_corrected_position_error_mm':max(x['corrected_position_max_mm'] for x in stats),'maximum_corrected_rotation_error_deg':max(x['corrected_rotation_max_deg'] for x in stats),
            'urdf_sha256':adapter.urdf_sha256,'manifest_sha256':sha256(args.manifest),'rows':stats,'selected_probe':{'host':row['host'],'episode':row['episode'],'frames':len(time),'gripper_transition_frames':(np.flatnonzero(np.diff(action[:,6])!=0)+1).tolist()},
            'source_mode':'operational_space_inverse_dynamics_effort','source_scene_unchanged':sha256(manifest['task_usd'])==manifest['task_sha256'],
            'simulation_started':False,'physics_replay_verified':False,'training_started':False,'camera_inputs_frozen':False,
            'limits':['Legacy and recorded tools are different valid coordinate conventions; their difference is not itself a controller failure.','The previous position-drive replay runner is incompatible with this source mode; its actuator settings must not be substituted.','No VLA-input images, LeRobot export, simulation grasp result or camera selection is certified by this CPU probe.']}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('HGWIND_ACTION_PROBE '+str(out));print(json.dumps({k:report[k] for k in ['episodes','frames','maximum_corrected_position_error_mm','maximum_corrected_rotation_error_deg']}))


if __name__=='__main__':main()
