#!/usr/bin/env python3
"""CPU-only perception prototype on saved live RGB. Truth is evaluator-only."""
import argparse,json,sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import h5py,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from capstone_vla.ball_visual_geometry import locate_ball,approach_geometry
from capstone_vla.hf_archive import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();run=args.run_dir.resolve()
    cfg=json.loads((run/'run_config.json').read_text());report=json.loads((run/'report.json').read_text())
    assert report['full_requested_integration_passed']
    for path,digest in report['artifacts'].items():assert sha256(run/path)==digest,path
    assert sha256(cfg['task'])==cfg['task_sha256']
    from pxr import Usd,UsdGeom
    stage=Usd.Stage.Open(cfg['task'],load=Usd.Stage.LoadNone)
    ep=Path(report['probe_episode']);meta=json.loads((ep/'meta.json').read_text())
    rows=[];calibration=[]
    with h5py.File(run/'live_observations.hdf5') as images:
        names=json.loads(images.attrs['camera_pose_order_json'])
        records={v['name']:v for v in json.loads(images.attrs['camera_records_json'])}
        for name in names:
            cam=UsdGeom.Camera(stage.GetPrimAtPath(records[name]['prim_path']))
            assert cam.GetProjectionAttr().Get()=='perspective'
            assert not cam.GetHorizontalApertureOffsetAttr().Get() and not cam.GetVerticalApertureOffsetAttr().Get()
            assert cam.GetPrim().GetAttribute('cameraProjectionType').Get() in (None,'pinhole')
            focal=float(cam.GetFocalLengthAttr().Get());aperture=float(cam.GetHorizontalApertureAttr().Get())
            assert focal>0 and aperture>0
            f=640*focal/aperture
            calibration.append([[f,0,320],[0,f,240],[0,0,1]])
        # Entire estimator pass finishes before evaluator opens object truth.
        for i in range(len(images['state'])):
            row=locate_ball([images[name][i] for name in names],images['camera_pose_world'][i],calibration)
            rows.append(dict(frame=i,**row))
    target=meta['policy']['roles']['A'];target_index=meta['object_paths'].index(target)
    with h5py.File(ep/'replay.hdf5') as truth:
        for row in rows:
            if row['valid']:
                i=row['frame'];position=np.array(row['position_world_m'])
                row['evaluation_error_mm']=float(np.linalg.norm(position-truth['obs/object_pos'][i,target_index])*1000)
                row['estimated_approach_geometry']=approach_geometry(position,truth['obs/finger_segment_world'][i])
    first_contact=(report.get('independent_grasp') or {}).get('measurements',{}).get('first_contact_frame')
    def summary(items):
        errors=[v['evaluation_error_mm'] for v in items if v['valid']]
        return dict(frames=len(items),valid_frames=len(errors),coverage=len(errors)/len(items) if items else None,
            error_median_p95_max_mm=np.percentile(errors,[50,95,100]).tolist() if errors else None,
            missing_frames=[v['frame'] for v in items if not v['valid']])
    result=dict(schema='physical-vla-lab.ball-visual-geometry.v1',source_run=str(run),
        scope='Exposed-case orange-ball RGB prototype; not runtime protection or certified clearance',
        estimator_inputs=['RGB pair','camera intrinsics','measured camera poses'],truth_used_for_estimation=False,
        truth_used_for_evaluation=True,physics_run=False,gpu_used=False,heldout_evaluation=False,
        camera_names=names,intrinsics=calibration,first_contact_frame=first_contact,
        all_frames=summary(rows),before_first_contact=summary(rows[:first_contact]) if first_contact is not None else None,
        rows=rows,limitations=['color-specific detector','no temporal extrapolation','missing/ambiguous views rejected',
        'partial occlusion and bbox centroid bias','5mm ray gap is not a3D error bound',
        'reference points omit finite finger surfaces','runtime guard not implemented'],
        evidence_sha256={str(p):sha256(p) for p in [run/'report.json',run/'run_config.json',run/'live_observations.hdf5',
            ep/'replay.hdf5',Path(cfg['task']),Path(__file__),ROOT/'src/capstone_vla/ball_visual_geometry.py']})
    out=run/'visual_geometry_audits'/datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True);(out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('VISUAL_GEOMETRY_CPU_COMPLETED '+json.dumps(dict(before_contact=result['before_first_contact'],all_valid=result['all_frames']['valid_frames'],all_frames=len(rows))),flush=True)
    print('REPORT '+str(out/'report.json'))

if __name__=='__main__':main()
