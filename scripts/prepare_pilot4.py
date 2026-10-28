#!/usr/bin/env python3
"""Prepare an isolated pilot-4 scene; reuse the existing repoint implementation."""
from pathlib import Path
import hashlib
import importlib.util
import json
import re
import sys
from pxr import Sdf,Usd,UsdPhysics,UsdUtils
ROOT=Path(__file__).resolve().parents[1]
repo=ROOT/'data/derived/hf_pilot_render_repo'
workspace=ROOT/'data/derived/development_validation_20261025/pilot4_workspace'
source=repo/'Tasks/1s1p/grasp_fixed_axis_rotation_1s1p_akrsd68.usd'
raw=ROOT/'data/raw/hf_pilot_2026_10_24'
record=json.loads((raw/'pilot_manifest.json').read_text())['records'][3]
episode=raw/record['repo']/record['episode'];meta=json.loads((episode/'meta.json').read_text())
expected=meta['provenance']['source_task']['sha256']
if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:raise ValueError('Task source hash differs from recorded source')
if workspace.exists():raise FileExistsError(workspace)
workspace.mkdir(parents=True)
for name in ('Robots','SolidObjects','InteractableObjects','Sensors','Scripts'):
    if (repo/name).exists():(workspace/name).symlink_to(repo/name,target_is_directory=True)
(workspace/'Backgrounds').symlink_to(ROOT/'Backgrounds/Backgrounds_ultra',target_is_directory=True)
task=workspace/'Tasks/1s1p'/source.name;task.parent.mkdir(parents=True)
spec=importlib.util.spec_from_file_location('existing_repoint',repo/'Scripts/fix_task/repoint_tasks.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
original=Sdf.Layer.FindOrOpen(str(source));layer=Sdf.Layer.CreateAnonymous('.usda');layer.TransferContent(original)
before=layer.ExportToString();final=set()
changed=module._repoint_layer(layer,'ultra',src='extreme',final_paths=final)
missing=[path for path in sorted(final) if not (task.parent/path).exists()]
if missing:raise ValueError('missing repointed assets: '+repr(missing[:5]))
after=layer.ExportToString()
if after!=before.replace('Home_ka6zmk7_extreme','Home_ka6zmk7_ultra'):
    raise ValueError('unexpected edits beyond the approved background prefix')
layer.Export(str(task))
stage=Usd.Stage.Open(str(task))
objects=meta['object_paths'];joints=[]
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.Joint):
        joint=UsdPhysics.Joint(prim);bodies=list(map(str,[*joint.GetBody0Rel().GetTargets(),*joint.GetBody1Rel().GetTargets()]))
        if any(any(body==obj or body.startswith(obj+'/') for obj in objects) for body in bodies):
            joints.append({'path':str(prim.GetPath()),'type':prim.GetTypeName(),'bodies':bodies,'enabled':bool(joint.GetJointEnabledAttr().Get())})
prim=stage.GetPrimAtPath(objects[0]);rb=UsdPhysics.RigidBodyAPI(prim)
report={'task_id':meta['task']['id'],'source_task_sha256':expected,'prepared_task':str(task),
        'prepared_task_sha256':hashlib.sha256(task.read_bytes()).hexdigest(),
        'only_background_prefix_changed':True,'background_variant_used':'ultra','changed_asset_paths':changed,
        'missing_background_direct_references':missing,'object_paths':objects,'object_joint_connections':joints,
        'authored_object_angular_velocity_deg_s':list(rb.GetAngularVelocityAttr().Get() or []),
        'episode':str(episode),'source_repository':record['repo'],'source_revision':record['revision'],
        'scope':'development validation; variant substitution is not certified physical equivalence'}
(workspace/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
