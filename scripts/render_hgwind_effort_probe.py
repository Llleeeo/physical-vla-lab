#!/usr/bin/env python3
"""Visualize saved HGwind probe states in a separate work copy; --run renders."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from capstone_vla.hgwind_cohort import TASK_ID,require
from capstone_vla.hf_archive import sha256


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-run',type=Path,required=True)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args();run=args.probe_run.resolve();run.relative_to(ROOT/'data/derived/hgwind_effort_probes')
    plan=json.loads((run/'plan.json').read_text());report=json.loads((run/'report.json').read_text())
    episode=Path(report['probe_episode']);episode.relative_to(run/'worker_output')
    meta=json.loads((episode/'meta.json').read_text());require(meta['task']['id']==TASK_ID,'Unexpected task')
    task=Path(plan['task']);require(sha256(task)==plan['task_sha256'],'Recorded task scene changed')
    manifest=json.loads((ROOT/'data/derived/hgwind_40_camera_v2_20261028/render_manifest.json').read_text())
    require(manifest['task_sha256']==plan['task_sha256'],'Render camera scene differs from probe')
    lab=Path(json.loads(Path(plan['controller_config']).read_text())['isaaclab_root'])
    require((lab/'VERSION').read_text().strip()=='2.2.1' and ((lab/'_isaac_sim').resolve()/'VERSION').read_text().startswith('4.5.0'),'SDK mismatch')
    bundle=Path(plan['bundle_manifest']).parent
    require(sha256(plan['bundle_manifest'])==plan['bundle_manifest_sha256'],'Frozen render bundle changed')
    staged_root=run/'visualization/Trajectory';staged=staged_root/'1s1p'/TASK_ID/episode.name
    protected={path:sha256(path) for path in [task,*[episode/name for name in ['meta.json','replay.hdf5','trajectory.hdf5'] if (episode/name).is_file()]]}
    staged.mkdir(parents=True,exist_ok=True)
    for source in protected:
        if source==task:continue
        destination=staged/source.name
        if destination.exists():require(sha256(destination)==protected[source],'Staged source differs: '+source.name)
        else:shutil.copy2(source,destination)
    os.environ.update(CAPSTONE_ISAACLAB_PATH=str(lab),ISAACLAB_PATH=str(lab),
        ISAAC_SIM_ROOT=str((lab/'_isaac_sim').resolve()),OMNI_KIT_ACCEPT_EULA='YES',
        ROBOPHYS_CAPSTONE_OFFLINE_TASK='1',ROBOPHYS_TASK_CONTEXT=Path(manifest['task_context']).read_text(),PYTHONDONTWRITEBYTECODE='1',TERM='xterm')
    adapter=load(bundle/'scripts/sim/dispatch_hgwind_collection.py','probe_render_environment')
    adapter.install_environment_adapter();repo=adapter.REPO
    renderer=load(repo/'Scripts/tools/render_saved_trajectories.py','probe_saved_state_renderer')
    renderer.SINGLE_GPU_KIT_ARGS+=' --portable-root '+str(ROOT/'data/derived/.cache/isaacsim/linux')
    def scene(query):
        require(query in (TASK_ID,TASK_ID.rsplit('_',1)[-1]),'Unexpected render task')
        return SimpleNamespace(task_id=TASK_ID,scene_family='1s1p',usd_path=task,relative_usd=str(task))
    renderer.resolve_task_scene=scene
    dependencies=renderer.resolve_task_dependencies
    renderer.resolve_task_dependencies=lambda _repo,tasks:dependencies(ROOT,tasks)
    sys.argv=[str(repo/'Scripts/tools/render_saved_trajectories.py'),'--launcher',str(lab/'isaaclab.sh'),
        '--out-root',str(staged_root),'--task-id',TASK_ID,'--trajectory-name',episode.name,
        '--only-missing','--device','cuda:0','--render-quality','balanced','--camera-fps','30']
    if not args.run:sys.argv.append('--dry-run')
    print('SAVED_STATE_VISUALIZATION source='+str(episode)+' frames='+str(meta['num_steps'])+' camera_fps=30 new_control_rollout=false training=false',flush=True)
    print('VIDEO_DIRECTORY '+str(staged/'video'),flush=True)
    try:return renderer.main()
    finally:
        require(all(sha256(path)==digest for path,digest in protected.items()),'Original probe evidence changed')
        print('ORIGINAL_PROBE_EVIDENCE_UNCHANGED',flush=True)


if __name__=='__main__':raise SystemExit(main())
