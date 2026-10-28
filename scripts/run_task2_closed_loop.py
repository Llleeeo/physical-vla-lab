#!/usr/bin/env python3
"""Single development validation episode; CPU dry-run unless --run is explicit."""
import argparse,json,os,sys,subprocess,signal,shlex,hashlib,time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np,h5py
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from capstone_vla.closed_loop_patch import patch_policy_worker
from capstone_vla.live_policy_protocol import wait_file
from capstone_vla.component_controls import validate_component_bundle,annotate_component_worker
BASE=ROOT/'data/derived/task2_multiepisode_20261026/training_runs/20261026_185108_323227'


def select_development_episode(manifest, record, role):
 """Keep a fixed validation default and allow one explicit TRAIN debugging case."""
 if role == 'validation':
  episode_id = manifest['selected']['validation'][0]['episode']
  if episode_id != 'traj_00042':raise ValueError('first frozen development validation changed')
  expected, forbidden = 'validation_episodes', 'train_episodes'
 elif role == 'train-debug':
  episode_id = 'traj_00001'
  expected, forbidden = 'train_episodes', 'validation_episodes'
 else:raise ValueError('unsupported episode role')
 if episode_id not in record[expected] or episode_id in record[forbidden]:
  raise ValueError('episode split does not match requested role')
 return episode_id


def fixed_command_files(parent, prior):
 origin=prior.get('command_origin','saved_live_policy')
 if origin=='saved_live_policy':return parent/'live_observations.hdf5',parent/'physics_trace.hdf5',origin
 if origin=='recorded_expert_observation_predictions':
  if prior.get('schema')!='capstone_teacher_command_bundle_v1' or prior.get('training_eligible') is not False:
   raise ValueError('invalid teacher diagnostic bundle')
  return parent/'commands.hdf5',parent/'commands.hdf5',origin
 if origin=='expert_model_component_diagnostic':
  validate_component_bundle(prior)
  return parent/'commands.hdf5',parent/'commands.hdf5',origin
 raise ValueError('unknown fixed-command origin')


def select_capacity_checkpoint(run, role, default_checkpoint, manifest_sha256, source_sha256):
 """Allow isolated TRAIN001 fit weights only for explicit TRAIN debugging."""
 if role!='train-debug':raise ValueError('capacity-fit weights are restricted to TRAIN debugging')
 run=run.resolve();checkpoint=run/'checkpoint_final'
 report=json.loads((run/'report.json').read_text());fit=json.loads((run/'run_config.json').read_text())
 if report.get('status')!='SINGLE_EPISODE_CAPACITY_COMPLETED' or fit.get('train_episodes')!=['traj_00001'] or fit.get('validation_opened') is not False or fit.get('reserved_test_opened') is not False:raise ValueError('invalid capacity diagnostic scope')
 if fit.get('frozen_manifest_sha256')!=manifest_sha256 or fit.get('source_replay_sha256')!=source_sha256:raise ValueError('capacity source/split mismatch')
 if Path(fit['initial_checkpoint']).resolve()!=default_checkpoint.resolve():raise ValueError('unexpected initial checkpoint')
 if hashlib.sha256((default_checkpoint/'model.safetensors').read_bytes()).hexdigest()!=fit['initial_checkpoint_sha256']:raise ValueError('initial checkpoint changed')
 if Path(report['checkpoint_final']).resolve()!=checkpoint or json.loads((checkpoint/'capstone_run.json').read_text())['completed_steps']!=report['optimizer_steps']:raise ValueError('final capacity checkpoint mismatch')
 if not (checkpoint/'model.safetensors').is_file():raise FileNotFoundError(checkpoint/'model.safetensors')
 if json.loads((checkpoint/'capstone_normalization.json').read_text())!=json.loads((default_checkpoint/'capstone_normalization.json').read_text()):raise ValueError('capacity normalization differs')
 return checkpoint

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--episode-role',choices=['validation','train-debug'],default='validation',help='train-debug selects training traj_00001; never a generalization evaluation');p.add_argument('--frames',type=int,default=8,help='8-frame integration smoke by default; 0 for full recorded duration');p.add_argument('--prefix',type=int,default=4);p.add_argument('--seed',type=int,default=60000,help='model sampling seed only; scene initial state remains fixed');p.add_argument('--capacity-run',type=Path,help='explicit completed TRAIN001 fit diagnostic; default model unchanged');p.add_argument('--fixed-policy-run',type=Path,help='controller-only replay of saved VLA commands; no model process');p.add_argument('--velocity-reference',choices=['causal','zero'],default='causal');p.add_argument('--output',type=Path);a=p.parse_args()
 if a.capacity_run and a.fixed_policy_run:p.error('capacity live run cannot also replay fixed commands')
 if a.frames<0 or not 1<=a.prefix<=20:p.error('invalid frames/prefix')
 if not 0<=a.seed<=2147483647:p.error('seed must be in [0, 2147483647]')
 repo=ROOT/'data/derived/hf_pilot_render_repo';worker=repo/'Scripts/expert_data/collect_worker.py';patched=patch_policy_worker(worker.read_text())
 manifest_path=ROOT/'data/derived/task2_expansion_inventory_20261025/split_manifest.json';manifest=json.loads(manifest_path.read_text())
 record=json.loads((BASE/'run_config.json').read_text())
 if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=record['frozen_manifest_sha256']:raise ValueError('split changed')
 episode_id=select_development_episode(manifest,record,a.episode_role)
 taskid=manifest['task_id'];episode=ROOT/'data/raw/task2_expansion_20261025/Trajectory/1s1p'/taskid/episode_id;source=episode/'replay.hdf5';meta=json.loads((episode/'meta.json').read_text())
 cfg_path=ROOT/'configs/franka_action_interface.json';cfg=json.loads(cfg_path.read_text());task=repo/'Tasks/1s1p'/f'{taskid}.usd';checkpoint=BASE/'checkpoint_final'
 if a.capacity_run:checkpoint=select_capacity_checkpoint(a.capacity_run,a.episode_role,checkpoint,record['frozen_manifest_sha256'],hashlib.sha256(source.read_bytes()).hexdigest())
 for path in [source,task,checkpoint/'model.safetensors',Path(cfg['urdf']),Path(cfg['isaaclab_root'])/'isaaclab.sh']:
  if not path.is_file():raise FileNotFoundError(path)
 with h5py.File(source) as f:
  length=len(f['actions']);source_joint_order=meta['joint_names']
 if source_joint_order!=[f'panda_joint{i}' for i in range(1,8)]+['panda_finger_joint1','panda_finger_joint2']:raise ValueError('wrong robot joints')
 frames=min(a.frames or length,length)
 if meta['control_hz']!=60 or meta['arm_execution']['mode']!='position_drive':raise ValueError('wrong control contract')
 tasks=ROOT/'data/derived/task2_multiepisode_20261026/train/meta/tasks.jsonl';instruction=json.loads(json.loads(tasks.read_text().splitlines()[0])['task'])['capstone_instruction']
 if instruction!=meta['task_plan']['task_cn']:raise ValueError('instruction mismatch')
 fixed_commands=None;fixed_values=None;command_origin=None;component_sources=None
 if a.fixed_policy_run:
  parent=a.fixed_policy_run.resolve();prior=json.loads((parent/'run_config.json').read_text())
  if Path(prior['episode']).resolve()!=episode.resolve():raise ValueError('fixed replay has different initial episode')
  for rel in [str(task.relative_to(ROOT)),'configs/franka_action_interface.json','src/capstone_vla/franka.py','src/capstone_vla/task2_bridge.py']:
   if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=prior['sha256'][rel]:raise ValueError('fixed replay environment/controller changed: '+rel)
  checkpoint=Path(prior.get('checkpoint',str(checkpoint))).resolve()
  if not (checkpoint/'model.safetensors').is_file():raise FileNotFoundError(checkpoint/'model.safetensors')
  if prior.get('prediction_checkpoint_sha256') and hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest()!=prior['prediction_checkpoint_sha256']:raise ValueError('prediction checkpoint changed')
  fixed_commands,timestamp_archive,command_origin=fixed_command_files(parent,prior)
  if command_origin=='expert_model_component_diagnostic':component_sources=validate_component_bundle(prior)
  with h5py.File(fixed_commands) as f:fixed_values=f['action'][:]
  if fixed_values.shape!=(length,7) or not np.isfinite(fixed_values).all() or not np.all(np.abs(fixed_values[:,6])==1):raise ValueError('invalid original policy command sequence')
  with h5py.File(timestamp_archive) as f:np.testing.assert_allclose(f['timestamp_s'][:],np.arange(length)/60,rtol=0,atol=1e-7)
 reference_actions_used=component_sources is not None
 if reference_actions_used:patched=annotate_component_worker(patched,component_sources)
 out=(a.output or BASE/('fixed_command_runs' if fixed_commands else 'closed_loop_runs')/datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')).resolve()
 if out.exists():raise FileExistsError(out)

 step=meta['task_plan']['steps'][0];context={'schema_version':1,'source':'capstone_saved_task_metadata_probe','family':'1s1p','row':{'Task_ID':taskid,'Skill_1':step['skill'],'Skill_2':'','Phenomenon_1':step['phenomenon'],'Phenomenon_2':'','Task_CN':instruction,'Task_EN':'','build_ok':''}}
 command=[str(Path(cfg['isaaclab_root'])/'isaaclab.sh'),'-p',str(ROOT/'scripts/sim/run_task2_policy_worker.py'),'--task',str(task),'--replay_trajectories',str(source),'--replay_no_write','--pilot-comparison-out',str(out/'physics_trace.hdf5'),'--out_root',str(out/'worker_output'),'--control_hz','60','--grasp_mode','physical','--device','cuda:0','--fast_exit_after_save','--no-visualization','--experience','isaacsim.exp.base.python.kit','--kit_args=--portable-root '+str(ROOT/'data/derived/.cache/isaacsim/linux')+' --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/enabled=false --/renderer/multiGpu/maxGpuCount=1 --/app/extensions/fsWatcherEnabled=false']
 print('CPU_PREFLIGHT_OK episode='+episode_id+' frames='+str(frames)+' prefix='+str(a.prefix)+' physical_grasp=true source_actions_used='+str(reference_actions_used).lower(),flush=True)
 print('EPISODE_ROLE '+a.episode_role+' generalization_claim=false model_seed='+str(a.seed),flush=True)
 print('RUN_KIND '+('fixed_policy_commands_no_model' if fixed_commands else 'live_policy_feedback')+' velocity_reference='+a.velocity_reference,flush=True)
 print('COMMAND_ORIGIN '+str(command_origin or 'live_policy'),flush=True)
 print('POLICY_CHECKPOINT '+str(checkpoint),flush=True)
 print('SIMULATION_MODE synchronous_pause_during_inference; real_time_performance_not_evaluated',flush=True)
 print('COMMAND',shlex.join(command),flush=True)
 if not a.run:print('GPU_NOT_STARTED; add --run');return
 out.mkdir(parents=True,exist_ok=False);ipc=out/'ipc';ipc.mkdir();print('OUTPUT',out,flush=True)
 config={'capacity_training_run':str(a.capacity_run.resolve()) if a.capacity_run else None,'episode_role':a.episode_role,'dataset_split':('train' if a.episode_role=='train-debug' else 'validation'),'generalization_claim':False,'purpose':('training_episode_closed_loop_debug' if a.episode_role=='train-debug' else 'development_validation'),'output':str(out),'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest(),'instruction':instruction,'seed':a.seed,'episode':str(episode),'frames':frames,'prefix':a.prefix,'task':str(task),'control_hz':60,'servo_hz':180,'history_offsets':[-2,0],'source_actions_used_for_control':reference_actions_used,'diagnostic_component_sources':component_sources,'fixed_command_origin':command_origin,'fixed_policy_commands':str(fixed_commands) if fixed_commands else None,'fixed_command_sha256':hashlib.sha256(fixed_commands.read_bytes()).hexdigest() if fixed_commands else None,'fixed_source_config_sha256':hashlib.sha256((a.fixed_policy_run/'run_config.json').read_bytes()).hexdigest() if fixed_commands else None,'velocity_reference':a.velocity_reference,'physics_feedback':('fixed commands with IK feedback' if fixed_commands else 'live, synchronous'),'real_time':False,'official_success_verified':False,'patched_worker_sha256':hashlib.sha256(patched.encode()).hexdigest(),
 'sha256':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [worker,source,task,ROOT/'scripts/sim/run_task2_policy_worker.py',ROOT/'src/capstone_vla/component_controls.py',ROOT/'src/capstone_vla/task2_bridge.py',ROOT/'src/capstone_vla/franka.py',ROOT/'src/capstone_vla/closed_loop.py',ROOT/'scripts/task2_policy_server.py',ROOT/'src/capstone_vla/fixed_policy_replay.py',ROOT/'src/capstone_vla/closed_loop_patch.py',cfg_path]}}
 configpath=out/'run_config.json';configpath.write_text(json.dumps(config,indent=2)+'\n')
 env=dict(os.environ,TERM='xterm',OMNI_KIT_ACCEPT_EULA='YES',PYTHONDONTWRITEBYTECODE='1',CAPSTONE_TASK2_WORKER=str(worker),CAPSTONE_FRANKA_CONFIG=str(cfg_path),CAPSTONE_TASK2_MODE='cartesian',CAPSTONE_POLICY_RUN=str(out),CAPSTONE_POLICY_PREFIX=str(a.prefix),CAPSTONE_POLICY_FRAMES=str(frames),CAPSTONE_IK_SOLVER='adaptive',CAPSTONE_BOUNDED_FROM_COMMAND='0',CAPSTONE_VELOCITY_REFERENCE=a.velocity_reference,CAPSTONE_POLICY_COMMANDS=(str(fixed_commands) if fixed_commands else ''),CAPSTONE_DIAGNOSTIC_COMPONENT_SOURCES=(json.dumps(component_sources) if component_sources else ''),CAPSTONE_ACTUATOR_PROFILE='capstone',ROBOPHYS_CAPSTONE_OFFLINE_TASK='1',ROBOPHYS_TASK_CONTEXT=json.dumps(context,ensure_ascii=True))
 lab=Path(cfg['isaaclab_root']);env['PYTHONPATH']=':'.join([str(ROOT/'src'),*[str(path) for path in (lab/'source').iterdir() if path.is_dir()]])
 cache=ROOT/'data/derived/.cache/isaacsim/linux';env['OMNI_CONFIG_PATH']=str(cache/'config');env['XDG_CACHE_HOME']=str(cache/'xdg')
 server=None;sim=None
 try:
  with (out/'policy_server.log').open('x') as model_log,(out/'simulation.log').open('x') as sim_log:
   if fixed_commands is None:
    server=subprocess.Popen([str(ROOT/'.venv_dynamicvla/bin/python'),str(ROOT/'scripts/task2_policy_server.py'),'--config',str(configpath)],cwd=ROOT,env=env,stdout=model_log,stderr=subprocess.STDOUT,start_new_session=True)
    wait_file(ipc/'ready',ipc,timeout=180)
   sim=subprocess.Popen(command,cwd=repo,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',start_new_session=True)
   for line in sim.stdout:
    sim_log.write(line);sim_log.flush()
    if any(v in line for v in ['LIVE_POLICY','COLLECT ATTEMPT','PILOT COMPARISON','Traceback','Error','ABORT']):print(line.rstrip(),flush=True)
   code=sim.wait()
   if code:raise RuntimeError('simulation failed, see '+str(out/'simulation.log'))
  with h5py.File(out/'physics_trace.hdf5') as f,h5py.File(out/'live_observations.hdf5') as obs:
   if len(f['actions'])!=frames or int(f.attrs['servo_calls'])!=3*frames or len(obs['state'])!=frames:raise RuntimeError('incomplete rollout or cadence mismatch')
   if np.asarray(f['grasp_assistance_audit']).any():raise RuntimeError('unexpected attachment assistance')
   if fixed_commands:np.testing.assert_array_equal(obs['action'][:],fixed_values[:frames])
   if bool(f.attrs['expert_actions_used_for_control'])!=reference_actions_used:raise RuntimeError('expert action provenance mismatch')
   if component_sources and json.loads(f.attrs['diagnostic_component_sources_json'])!=component_sources:raise RuntimeError('component provenance mismatch')
   result={'status':('FIXED_POLICY_COMMAND_REPLAY_PASS' if fixed_commands else 'CLOSED_LOOP_INTEGRATION_PASS'),'frames':frames,'prefix':a.prefix,'servo_calls':int(f.attrs['servo_calls']),
    'bilateral_contact_frames':int(np.count_nonzero(f['contact_L'][:]&f['contact_R'][:])),
    'object_z_change_m':float(f['object_pos'][-1,0,2]-f['object_pos'][0,0,2]),
    'fixed_command_origin':command_origin,'source_actions_used_for_control':reference_actions_used,'diagnostic_component_sources':component_sources,'episode_role':a.episode_role,'generalization_claim':False,'official_success_verified':False,'real_time_evaluated':False,'new_model_inference':fixed_commands is None,'scope':('expert-component diagnostic; not learned-policy success' if reference_actions_used else ('controller-only diagnostic on saved VLA commands, not a new learned-policy rollout' if fixed_commands else 'live RGB/proprioception feedback; first integration, not benchmark success'))}
  (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(result['status']+' report='+str(out/'report.json'),flush=True)
 finally:
  (ipc/'stop').write_text('stop\n')
  for process in [sim,server]:
   if process is not None and process.poll() is None:
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
if __name__=='__main__':main()
