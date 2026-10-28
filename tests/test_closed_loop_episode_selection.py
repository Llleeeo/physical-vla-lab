import importlib.util
from pathlib import Path
import unittest
import tempfile,json,hashlib

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('closed_loop_runner', ROOT/'scripts/run_task2_closed_loop.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

class EpisodeSelectionTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {'selected': {'validation': [{'episode': 'traj_00042'}]}}
        self.record = {'train_episodes': ['traj_00001'], 'validation_episodes': ['traj_00042']}

    def test_explicit_train_debug_and_validation(self):
        self.assertEqual(runner.select_development_episode(self.manifest, self.record, 'train-debug'), 'traj_00001')
        self.assertEqual(runner.select_development_episode(self.manifest, self.record, 'validation'), 'traj_00042')

    def test_split_overlap_or_absent_membership_rejected(self):
        for role, expected, forbidden, episode in [
            ('train-debug', 'train_episodes', 'validation_episodes', 'traj_00001'),
            ('validation', 'validation_episodes', 'train_episodes', 'traj_00042'),
        ]:
            missing = {key: list(value) for key, value in self.record.items()}
            missing[expected].remove(episode)
            with self.assertRaises(ValueError):runner.select_development_episode(self.manifest, missing, role)
            overlap = {key: list(value) for key, value in self.record.items()}
            overlap[forbidden].append(episode)
            with self.assertRaises(ValueError):runner.select_development_episode(self.manifest, overlap, role)

    def test_unknown_role_and_changed_validation_order_rejected(self):
        with self.assertRaises(ValueError):runner.select_development_episode(self.manifest, self.record, 'test')
        self.manifest['selected']['validation'][0]['episode'] = 'traj_00043'
        with self.assertRaises(ValueError):runner.select_development_episode(self.manifest, self.record, 'validation')


    def test_fixed_command_origin_preserves_live_default(self):
        root=Path('/tmp/example-command-run')
        command,clock,origin=runner.fixed_command_files(root,{})
        self.assertEqual(command,root/'live_observations.hdf5')
        self.assertEqual(clock,root/'physics_trace.hdf5')
        self.assertEqual(origin,'saved_live_policy')
        bundle={'command_origin':'recorded_expert_observation_predictions','schema':'capstone_teacher_command_bundle_v1','training_eligible':False}
        command,clock,origin=runner.fixed_command_files(root,bundle)
        self.assertEqual(command,root/'commands.hdf5')
        self.assertEqual(clock,command)
        with self.assertRaises(ValueError):runner.fixed_command_files(root,{'command_origin':'unknown'})
        bundle['training_eligible']=True
        with self.assertRaises(ValueError):runner.fixed_command_files(root,bundle)


    def test_capacity_weights_rejected_for_validation(self):
        with self.assertRaisesRegex(ValueError,'TRAIN debugging'):
            runner.select_capacity_checkpoint(Path('/tmp/unused'), 'validation',Path('/tmp/base'),'manifest','source')

    def test_capacity_weight_source_and_normalization_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);base=root/'base';base.mkdir();fit=root/'fit';final=fit/'checkpoint_final';final.mkdir(parents=True)
            (base/'model.safetensors').write_bytes(b'fixture-initial')
            (final/'model.safetensors').write_bytes(b'fixture-final')
            stats={'action':{'mean':[0.0],'std':[1.0]}}
            for folder in [base,final]:(folder/'capstone_normalization.json').write_text(json.dumps(stats))
            metadata={'train_episodes':['traj_00001'],'validation_opened':False,'reserved_test_opened':False,'frozen_manifest_sha256':'manifest','source_replay_sha256':'source','initial_checkpoint':str(base),'initial_checkpoint_sha256':hashlib.sha256((base/'model.safetensors').read_bytes()).hexdigest()}
            (fit/'run_config.json').write_text(json.dumps(metadata));(fit/'report.json').write_text(json.dumps({'status':'SINGLE_EPISODE_CAPACITY_COMPLETED','checkpoint_final':str(final),'optimizer_steps':1000}));(final/'capstone_run.json').write_text(json.dumps({'completed_steps':1000}))
            self.assertEqual(runner.select_capacity_checkpoint(fit,'train-debug',base,'manifest','source'),final)
            with self.assertRaisesRegex(ValueError,'source/split mismatch'):runner.select_capacity_checkpoint(fit,'train-debug',base,'manifest','wrong')
            (final/'capstone_normalization.json').write_text(json.dumps({'action':{'mean':[1.0],'std':[1.0]}}))
            with self.assertRaisesRegex(ValueError,'normalization differs'):runner.select_capacity_checkpoint(fit,'train-debug',base,'manifest','source')

if __name__ == '__main__':unittest.main()
