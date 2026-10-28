"""Reject overlapping allocations, incomplete delivery and wrong saved controls."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


worker = module('run_hgwind_production_worker')
dual = module('collect_hgwind_dual')


class ProductionChecks(unittest.TestCase):
    def test_host_seed_swap_rejected(self):
        with self.assertRaises(ValueError):
            worker.validate_allocation('hongtao', 202610280, 200, 50, False)

    def test_smoke_cannot_use_production_namespace(self):
        with self.assertRaises(ValueError):
            worker.validate_allocation('local', 202610280, 200, 50, True)

    def test_actual_sampler_has_400_distinct_attempt_seeds(self):
        import sys
        sys.path.insert(0, str(ROOT / 'data/derived/hgwind_current_baseline_20261027/repo/Scripts'))
        from dataset.trajectory_variation import sample_trajectory_variation
        actual = [sample_trajectory_variation(base, i).attempt_seed
                  for base in (202610280, 202710280) for i in range(200)]
        self.assertEqual(len(set(actual)), 400)

    def test_delivered_payload_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'replay.hdf5').write_bytes(b'recorded states')
            receipt = {'files': [{'path': 'replay.hdf5', 'bytes': 15,
                                 'sha256': dual.digest(root / 'replay.hdf5')}]}
            (root / 'transport_manifest.json').write_text(json.dumps(receipt))
            dual.verify_transport(root)
            (root / 'replay.hdf5').write_bytes(b'changed states!')
            with self.assertRaises(RuntimeError):
                dual.verify_transport(root)

    def test_unlisted_stale_delivery_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'transport_manifest.json').write_text(json.dumps({'files': []}))
            (root / 'stale_episode.json').write_text('{}')
            with self.assertRaises(RuntimeError):
                dual.verify_transport(root)

    def test_parent_transport_paths_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'transport_manifest.json').write_text(json.dumps({'files': [
                {'path': '../outside', 'bytes': 1, 'sha256': 'none'}]}))
            with self.assertRaises(RuntimeError):
                dual.verify_transport(root)

    def test_saved_success_requires_physical_grasp_and_validity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ep = root / 'Trajectory/1s1p' / worker.TASK_ID / 'traj_00000'
            ep.mkdir(parents=True)
            (ep / 'replay.hdf5').write_bytes(b'fixture')
            metadata = {'success': True, 'trajectory_validity': {'passed': True},
                        'grasp_mode': 'physical_finger_contact_only', 'attempt_index': 0,
                        'num_steps': 1, 'capstone_expert_control': {'profile': 'smooth_lift_v1'}}
            (ep / 'meta.json').write_text(json.dumps(metadata))
            self.assertTrue(worker.episode_summary(root)[0]['success'])
            metadata['grasp_mode'] = 'fixedforce_after_bilateral_fingertip_contact'
            (ep / 'meta.json').write_text(json.dumps(metadata))
            self.assertFalse(worker.episode_summary(root)[0]['success'])
            metadata['capstone_expert_control']['profile'] = 'another_profile'
            (ep / 'meta.json').write_text(json.dumps(metadata))
            with self.assertRaises(RuntimeError):
                worker.episode_summary(root)


if __name__ == '__main__':
    unittest.main()
