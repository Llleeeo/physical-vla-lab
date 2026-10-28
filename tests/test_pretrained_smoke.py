import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('capstone_model_smoke',ROOT/'scripts/smoke_task2_pretrained.py')
smoke=importlib.util.module_from_spec(spec);spec.loader.exec_module(smoke)

class ModelSmokeTests(unittest.TestCase):
    def test_pretrained_architecture_preserved_and_backbone_frozen(self):
        raw=json.loads((ROOT/'configs/dynamicvla_dom_pretrained.json').read_text())
        cfg=smoke.make_config(raw)
        self.assertEqual(cfg.input_features['observation.state'].shape,(6,))
        self.assertEqual(cfg.output_features['action'].shape,(7,))
        self.assertTrue(cfg.freeze_vision_model and cfg.freeze_text_model and cfg.freeze_connector)
        self.assertFalse(cfg.enable_streaming)

    def test_wrong_observation_history_rejected(self):
        raw=json.loads((ROOT/'configs/dynamicvla_dom_pretrained.json').read_text());raw['n_obs_steps']=1
        with self.assertRaises(ValueError):smoke.make_config(raw)

    def test_actual_upstream_loss_masks_padded_target_gradients(self):
        values=torch.ones(1,20,32,requires_grad=True)
        dummy=SimpleNamespace(config=SimpleNamespace(adapt_to_pi_aloha=False,max_action_dim=32),
              normalize_inputs=lambda b:b,normalize_targets=lambda b:b,
              prepare_images=lambda b:([],[]),prepare_state=lambda b:None,
              prepare_language=lambda b:(None,None),prepare_action=lambda b:None,
              model=SimpleNamespace(forward=lambda *args:values.square()))
        padding=torch.tensor([[False]*7+[True]*13])
        loss,details=smoke.DynamicVLAPolicy.forward(dummy,{'actions_id_pad':padding})
        self.assertAlmostEqual(loss.item(),7/20,places=6)
        loss.backward()
        self.assertTrue(torch.all(values.grad[:,7:]==0))
        self.assertTrue(torch.all(values.grad[:,:7]>0))
        self.assertIn('losses_after_in_ep_bound',details)

if __name__=='__main__':unittest.main()
