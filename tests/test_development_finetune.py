import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('capstone_development_train',ROOT/'scripts/finetune_task2_development.py')
training=importlib.util.module_from_spec(spec);spec.loader.exec_module(training)

class DevelopmentTrainingTests(unittest.TestCase):
    def test_report_metric_excludes_padded_steps_and_artificial_dimensions(self):
        losses=torch.full((1,2,32),1000.)
        losses[0,0,:7]=4.
        self.assertEqual(float(training.valid_flow_mse(losses,torch.tensor([[False,True]]))),4.)
        with self.assertRaises(ValueError):training.valid_flow_mse(losses,torch.ones((1,2),dtype=torch.bool))

    def test_invalid_or_mislabelled_training_plan_rejected(self):
        plan=json.loads((ROOT/'configs/task2_development_finetune.json').read_text())
        training.validate_plan(plan,232)
        for name,value in [('steps',0),('probe_frames',[232]),('heldout_evaluation',True)]:
            with self.assertRaises(ValueError):training.validate_plan({**plan,name:value},232)

    def test_official_checkpoint_serializer_and_sidecar_on_tiny_fixture(self):
        from smoke_task2_pretrained import DynamicVLAPolicy,make_config
        policy=DynamicVLAPolicy.__new__(DynamicVLAPolicy)
        torch.nn.Module.__init__(policy)
        policy.config=make_config(json.loads((ROOT/'configs/dynamicvla_dom_pretrained.json').read_text()))
        policy.model=torch.nn.Module()
        policy.model.action_out_proj=torch.nn.Linear(4,7)
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'checkpoint'
            training.checkpoint(policy,target,{'action':{'mean':np.zeros(7),'std':np.ones(7)}},{'fixture':True})
            self.assertTrue((target/'config.json').is_file())
            self.assertTrue((target/'model.safetensors').is_file())
            stats=json.loads((target/'capstone_normalization.json').read_text())
            self.assertEqual(stats['action']['std'],[1.]*7)

if __name__=='__main__':unittest.main()
