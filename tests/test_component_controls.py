import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from capstone_vla.component_controls import compose_commands,validate_component_bundle,annotate_component_worker,VARIANTS

class ComponentControlsTests(unittest.TestCase):
    def setUp(self):
        self.expert=np.zeros((8,7));self.model=np.ones((8,7))
        self.expert[:,:6]=np.arange(48).reshape(8,6)/100
        self.model[:,:6]+=np.arange(48).reshape(8,6)/10
        self.expert[:,6]=[1,1,1,-1,-1,-1,-1,-1]
        self.model[:,6]=[1,-1,1,1,1,-1,1,-1]

    def test_channels_preserve_exact_values_and_event_times(self):
        before_a,before_b=self.expert.copy(),self.model.copy()
        for variant,expected in VARIANTS.items():
            result,sources=compose_commands(self.expert,self.model,variant)
            np.testing.assert_array_equal(result[:,:6],(self.expert if expected['pose']=='expert' else self.model)[:,:6])
            np.testing.assert_array_equal(result[:,6],(self.expert if expected['gripper']=='expert' else self.model)[:,6])
            self.assertEqual(sources,expected)
        np.testing.assert_array_equal(self.expert,before_a);np.testing.assert_array_equal(self.model,before_b)

    def test_invalid_commands_cannot_be_silently_relabelled(self):
        invalid=self.model.copy();invalid[0,6]=0
        with self.assertRaises(ValueError):compose_commands(self.expert,invalid,'expert_all')
        invalid=self.model.copy();invalid[0,0]=np.nan
        with self.assertRaises(ValueError):compose_commands(self.expert,invalid,'expert_all')
        with self.assertRaises(ValueError):compose_commands(self.expert,self.model[:4],'expert_all')
        with self.assertRaises(ValueError):compose_commands(self.expert,self.model,'unknown')

    def test_bundle_must_disclose_expert_channels(self):
        prior={'schema':'capstone_component_command_bundle_v1','training_eligible':False,'component_variant':'model_pose_expert_gripper','component_sources':{'pose':'model','gripper':'expert'}}
        self.assertEqual(validate_component_bundle(prior),prior['component_sources'])
        prior['component_sources']['gripper']='model'
        with self.assertRaises(ValueError):validate_component_bundle(prior)
        prior['component_sources']['gripper']='expert';prior['training_eligible']=True
        with self.assertRaises(ValueError):validate_component_bundle(prior)

    def test_worker_annotation_changes_only_provenance(self):
        anchor="                    pilot_trace.attrs['expert_actions_used_for_control'] = False"
        source='before\n'+anchor+'\nafter\n'
        out=annotate_component_worker(source,{'pose':'expert','gripper':'model'})
        self.assertTrue(out.startswith('before\n'));self.assertTrue(out.endswith('\nafter\n'))
        self.assertIn("expert_actions_used_for_control'] = True",out)
        self.assertIn('diagnostic_component_sources_json',out)
        with self.assertRaises(ValueError):annotate_component_worker('missing',{'pose':'expert','gripper':'model'})

if __name__=='__main__':unittest.main()
