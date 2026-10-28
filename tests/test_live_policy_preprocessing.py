import json,sys,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from task2_policy_server import prepare_input,DynamicVLAPolicy,make_config

class LivePreprocessingTests(unittest.TestCase):
 def setUp(self):
  self.cfg=make_config(json.loads((ROOT/'configs/dynamicvla_dom_pretrained.json').read_text()))
  self.owner=SimpleNamespace(config=self.cfg)
  self.rgb={name:np.zeros((2,480,640,3),np.uint8) for name in ['wrist_cam','opst_cam']}
  for value in self.rgb.values():value[:,:,:,1]=128;value[:,:,:,2]=255
 def test_rgb_and_history_survive_upstream_view_preprocessing(self):
  states=np.arange(12).reshape(2,6);batch=prepare_input(states,self.rgb,'test')
  for key in self.cfg.image_features:
   self.assertTrue(batch[key].is_contiguous());self.assertEqual(tuple(batch[key].shape),(1,2,3,360,480))
   np.testing.assert_allclose(batch[key][0,0,:,10,10].numpy(),[0,128/255,1],atol=1e-6)
  np.testing.assert_array_equal(batch['observation.state'][0].numpy(),states)
  images,masks=DynamicVLAPolicy.prepare_images(self.owner,batch)
  self.assertEqual(len(images),2);self.assertEqual(tuple(images[0].shape),(1,6,384,384));self.assertTrue(all(mask.all() for mask in masks))
 def test_old_noncontiguous_layout_reproduces_the_reported_error(self):
  batch=prepare_input(np.zeros((2,6)),self.rgb,'test')
  for key in self.cfg.image_features:
   batch[key]=batch[key].permute(0,1,3,4,2).contiguous().permute(0,1,4,2,3)
   self.assertFalse(batch[key].is_contiguous())
  with self.assertRaisesRegex(RuntimeError,'view size is not compatible'):
   DynamicVLAPolicy.prepare_images(self.owner,batch)
