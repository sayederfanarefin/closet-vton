import importlib.util,sys,types
from pathlib import Path
import numpy as np
import torch

p=Path(__file__).resolve().parents[1]/'comfy_custom_nodes/crop_integrity/segmentation_fallback.py'
spec=importlib.util.spec_from_file_location('fallback_unit',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_empty_nonfinite_and_full_person_masks_rejected():
 for a,reason in [(np.zeros((100,100)),'empty_mask'),(np.full((100,100),np.nan),'nonfinite_mask'),(np.ones((100,100)),'implausible_mask_area')]:
  valid,q=m.assess_mask(a,'target',[0,0,100,100],[0,0,100,20]);assert not valid and q['reason']==reason

def test_face_overlap_rejected_even_for_otherwise_plausible_mask():
 a=np.zeros((100,100));a[5:70,20:80]=1
 valid,q=m.assess_mask(a,'target',[0,0,100,100],[20,0,80,20]);assert not valid and q['reason']=='face_or_head_overlap'

def test_clothing_region_accepted_and_lower_body_rejected():
 a=np.zeros((100,100));a[25:65,20:80]=1
 assert m.assess_mask(a,'target',[0,0,100,100],[0,0,100,20])[0]
 b=np.zeros((100,100));b[80:99,20:80]=1
 assert m.assess_mask(b,'target',[0,0,100,100],[0,0,100,20])[1]['reason']=='mask_outside_upper_body'

def test_unreliable_sam_masks_rejected():
 a=np.zeros((100,100));a[25:65,20:80]=1
 assert m.assess_mask(a,'reference',predicted_iou=.5)[1]['reason']=='low_sam_quality'
 assert m.assess_mask(a,'reference',stability=.5)[1]['reason']=='unstable_sam_mask'

def test_skirt_mask_rejected_even_when_inside_broad_upper_body_zone():
 a=np.zeros((100,100));a[47:74,20:80]=1
 assert m.assess_mask(a,'target',[0,0,100,100])[1]['reason']=='mask_misses_chest'

def test_primary_success_is_returned_unchanged_without_fallback(monkeypatch):
 image=torch.rand(1,32,24,3);mask=torch.ones(1,32,24)
 class Original:
  def main(self,*args):return image,mask
 monkeypatch.setitem(sys.modules,'nodes',types.SimpleNamespace(NODE_CLASS_MAPPINGS={'GroundingDinoSAMSegment (segment anything)':Original}))
 r=m.AdaptiveClothingSegment().segment(None,None,image,'outfit',.75,'target')
 assert r['result'][0] is image and r['result'][1] is mask
