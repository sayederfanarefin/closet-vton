import numpy as np
from comfy_custom_nodes.crop_integrity.segmentation_fallback import assess_mask
from comfy_custom_nodes.crop_integrity.segmentation_rescue import clean_components,validate_semantic

def test_disconnected_garment_parts_preserved_but_speckle_removed():
 a=np.zeros((100,100),bool);a[25:60,30:70]=True;a[25:60,10:20]=True;a[25:60,80:90]=True;a[90,90]=True
 b=clean_components(a)
 assert b[30,15] and b[30,50] and b[30,85] and not b[90,90]

def test_low_crop_top_uses_semantic_waist_evidence():
 a=np.zeros((100,100),bool);a[47:62,25:75]=True;labels=np.zeros((100,100),np.uint8);labels[a]=4;labels[80:100,20:80]=6
 assert assess_mask(a,'target',[0,0,100,100])[1]['reason']=='mask_misses_chest'
 ok,q=validate_semantic(a,labels,{'flip_iou':.97,'mean_class_confidence':.95},'target',[0,0,100,100],None,assess_mask)
 assert ok and q['chest_check']=='semantic_upper_clothes_above_waist'

def test_lower_body_mask_cannot_use_crop_top_override():
 a=np.zeros((100,100),bool);a[47:62,25:75]=True;labels=np.zeros((100,100),np.uint8);labels[a]=5;labels[80:100,20:80]=6
 ok,q=validate_semantic(a,labels,{'flip_iou':.97,'mean_class_confidence':.95},'target',[0,0,100,100],None,assess_mask)
 assert not ok and q['reason']=='mask_misses_chest'

def test_semantic_parser_must_agree_under_horizontal_flip():
 a=np.zeros((100,100),bool);a[25:60,30:70]=True;labels=a.astype(np.uint8)*4
 ok,q=validate_semantic(a,labels,{'flip_iou':.50,'mean_class_confidence':.99},'reference',None,None,assess_mask)
 assert not ok and q['reason']=='parser_flip_inconsistent'

def test_semantic_override_keeps_face_protection():
 a=np.zeros((100,100),bool);a[10:65,20:80]=True;labels=a.astype(np.uint8)*4
 ok,q=validate_semantic(a,labels,{'flip_iou':.99,'mean_class_confidence':.99},'target',[0,0,100,100],[20,0,80,25],assess_mask)
 assert not ok and q['reason']=='face_or_head_overlap'
