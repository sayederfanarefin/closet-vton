"""Regression checks for the empty-mask fallback and crop panel ordering."""
import json
import sys
import types
from pathlib import Path
import importlib.util
import pytest
import torch

path=Path(__file__).resolve().parents[1]/'comfy_custom_nodes/crop_integrity/__init__.py'
spec=importlib.util.spec_from_file_location('crop_integrity_test',path)
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)

def test_empty_target_mask_is_rejected_before_original_node(monkeypatch):
    image=torch.ones(1,32,24,3)
    with pytest.raises(ValueError,match='empty_target_mask'):
        mod.SafeICLoraCompose().compose(image,'patch_right',1536,'#FF0000',second_image=image,second_mask=torch.zeros(1,32,24))

def test_empty_segmenter_image_does_not_enter_fit_image():
    with pytest.raises(ValueError,match='invalid_segmented_reference'):
        mod.validate_composition_inputs(torch.zeros(1,32,24),torch.ones(1,32,24,3),torch.ones(1,32,24))

@pytest.mark.parametrize('horizontal',[True,False])
def test_preserves_target_panel_coordinates_and_actual_canvas_size(monkeypatch,horizontal):
    first=torch.zeros(1,32,24,3);target=torch.ones_like(first);mask=torch.ones(1,32,24)
    canvas=torch.cat([first,target],dim=2 if horizontal else 1)
    x,y=(24,0) if horizontal else (0,32)
    class Original:
        def add_mask(self,*args):return canvas,mask,x,y,24,32,999,999
    monkeypatch.setitem(sys.modules,'nodes',types.SimpleNamespace(NODE_CLASS_MAPPINGS={'AddMaskForICLora':Original}))
    r=mod.SafeICLoraCompose().compose(first,'patch_right' if horizontal else 'patch_bottom',1536,'#FF0000',second_image=target,second_mask=mask)
    assert r[2:]==(x,y,24,32,canvas.shape[2],canvas.shape[1])
    assert torch.equal(r[0][:,y:y+32,x:x+24],target)

def test_postcrop_rejects_reference_panel_before_person_detector():
    image=torch.zeros(1,32,24,3);canvas=torch.zeros(1,32,48,3)
    sys.modules.setdefault('nodes',types.SimpleNamespace(NODE_CLASS_MAPPINGS={}))
    with pytest.raises(ValueError,match='reference_panel_match'):
        mod.ValidatePersonCrop().validate(image,canvas,torch.ones(1,32,24),None,24,0,24,32)
