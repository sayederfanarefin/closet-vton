"""Fail-closed IC-LoRA composition and post-crop validation.

Install this directory under ComfyUI/custom_nodes. Requires the original IC-LoRA
and Segment Anything packs. Primary segmentation is unchanged; failed detection
can use the bounded AdaptiveClothingSegment fallback. Models/sampler are unchanged.
"""
import json
import sys
import time
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def mask_diagnostics(mask):
    if mask is None:
        return {'rho': None, 'mask_bbox': None, 'mask_nonzero': 0}
    a = mask.detach().float().cpu().numpy()
    if a.ndim == 3:
        a = a[0]
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError('Target mask must be a finite [B,H,W] tensor')
    yy, xx = np.where(a > .5)
    return {'rho': float((a > .5).mean()),
            'mask_bbox': None if not len(xx) else [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)],
            'mask_nonzero': int(np.count_nonzero(a)),
            'mask_touches_edge': bool(len(xx) and (xx.min()==0 or yy.min()==0 or xx.max()==a.shape[1]-1 or yy.max()==a.shape[0]-1))}


def reject(category, diagnostics):
    diagnostics = {**diagnostics, 'peak_cuda_allocated_bytes': int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0}
    raise ValueError('CROP_VALIDITY_REJECTED ' + json.dumps({'category': category, 'valid': False, **diagnostics}, sort_keys=True))


def validate_composition_inputs(first_image, second_image, second_mask):
    d = mask_diagnostics(second_mask)
    if second_image is None or second_image.ndim != 4 or second_image.shape[-1] != 3:
        reject('invalid_target_image', d)
    if first_image.ndim != 4 or first_image.shape[-1] != 3:
        reject('invalid_segmented_reference', d)
    if second_mask is None or d['mask_nonzero'] == 0:
        reject('empty_target_mask', d)
    return d


class SafeICLoraCompose:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'first_image': ('IMAGE',),
                             'patch_mode': (['auto', 'patch_right', 'patch_bottom'],),
                             'output_length': ('INT', {'default': 1536}),
                             'patch_color': (['#FF0000', '#00FF00', '#0000FF', '#FFFFFF'],)},
                'optional': {'first_mask': ('MASK',), 'second_image': ('IMAGE',), 'second_mask': ('MASK',)}}
    RETURN_TYPES = ('IMAGE', 'MASK', 'INT', 'INT', 'INT', 'INT', 'INT', 'INT')
    RETURN_NAMES = ('IMAGE', 'MASK', 'x_offset', 'y_offset', 'target_width', 'target_height', 'total_width', 'total_height')
    FUNCTION = 'compose'
    CATEGORY = 'crop_integrity'

    def compose(self, first_image, patch_mode, output_length, patch_color,
                first_mask=None, second_image=None, second_mask=None):
        diagnostics = validate_composition_inputs(first_image, second_image, second_mask)
        import nodes
        original = nodes.NODE_CLASS_MAPPINGS['AddMaskForICLora']()
        result = original.add_mask(first_image, patch_mode, output_length, patch_color,
                                   first_mask, second_image, second_mask)
        canvas, mask, x, y, width, height, _, _ = result
        # Derive the target rectangle from the actual fitted panel and canvas.
        # Neither mask bounds nor input orientation determines panel assignment.
        horizontal = canvas.shape[2] == 2 * width and canvas.shape[1] == height
        vertical = canvas.shape[1] == 2 * height and canvas.shape[2] == width
        if horizontal:
            expected_x, expected_y = width, 0
        elif vertical:
            expected_x, expected_y = 0, height
        else:
            reject('invalid_canvas_geometry', {**diagnostics, 'canvas_shape': list(canvas.shape)})
        if (x, y) != (expected_x, expected_y):
            reject('crop_origin_mismatch', {**diagnostics, 'actual': [x,y], 'expected': [expected_x,expected_y]})
        # Unlike the original node, total_width/height use BCHW axes correctly.
        return canvas, mask, expected_x, expected_y, width, height, canvas.shape[2], canvas.shape[1]


class ValidatePersonCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',), 'canvas': ('IMAGE',), 'target_mask': ('MASK',),
                             'grounding_dino_model': ('GROUNDING_DINO_MODEL',),
                             **{k: ('INT', {'default': 0}) for k in ['x', 'y', 'width', 'height']}}}
    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'validate'
    CATEGORY = 'crop_integrity'
    OUTPUT_NODE = True

    def validate(self, image, canvas, target_mask, grounding_dino_model, x, y, width, height):
        import nodes
        start = time.perf_counter()
        d = mask_diagnostics(target_mask)
        if tuple(image.shape[1:3]) != (height, width) or min(width,height) <= 0:
            reject('invalid_crop_dimensions', {**d, 'crop_shape': list(image.shape)})
        if x < 0 or y < 0 or x+width > canvas.shape[2] or y+height > canvas.shape[1]:
            reject('crop_out_of_canvas', d)
        reference = canvas[:, :height, :width, :]
        # Fixed RGB MAE test against the actual segmented reference panel.
        d['reference_panel_mae'] = float((image-reference).abs().mean())
        if d['reference_panel_mae'] < .04:
            reject('reference_panel_match', d)
        segment_class = nodes.NODE_CLASS_MAPPINGS['GroundingDinoSAMSegment (segment anything)']
        predict = sys.modules[segment_class.__module__].groundingdino_predict
        pil = Image.fromarray(np.clip(image[0].detach().cpu().numpy()*255,0,255).astype(np.uint8))
        # This is a separate output validator, not a change to outfit threshold .75.
        boxes = predict(grounding_dino_model, pil, 'person', .25)
        d['person_box_count'] = int(boxes.shape[0])
        d['person_boxes'] = boxes.tolist()
        d['person_detector_threshold'] = .25
        d['validation_sec'] = time.perf_counter()-start
        d['peak_cuda_allocated_bytes'] = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        if not len(boxes):
            reject('no_person_detected', d)
        d.update(valid=True, category='valid', timestamp=datetime.now(timezone.utc).isoformat())
        return {'ui': {'crop_validity': [json.dumps(d)]}, 'result': (image,)}

NODE_CLASS_MAPPINGS = {'SafeICLoraCompose': SafeICLoraCompose, 'ValidatePersonCrop': ValidatePersonCrop}

from .segmentation_fallback import AdaptiveClothingSegment
NODE_CLASS_MAPPINGS['AdaptiveClothingSegment'] = AdaptiveClothingSegment
