"""Independent, image-derived recovery after the existing policy exhausts candidates."""
import cv2
import numpy as np
import torch

def clean_components(mask,minimum=16):
    """Keep disconnected garment parts, discarding only isolated speckles."""
    n,labels,stats,_=cv2.connectedComponentsWithStats(np.asarray(mask,dtype=np.uint8),8)
    ids=[i for i in range(1,n) if stats[i,cv2.CC_STAT_AREA]>=minimum]
    return np.isin(labels,ids)

def upper_garment_above_waist(mask,labels):
    """Evidence for low-cut tops that a fixed full-person chest band misses."""
    lower=np.isin(labels,[5,6]);ys=np.where(lower)[0]
    if lower.mean()<.02 or not mask.any():return False,{}
    waist=float(np.percentile(ys,2));above=float((np.where(mask)[0]<waist).mean())
    upper=float((mask&(labels==4)).sum()/mask.sum())
    return above>=.90 and upper>=.90,{'parsed_waist_y':waist,'garment_above_waist_fraction':above,'upper_clothes_label_fraction':upper}

def validate_semantic(mask,labels,info,role,person_box,head_box,assess):
    if info['flip_iou']<.80:return False,{'reason':'parser_flip_inconsistent',**info}
    if info['mean_class_confidence']<.65:return False,{'reason':'parser_low_confidence',**info}
    ok,q=assess(mask,role,person_box,head_box)
    # The semantic model identifies Upper-clothes independently of DINO/SAM.
    # A crop top can sit below the fixed chest band while remaining above pants.
    if not ok and role=='target' and q['reason']=='mask_misses_chest':
        evidence,anatomy=upper_garment_above_waist(mask,labels)
        if evidence:ok=True;q.update(reason='accepted',chest_check='semantic_upper_clothes_above_waist',**anatomy)
    q.pop('predicted_iou',None);q.pop('stability',None)
    return ok,{**q,**info,'quality_basis':'semantic_posterior_and_flip_consistency'}

def recover(rgb,role,person_box,head_box,report,module,sam_model,predictor,assess):
    details=[];h,w=rgb.shape[:2]
    if role=='reference':
        boxes=[];seen=set()
        for attempt in report['attempts']:
            for c in attempt['candidates']:
                if 'box_xyxy' not in c:continue
                key=tuple(int(v/4) for v in c['box_xyxy'])
                if key not in seen:boxes.append(c['box_xyxy']);seen.add(key)
        for coordinates in boxes[:3]:
            if predictor is None:
                predictor=module.SamPredictorHQ(sam_model,hasattr(sam_model,'model_name') and 'hq' in sam_model.model_name);predictor.set_image(rgb)
            box=torch.tensor([coordinates]);transformed=predictor.transform.apply_boxes_torch(box,rgb.shape[:2]).to(module.comfy.model_management.get_torch_device())
            with torch.inference_mode():logits,ious,_=predictor.predict_torch(point_coords=None,point_labels=None,boxes=transformed,multimask_output=True,return_logits=True)
            for i in torch.argsort(ious[0],descending=True).tolist():
                a=logits[0,i].detach().cpu().numpy();mask=clean_components(a>float(sam_model.mask_threshold));stability=float((a>1).sum()/max(1,(a>-1).sum()))
                ok,q=assess(mask,role,person_box,head_box,float(ious[0,i]),stability)
                x0,y0,x1,y1=map(int,coordinates);support=float(mask[max(0,y0-12):min(h,y1+12),max(0,x0-12):min(w,x1+12)].sum()/max(1,mask.sum()))
                q.update(engine='sam_multimask',multimask_index=i,box_xyxy=coordinates,box_support=support,component_policy='all_box_supported_parts')
                if support<.98:ok=False;q['reason']='mask_outside_detection_box'
                details.append(q)
                if ok:return mask,q,details
    from .semantic_parser import clothing_candidate
    mask,labels,info=clothing_candidate(rgb,'atr');mask=clean_components(mask)
    ok,q=validate_semantic(mask,labels,info,role,person_box,head_box,assess);details.append(q)
    if ok:return mask,q,details
    return None,None,details
