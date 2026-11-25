"""Bounded clothing-only segmentation recovery; primary successes are unchanged."""
import hashlib
import json
import sys
import time
import numpy as np
import torch
from PIL import Image


def assess_mask(mask, role, person_box=None, head_box=None, predicted_iou=1., stability=1.):
    """Conservative fallback-only geometry checks. Never consumes evaluation masks."""
    raw=np.asarray(mask)
    if not np.isfinite(raw).all():return False,{'reason':'nonfinite_mask'}
    a=raw.astype(bool)
    if a.ndim!=2 or not a.any():return False,{'reason':'empty_mask','rho':0.}
    h,w=a.shape;rho=float(a.mean());yy,xx=np.where(a)
    info={'rho':rho,'bbox':[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],'predicted_iou':float(predicted_iou),'stability':float(stability)}
    if not .02<=rho<(.70 if role=='target' else .90):return False,{**info,'reason':'implausible_mask_area'}
    if predicted_iou<.80:return False,{**info,'reason':'low_sam_quality'}
    if stability<.85:return False,{**info,'reason':'unstable_sam_mask'}
    if head_box is not None:
        x0,y0,x1,y1=[int(v) for v in head_box];protected=a[max(0,y0):min(h,y1),max(0,x0):min(w,x1)]
        overlap=float(protected.mean()) if protected.size else 0.;info['head_overlap']=overlap
        if overlap>.10:return False,{**info,'reason':'face_or_head_overlap'}
    if role=='target':
        if person_box is None:return False,{**info,'reason':'no_target_person_anchor'}
        x0,y0,x1,y1=person_box;ph=y1-y0
        zone=np.zeros_like(a);zone[max(0,int(y0+.18*ph)):min(h,int(y0+.75*ph)),max(0,int(x0)):min(w,int(x1))]=True
        torso=float((a&zone).sum()/a.sum());info['upper_body_fraction']=torso
        if torso<.55:return False,{**info,'reason':'mask_outside_upper_body'}
        chest=a[max(0,int(y0+.20*ph)):min(h,int(y0+.43*ph)),max(0,int(x0)):min(w,int(x1))]
        chest_fraction=float(chest.sum()/a.sum());info['chest_fraction']=chest_fraction
        if chest_fraction<.01:return False,{**info,'reason':'mask_misses_chest'}
    return True,{**info,'reason':'accepted'}


def score_boxes(module, model, image, prompt):
    transform=module.T.Compose([module.T.RandomResize([800],max_size=1333),module.T.ToTensor(),module.T.Normalize([.485,.456,.406],[.229,.224,.225])])
    tensor,_=transform(image.convert('RGB'),None);device=module.comfy.model_management.get_torch_device()
    caption=prompt.strip().lower()
    if not caption.endswith('.'):caption+='.'
    with torch.inference_mode():out=model(tensor.to(device)[None],captions=[caption])
    scores=out['pred_logits'].sigmoid()[0].max(dim=1)[0].detach().cpu();boxes=out['pred_boxes'][0].detach().cpu()
    width,height=image.size;boxes=boxes*torch.tensor([width,height,width,height]);boxes[:,:2]-=boxes[:,2:]/2;boxes[:,2:]+=boxes[:,:2]
    return boxes,scores


def find_head(rgb,person_box):
    import cv2
    h,w=rgb.shape[:2];scale=384/h;small=cv2.resize(rgb,(max(1,int(w*scale)),384));gray=cv2.cvtColor(small,cv2.COLOR_RGB2GRAY)
    detector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    faces=detector.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(18,18))
    candidates=[]
    for x,y,fw,fh in faces:
        x,y,fw,fh=[float(v)/scale for v in (x,y,fw,fh)]
        if person_box is not None and y+fh/2>person_box[1]+.35*(person_box[3]-person_box[1]):continue
        if person_box is None and y+fh/2>.4*h:continue
        candidates.append((fw*fh,[max(0,x-.15*fw),max(0,y-.2*fh),min(w,x+1.15*fw),min(h,y+1.2*fh)]))
    if candidates:return max(candidates,key=lambda p:p[0])[1],'haar_face'
    if person_box is not None:
        x0,y0,x1,y1=person_box
        return [x0,y0,x1,y0+.20*(y1-y0)],'person_head_prior'
    return None,'none'


class AdaptiveClothingSegment:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'sam_model':('SAM_MODEL',),'grounding_dino_model':('GROUNDING_DINO_MODEL',),'image':('IMAGE',),'prompt':('STRING',{'default':'outfit'}),'threshold':('FLOAT',{'default':.75}),'role':(['target','reference'],)}}
    RETURN_TYPES=('IMAGE','MASK');FUNCTION='segment';CATEGORY='crop_integrity';OUTPUT_NODE=True

    def segment(self,sam_model,grounding_dino_model,image,prompt,threshold,role):
        import nodes
        start=time.perf_counter();original_class=nodes.NODE_CLASS_MAPPINGS['GroundingDinoSAMSegment (segment anything)'];module=sys.modules[original_class.__module__]
        primary_image,primary_mask=original_class().main(grounding_dino_model,sam_model,image,prompt,threshold)
        report={'role':role,'primary_prompt':prompt,'primary_threshold':threshold,'used_fallback':False,'attempts':[],'policy_version':'semantic_rescue_v1'}
        if primary_image.ndim==4 and primary_image.shape[-1]==3 and bool(primary_mask.any()):
            report.update(selected='primary_unchanged',valid=True,rho=float((primary_mask>.5).float().mean()),elapsed_sec=time.perf_counter()-start)
            return {'ui':{'segmentation_fallback':[json.dumps(report)]},'result':(primary_image,primary_mask)}
        if image.shape[0]!=1:raise ValueError('Adaptive clothing fallback supports one input photo per request')
        rgb=np.clip(image[0].detach().cpu().numpy()*255,0,255).astype(np.uint8);pil=Image.fromarray(rgb);h,w=rgb.shape[:2]
        person_box=None
        if role=='target':
            pb,ps=score_boxes(module,grounding_dino_model,pil,'person');idx=int(ps.argmax())
            if float(ps[idx])>.25:person_box=pb[idx].tolist()
        head_box,head_source=find_head(rgb,person_box)
        report.update(person_box=person_box,head_box=head_box,head_source=head_source)
        # Reference photos may be flat lays: protect a detected face, but don't infer
        # a human head solely from a garment-shaped false-positive person box.
        prompts=['shirt','clothing','dress . blouse . sweater . jacket . camisole . tank top .']
        # Prefer an upper-garment detection over a broad outfit box, which can
        # include an unrelated skirt/trousers. Broad prompts are last resort.
        stages=[(p,.75) for p in prompts]+[(p,t) for t in [.60,.50] for p in [prompts[0],prompts[2]]]+[(p,t) for t in [.60,.50] for p in [prompts[1],prompt]]
        predictions={};predictor=None;seen=set();sam_count=0
        for candidate_prompt,cutoff in stages:
            if candidate_prompt not in predictions:predictions[candidate_prompt]=score_boxes(module,grounding_dino_model,pil,candidate_prompt)
            boxes,scores=predictions[candidate_prompt];indices=torch.argsort(scores,descending=True);attempt={'prompt':candidate_prompt,'threshold':cutoff,'maximum_score':float(scores.max()),'candidates':[]};report['attempts'].append(attempt)
            for idx in [int(i) for i in indices if float(scores[i])>cutoff][:3]:
                box=boxes[idx].clone();box[0::2].clamp_(0,w);box[1::2].clamp_(0,h);key=tuple(int(v/4) for v in box)
                if key in seen:continue
                seen.add(key)
                entry={'score':float(scores[idx]),'box_xyxy':box.tolist()};attempt['candidates'].append(entry)
                if sam_count>=12:entry['reason']='sam_candidate_budget_exhausted';continue
                if predictor is None:
                    predictor=module.SamPredictorHQ(sam_model,hasattr(sam_model,'model_name') and 'hq' in sam_model.model_name);predictor.set_image(rgb)
                transformed=predictor.transform.apply_boxes_torch(box[None],rgb.shape[:2]).to(module.comfy.model_management.get_torch_device())
                with torch.inference_mode():logits,iou,_=predictor.predict_torch(point_coords=None,point_labels=None,boxes=transformed,multimask_output=False,return_logits=True)
                sam_count+=1;a=logits[0,0].detach().cpu().numpy();binary=a>float(sam_model.mask_threshold);union=int((a>-1).sum());stability=float((a>1).sum()/max(1,union))
                import cv2
                n,labels,stats,_=cv2.connectedComponentsWithStats(binary.astype(np.uint8),8)
                if n>1:
                    largest=1+int(stats[1:,cv2.CC_STAT_AREA].argmax());fraction=float(stats[largest,cv2.CC_STAT_AREA]/max(1,binary.sum()));entry['largest_component_fraction']=fraction
                    if fraction<.75:entry['reason']='fragmented_mask';continue
                    binary=labels==largest
                ok,quality=assess_mask(binary,role,person_box,head_box,float(iou[0,0]),stability);entry.update(quality)
                if not ok:continue
                # Only tiny incidental head leakage is removed; large overlap was rejected.
                if head_box is not None:
                    x0,y0,x1,y1=[int(v) for v in head_box];binary[max(0,y0):min(h,y1),max(0,x0):min(w,x1)]=False
                mask=torch.from_numpy(binary.astype(np.float32))[None];segmented=torch.from_numpy(rgb.astype(np.float32)/255.)[None]*mask[:,:,:,None]
                report.update(used_fallback=True,selected_prompt=candidate_prompt,selected_threshold=cutoff,selected_score=float(scores[idx]),selected_mask_quality=quality,selected_mask_sha256=hashlib.sha256(binary.tobytes()).hexdigest(),rho=float(mask.mean()),sam_candidates=sam_count,valid=True,elapsed_sec=time.perf_counter()-start)
                return {'ui':{'segmentation_fallback':[json.dumps(report)]},'result':(segmented,mask)}
        # Preserve every earlier success. Invoke independent recovery only after
        # the existing prompt/SAM policy has rejected all of its candidates.
        from .segmentation_rescue import recover
        binary,quality,details=recover(rgb,role,person_box,head_box,report,module,sam_model,predictor,assess_mask)
        report['rescue_attempts']=details
        if binary is not None:
            if head_box is not None:
                x0,y0,x1,y1=[int(v) for v in head_box];binary[max(0,y0):min(h,y1),max(0,x0):min(w,x1)]=False
            if binary.any():
                mask=torch.from_numpy(binary.astype(np.float32))[None];segmented=torch.from_numpy(rgb.astype(np.float32)/255.)[None]*mask[:,:,:,None]
                report.update(used_fallback=True,rescue_used=True,selected=quality['engine'],selected_mask_quality=quality,selected_mask_sha256=hashlib.sha256(binary.tobytes()).hexdigest(),rho=float(mask.mean()),sam_candidates=sam_count,valid=True,elapsed_sec=time.perf_counter()-start)
                return {'ui':{'segmentation_fallback':[json.dumps(report)]},'result':(segmented,mask)}
        report.update(valid=False,category='no_valid_clothing_fallback',sam_candidates=sam_count,elapsed_sec=time.perf_counter()-start,rho=0.)
        raise ValueError('CROP_VALIDITY_REJECTED '+json.dumps(report))
