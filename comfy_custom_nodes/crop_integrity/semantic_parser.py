"""Image-only SCHP fallback. No dataset parsing maps or evaluation masks are read."""
import os
import threading
from pathlib import Path
import cv2
import numpy as np
import torch
from .vendor.schp.architecture import resnet101
from .vendor.schp.transforms import get_affine_transform,transform_logits

_MODELS={}
_LOCK=threading.RLock()
_SPECS={'atr':('exp-schp-201908301523-atr.pth',18,512,[4,7]),
        'lip':('exp-schp-201908261155-lip.pth',20,473,[5,6,7])}

def model_path(kind):
    import folder_paths
    root=Path(os.getenv('CLOSET_HUMAN_PARSER_DIR',str(Path(folder_paths.models_dir)/'human_parsing')))
    return root/_SPECS[kind][0]

def load_model(kind):
    with _LOCK:
        if kind not in _MODELS:
            path=model_path(kind)
            if not path.is_file():raise FileNotFoundError(f'Clothing parser checkpoint missing: {path}')
            model=resnet101(num_classes=_SPECS[kind][1],pretrained=None)
            state=torch.load(path,map_location='cpu',weights_only=False)['state_dict']
            rename={'decoder.conv3.2.weight':'decoder.conv3.3.weight','decoder.conv3.3.weight':'decoder.conv3.4.weight','decoder.conv3.3.bias':'decoder.conv3.4.bias','decoder.conv3.3.running_mean':'decoder.conv3.4.running_mean','decoder.conv3.3.running_var':'decoder.conv3.4.running_var','fushion.3.weight':'fushion.4.weight','fushion.3.bias':'fushion.4.bias'}
            state={rename.get(k.removeprefix('module.'),k.removeprefix('module.')):v for k,v in state.items()}
            # CatVTON's inference architecture omits these auxiliary training heads.
            for key in ['edge.conv5.weight','edge.conv5.bias','decoder.conv4.weight','decoder.conv4.bias']:state.pop(key,None)
            model.load_state_dict(state,strict=True)
            _MODELS[kind]=model.eval().requires_grad_(False)
        return _MODELS[kind]

def parse_image(rgb,kind='atr'):
    """Official affine/BGR preprocessing; returns newly inferred class IDs/logits."""
    model=load_model(kind);h,w=rgb.shape[:2];size=_SPECS[kind][2]
    center=np.array([(w-1)/2,(h-1)/2],np.float32);scale=np.array([max(w-1,h-1)]*2,np.float32)
    affine=get_affine_transform(center,scale,0,[size,size])
    bgr=cv2.warpAffine(rgb[:,:,::-1],affine,(size,size),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
    a=bgr.astype(np.float32)/255.;a=(a-np.array([.406,.456,.485],np.float32))/np.array([.225,.224,.229],np.float32)
    tensor=torch.from_numpy(a.transpose(2,0,1).copy())[None]
    with _LOCK,torch.inference_mode():
        previous_threads=torch.get_num_threads()
        try:
            if previous_threads>4:torch.set_num_threads(4)
            logits=model(tensor);logits=torch.nn.functional.interpolate(logits,size=(size,size),mode='bilinear',align_corners=True)[0].permute(1,2,0).numpy()
        finally:
            if torch.get_num_threads()!=previous_threads:torch.set_num_threads(previous_threads)
    full=transform_logits(logits,center,scale,w,h,input_size=[size,size]);labels=full.argmax(axis=2)
    exp=np.exp(full-full.max(axis=2,keepdims=True));confidence=exp.max(axis=2)/exp.sum(axis=2)
    return labels,confidence

def clothing_candidate(rgb,kind='atr',check_flip=True):
    labels,confidence=parse_image(rgb,kind);mask=np.isin(labels,_SPECS[kind][3])
    info={'engine':'schp_'+kind,'clothing_labels':_SPECS[kind][3],'mean_class_confidence':float(confidence[mask].mean()) if mask.any() else 0.,'rho':float(mask.mean()),'checkpoint':str(model_path(kind))}
    if check_flip:
        flipped,_=parse_image(rgb[:,::-1].copy(),kind);other=np.isin(flipped[:,::-1],_SPECS[kind][3]);info['flip_iou']=float((mask&other).sum()/max(1,(mask|other).sum()))
    return mask,labels,info
