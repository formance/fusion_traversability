"""CMNeXt DELIVER RGB-L adapter used by the Docker worker."""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F


def deliver_forward(self, x):
    # DELIVER released weights: sum modality features at each stage.
    x_in = [t.clone() for t in x]
    batch = x[0].shape[0]
    outputs = []
    for i in range(self.num_stages):
        embed = getattr(self, f'patch_embed{i+1}')
        for k in range(len(x)):
            x_in[k], height, width = embed(x_in[k])
        for block in getattr(self, f'block{i+1}'):
            x_in = block(x_in, height, width)
        norm = getattr(self, f'norm{i+1}')
        for k in range(len(x)):
            x_in[k] = norm(x_in[k]).reshape(batch,height,width,-1).permute(0,3,1,2)
        outputs.append(torch.sum(torch.stack(x_in),dim=0))
    return outputs

class ModelAdapter:
    def __init__(self, checkpoint, device='cuda', short_side=376, amp=False, model_kind='cmnext'):
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable in the inference container')
        self.short_side = int(short_side)
        if self.short_side < 128:
            raise ValueError('short_side must be at least 128')
        self.amp = bool(amp and self.device.type == 'cuda')
        # Construct and load on the requested device instead of retaining a
        # full CPU model and checkpoint while transferring them to CUDA.
        self.model_name = 'StitchFusion-B2' if model_kind == 'stitchfusion' else 'CMNeXt-B2'
        with torch.device(self.device):
            if model_kind == 'stitchfusion':
                import types
                from semseg.models.stitchfusion import stitchfusion
                self.model = stitchfusion('stitchfusion-B2', 25, ['img', 'lidar'])
                del self.model.backbone.feature_cross
                self.model.backbone.forward = types.MethodType(deliver_forward, self.model.backbone)
            else:
                from semseg.models.cmnext import CMNeXt
                self.model = CMNeXt('CMNeXt-B2', 25, ['img', 'lidar'])
        state = torch.load(Path(checkpoint), map_location=self.device, weights_only=True, mmap=True)
        for key in ('state_dict', 'model_state_dict', 'model'):
            if isinstance(state, dict) and key in state and isinstance(state[key], dict):
                state = state[key]
                break
        if not isinstance(state, dict) or not state:
            raise ValueError('Checkpoint does not contain a model state dictionary')
        state = {key.removeprefix('module.'): value for key, value in state.items()}
        self.model.load_state_dict(state, strict=True, assign=True)
        self.model.eval()
        del state
        self.execution = {'preprocess':str(self.device),'model':str(self.device),
                          'postprocess':str(self.device),'precision':'amp' if self.amp else 'float32'}
        self.mean = torch.tensor([.485,.456,.406],device=self.device)[None,:,None,None]
        self.std = torch.tensor([.229,.224,.225],device=self.device)[None,:,None,None]

    @torch.inference_mode()
    def predict(self, rgb, lidar_rgb):
        if rgb.size != lidar_rgb.size:
            raise ValueError('RGB and LiDAR images must have identical dimensions')
        arrays=[torch.as_tensor(np.array(image.convert('RGB'),copy=True),device=self.device) for image in (rgb,lidar_rgb)]
        return self.predict_tensors(*arrays)

    @torch.inference_mode()
    def predict_tensors(self,rgb,lidar_rgb):
        if rgb.shape != lidar_rgb.shape or rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=torch.uint8 or lidar_rgb.dtype!=torch.uint8:
            raise ValueError('Expected two matching uint8 HWC tensors')
        height,width = rgb.shape[:2]
        scale = self.short_side / min(height,width)
        shape = (round(height*scale), round(width*scale))
        aligned = tuple((n+31)//32*32 for n in shape)
        values=[]
        for image in (rgb,lidar_rgb):
            # Transfer compact uint8 pixels once; resize and normalize on CUDA.
            value=image.permute(2,0,1)[None].to(self.device)
            for size in (shape,aligned):
                value=F.interpolate(value.float(),size=size,mode='bilinear',align_corners=False,antialias=False).round().clamp(0,255).to(torch.uint8)
            values.append(value.to(dtype=torch.float32)/255)
        values[0]=(values[0]-self.mean)/self.std
        with torch.autocast(device_type=self.device.type,enabled=self.amp):
            logits=self.model(values)
        logits=F.interpolate(logits.float(),size=(height,width),mode='bilinear',align_corners=False)
        return logits.argmax(1)[0].to('cpu',dtype=torch.uint8).numpy(),list(aligned)
