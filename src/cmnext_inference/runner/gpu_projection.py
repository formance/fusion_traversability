"""CUDA cloud filtering, camera projection, depth encoding and preview."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

class GPUProjection:
    def __init__(self, device):
        self.device = device
        path = Path(__file__).resolve().parents[1]/'config/turbo_lut.json'
        self.lut = torch.tensor(json.loads(path.read_text()), dtype=torch.uint8, device=device)
 
    def transform(self, points, matrix):
        matrix = torch.as_tensor(matrix, dtype=torch.float32, device=self.device)
        if matrix.shape != (4,4) or not torch.isfinite(matrix).all():
            raise ValueError('Invalid transform matrix')
        return points @ matrix[:3,:3].T + matrix[:3,3]

    def depth(self, points, k, width, height, maximum):
        points = points[torch.isfinite(points).all(1) & (points[:,2] > .05) & (points[:,2] <= maximum)]
        if not len(points):
            raise ValueError('No valid points in camera range')
        uvz = points @ k.T
        uv = (uvz[:,:2]/uvz[:,2,None]).round().long()
        keep = (uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
        if not keep.any():
            raise ValueError('No projected points inside image')
        uv = uv[keep]
        depth = torch.full((height*width,), float('inf'), device=self.device)
        depth.scatter_reduce_(0, uv[:,1]*width+uv[:,0], points[keep,2], reduce='amin', include_self=True)
        return depth.reshape(height,width)

    def color_depth(self, depth, maximum, dilation, grayscale=False):
        inverse = torch.where(torch.isfinite(depth), depth.reciprocal(), 0.)
        if dilation > 1:
            inverse = F.max_pool2d(inverse[None,None], dilation, stride=1, padding=dilation//2)[0,0]
        valid = inverse > 0
        index = torch.zeros(depth.shape, dtype=torch.uint8, device=self.device)
        index[valid] = (255*(1-torch.log1p(inverse[valid].reciprocal())/np.log1p(maximum)).clamp(0,1)).round().to(torch.uint8)
        if grayscale:
            return index.unsqueeze(-1).expand(-1,-1,3).contiguous(), valid
        color = self.lut[index.long()]
        color[~valid] = 0
        return color, valid

    @torch.inference_mode()
    def process(self, meta, arrays):
        rgb_array, xyz_array = arrays['rgb'], arrays['xyz']
        if rgb_array.dtype != np.uint8 or rgb_array.ndim != 3 or rgb_array.shape[2] != 3 or rgb_array.shape[0]*rgb_array.shape[1]>8_000_000:
            raise ValueError('Invalid RGB image')
        if xyz_array.dtype != np.float32 or xyz_array.ndim != 2 or xyz_array.shape[1] != 3 or len(xyz_array)>2_000_000:
            raise ValueError('Invalid XYZ scan')
        height,width = rgb_array.shape[:2]
        rgb = torch.as_tensor(np.array(rgb_array,copy=True),device=self.device)
        points = torch.as_tensor(np.array(xyz_array,copy=True),device=self.device)
        points = points[torch.isfinite(points).all(1)]
        base = self.transform(points, meta['base_from_lidar'])
        inside = torch.zeros(len(points), dtype=torch.bool, device=self.device)
        for box in meta['ego_boxes']:
            bounds = torch.as_tensor(box['bounds_xyz'],dtype=torch.float32,device=self.device)
            if bounds.shape != (3,2):raise ValueError('Invalid ego bounds')
            inside |= ((base>=bounds[:,0])&(base<=bounds[:,1])).all(1)
        camera = self.transform(points[~inside],meta['camera_from_lidar'])
        k = torch.as_tensor(meta['intrinsic'],dtype=torch.float32,device=self.device).reshape(3,3)
        if not torch.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0:raise ValueError('Invalid intrinsics')
        maximum = float(meta['maximum'])
        dilation = int(meta['dilation'])
        if not maximum>0 or dilation not in (1,3,5,7,9):raise ValueError('Invalid depth encoding settings')
        depth = self.depth(camera,k,width,height,maximum)
        # Experimental grayscale: preserve the former depth mapping and support.
        # DELIVER's exact point-to-gray mapping remains unpublished.
        lidar, encoded_valid = self.color_depth(depth,maximum,dilation)
        # Match CPU crop_points: test both floating and rounded pixel bounds.
        minimum,limit = meta['aligned_depth_range']
        selected = camera[(camera[:,2]>=minimum)&(camera[:,2]<=limit)]
        projected = selected @ k.T
        uv = projected[:,:2]/projected[:,2,None]
        pixel = uv.round()
        x,y,w,h = meta['roi']
        keep=(uv[:,0]>=x)&(uv[:,0]<x+w)&(uv[:,1]>=y)&(uv[:,1]<y+h)&(pixel[:,0]>=x)&(pixel[:,0]<x+w)&(pixel[:,1]>=y)&(pixel[:,1]<y+h)
        selected = selected[keep]
        if not len(selected):raise ValueError('No aligned camera points')
        outputs = {'observed':torch.isfinite(depth).to(torch.uint8)}
        if meta.get('need_lidar',True):outputs['lidar']=lidar
        if meta.get('need_points',False):outputs['points']=selected
        if meta.get('need_aligned',False):
            display_depth = self.depth(selected,k,width,height,maximum)
            colors,visible = self.color_depth(display_depth,maximum,3)
            aligned = (rgb.float()*.65).round().to(torch.uint8)
            aligned[visible]=colors[visible]
            outputs['aligned']=aligned
        stats={'projected_pixels':int(torch.isfinite(depth).sum()),'encoded_pixels':int(encoded_valid.sum()),'aligned_points':len(selected),'projection_device':str(self.device)}
        return rgb,lidar,outputs,stats
