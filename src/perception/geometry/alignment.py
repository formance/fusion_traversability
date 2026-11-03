"""Calibrated common-view output crop; points remain in camera optical coordinates."""
import copy
import numpy as np
from perception.geometry.preprocess import transform_points

def common_view_roi(intrinsic, width, height, camera_from_lidar, settings, lidar_settings):
    if not settings.get('enabled',True):return (0,0,width,height)
    if lidar_settings['horizontal_fov_deg']<359.9:
        raise ValueError('Automatic common-view crop currently requires a 360-degree LiDAR')
    minimum,maximum=settings['camera_depth_range_m']
    if minimum<=0 or maximum<=minimum:raise ValueError('Invalid common-view depth range')
    k=np.asarray(intrinsic,dtype=np.float64).reshape(3,3)
    origin=transform_points(np.zeros((1,3),np.float32),camera_from_lidar)[0]
    basis=transform_points(np.eye(3,dtype=np.float32),camera_from_lidar)-origin
    depths=np.geomspace(minimum,maximum,24)[:,None,None]
    columns=np.linspace(0,width-1,33)[None,None,:]
    rows=np.arange(height)[None,:,None]
    z=np.broadcast_to(depths,(24,height,33))
    rays=np.stack([np.broadcast_to((columns-k[0,2])/k[0,0]*z,z.shape),
                   np.broadcast_to((rows-k[1,2])/k[1,1]*z,z.shape),z],axis=-1)
    lidar=(rays-origin)@basis.T
    elevation=np.arctan2(lidar[...,2],np.hypot(lidar[...,0],lidar[...,1]))
    valid=((elevation>=lidar_settings['vertical_min_rad']) &
           (elevation<=lidar_settings['vertical_max_rad'])).all(axis=(0,2))
    # Choose one contiguous row interval that fits every sampled depth and column.
    indices=np.flatnonzero(valid)
    if not len(indices):raise ValueError('No common camera/LiDAR view in the configured depth range')
    groups=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
    band=max(groups,key=len)
    margin=int(settings.get('margin_pixels',2))
    x0,y0=margin,int(band[0])+margin
    x1,y1=width-margin,int(band[-1])+1-margin
    if x1<=x0 or y1<=y0:raise ValueError('Common-view crop is empty')
    return x0,y0,x1-x0,y1-y0

def crop_camera_info(info, roi, header):
    x,y,width,height=roi
    cropped=copy.deepcopy(info)
    cropped.header=header;cropped.width=width;cropped.height=height
    k=list(cropped.k);k[2]-=x;k[5]-=y;cropped.k=k
    projection=list(cropped.p);projection[2]-=x;projection[6]-=y;cropped.p=projection
    # K/P now describe the cropped image itself; don't apply the offset twice.
    cropped.roi.x_offset=0;cropped.roi.y_offset=0
    cropped.roi.width=width;cropped.roi.height=height;cropped.roi.do_rectify=False
    cropped.binning_x=0;cropped.binning_y=0
    return cropped

def crop_points(points, intrinsic, roi, settings):
    points=points[np.isfinite(points).all(axis=1)]
    minimum,maximum=settings['camera_depth_range_m'] if settings.get('enabled',True) else (.05,120.)
    points=points[(points[:,2]>=minimum)&(points[:,2]<=maximum)]
    if not len(points):raise ValueError('No points in common-view depth range')
    k=np.asarray(intrinsic,dtype=np.float64).reshape(3,3)
    uvz=points@k.T;uv=uvz[:,:2]/uvz[:,2,None]
    pixel=np.rint(uv).astype(np.int64)
    x,y,width,height=roi
    keep=(uv[:,0]>=x)&(uv[:,0]<x+width)&(uv[:,1]>=y)&(uv[:,1]<y+height)
    keep &= (pixel[:,0]>=x)&(pixel[:,0]<x+width)&(pixel[:,1]>=y)&(pixel[:,1]<y+height)
    return points[keep].astype(np.float32,copy=False)
