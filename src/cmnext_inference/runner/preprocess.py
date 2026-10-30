"""Point-cloud projection and explicitly experimental LiDAR image encoding."""
import cv2
import numpy as np

def transform_points(points, transform):
    q = transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    norm = np.sqrt(x*x+y*y+z*z+w*w)
    if norm < 1e-8:
        raise ValueError('Invalid TF quaternion')
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    rotation = np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                         [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                         [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]], dtype=np.float32)
    t = transform.translation
    return points @ rotation.T + np.array([t.x, t.y, t.z], dtype=np.float32)

def remove_ego_points(points, base_points, boxes):
    """Remove configurable base-frame self geometry without assuming sensor poses."""
    if points.shape != base_points.shape:
        raise ValueError('Sensor and base-frame cloud shapes differ')
    inside = np.zeros(len(points), dtype=bool)
    for box in boxes:
        bounds = np.asarray(box['bounds_xyz'], dtype=np.float32)
        if bounds.shape != (3, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError('Invalid ego-filter bounds: '+box.get('name', '?'))
        inside |= ((base_points >= bounds[:, 0]) & (base_points <= bounds[:, 1])).all(axis=1)
    return points[~inside]


def encode_projected_depth(camera_points, intrinsic, width, height, maximum=120., dilation=3):
    k = np.asarray(intrinsic, dtype=np.float32).reshape(3, 3)
    if not np.isfinite(k).all() or k[0,0] <= 0 or k[1,1] <= 0:
        raise ValueError('Invalid CameraInfo intrinsics')
    points = camera_points[np.isfinite(camera_points).all(1)]
    points = points[(points[:,2] > .05) & (points[:,2] <= maximum)]
    if not len(points):
        raise ValueError('No LiDAR points in front of the camera')
    projected = points @ k.T
    uv = np.rint(projected[:,:2] / projected[:,2,None]).astype(np.int32)
    inside = (uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
    uv, depths = uv[inside], points[inside,2]
    if not len(depths):
        raise ValueError('LiDAR projection has no points inside the image')
    flat = uv[:,1]*width + uv[:,0]
    depth = np.full(height*width, np.inf, dtype=np.float32)
    np.minimum.at(depth, flat, depths)
    depth = depth.reshape(height,width)
    valid = np.isfinite(depth)
    raw_count = int(valid.sum())
    inverse = np.zeros_like(depth)
    inverse[valid] = 1. / depth[valid]
    if dilation > 1:
        if dilation % 2 == 0 or dilation > 9:
            raise ValueError('Dilation must be an odd integer no greater than 9')
        inverse = cv2.dilate(inverse, np.ones((dilation,dilation), np.uint8))
    valid = inverse > 0
    filled_depth = np.zeros_like(depth)
    filled_depth[valid] = 1. / inverse[valid]
    index = np.zeros((height,width), np.uint8)
    index[valid] = np.rint(255*np.clip(1-np.log1p(filled_depth[valid])/np.log1p(maximum),0,1)).astype(np.uint8)
    rgb = cv2.cvtColor(cv2.applyColorMap(index, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    rgb[~valid] = 0
    return rgb, depth, {'projected_pixels': raw_count, 'encoded_pixels': int(valid.sum())}


def render_aligned_cloud(rgb, depth, maximum=120., marker_radius=1):
    """Display measured LiDAR returns projected onto this RGB frame, colored by z.

    Raw projection uses one nearest measured return per pixel. A 3x3 marker only
    improves point visibility in the viewer; it is not dense scene depth.
    """
    valid = np.isfinite(depth)
    inverse = np.zeros_like(depth)
    inverse[valid] = 1. / depth[valid]
    if marker_radius:
        size=2*marker_radius+1
        inverse=cv2.dilate(inverse,np.ones((size,size),np.uint8))
    visible=inverse>0
    index=np.zeros(depth.shape,np.uint8)
    z=1./inverse[visible]
    index[visible]=np.rint(255*np.clip(1-np.log1p(z)/np.log1p(maximum),0,1)).astype(np.uint8)
    colors=cv2.cvtColor(cv2.applyColorMap(index,cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
    aligned=np.rint(rgb*.65).astype(np.uint8)
    aligned[visible]=colors[visible]
    return aligned


def near_ground_blind_mask(intrinsic, width, height, camera_to_base, lidar_settings, settings):
    """Flat-ground ray intersections below the near LiDAR elevation/range envelope.

    This geometric mask is cached by the caller. It is a prior, never a return.
    Horizontal out-of-FOV and distant missing data are deliberately excluded.
    """
    if not settings.get('enabled', False):
        return np.zeros((height, width), dtype=bool)
    k = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    origin = transform_points(np.zeros((1, 3), np.float32), camera_to_base)[0]
    basis = transform_points(np.eye(3, dtype=np.float32), camera_to_base) - origin
    x = (np.arange(width, dtype=np.float32) - k[0, 2]) / k[0, 0]
    y = (np.arange(height, dtype=np.float32) - k[1, 2]) / k[1, 1]
    directions = x[None, :, None]*basis[0] + y[:, None, None]*basis[1] + basis[2]
    plane = float(settings.get('ground_plane_z_base_m', 0.))
    t = np.divide(plane-origin[2], directions[:, :, 2],
                  out=np.zeros((height, width), np.float32), where=directions[:, :, 2] < -1e-6)
    ground = origin + t[:, :, None]*directions
    offset = ground - np.asarray(lidar_settings['pose_xyz'], np.float32)
    roll, pitch, yaw = lidar_settings['pose_rpy']
    cr,sr,cp,sp,cy,sy = np.cos(roll),np.sin(roll),np.cos(pitch),np.sin(pitch),np.cos(yaw),np.sin(yaw)
    rotation = np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                         [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]], np.float32)
    local = offset @ rotation
    horizontal = np.hypot(local[:, :, 0], local[:, :, 1])
    elevation = np.arctan2(local[:, :, 2], horizontal)
    azimuth = np.arctan2(local[:, :, 1], local[:, :, 0])
    below = elevation < float(lidar_settings['vertical_min_rad'])
    too_close = np.linalg.norm(local, axis=2) < float(lidar_settings['simulation_range_min_m'])
    in_front = local[:, :, 0] > 0
    inside_horizontal = np.abs(azimuth) <= np.deg2rad(float(lidar_settings['horizontal_fov_deg'])/2)
    bounded = np.hypot(offset[:, :, 0], offset[:, :, 1]) <= float(settings.get('max_ground_radius_m', 2.))
    return (t>0) & in_front & inside_horizontal & bounded & (below | too_close)


def apply_ground_prior(labels, geometric_mask, measured_depth, settings):
    """Fill the selected blind region; preserve actual LiDAR return neighborhoods."""
    applied = geometric_mask.copy()
    if settings.get('preserve_measured_points', True):
        observed = np.isfinite(measured_depth).astype(np.uint8)
        margin = max(0, int(settings.get('observation_margin_pixels', 3)))
        if margin:
            observed = cv2.dilate(observed, np.ones((2*margin+1, 2*margin+1), np.uint8))
        applied &= observed == 0
    result = labels.copy()
    result[applied] = int(settings.get('ground_label_id', 0))
    return result, applied


def front_ground_region_mask(intrinsic,width,height,camera_to_base,region,plane_z=0.):
    """Nominal front ROI projected onto the assumed flat ground plane."""
    k=np.asarray(intrinsic,dtype=np.float64).reshape(3,3)
    origin=transform_points(np.zeros((1,3),np.float32),camera_to_base)[0]
    basis=transform_points(np.eye(3,dtype=np.float32),camera_to_base)-origin
    u=(np.arange(width,dtype=np.float32)-k[0,2])/k[0,0]
    v=(np.arange(height,dtype=np.float32)-k[1,2])/k[1,1]
    directions=u[None,:,None]*basis[0]+v[:,None,None]*basis[1]+basis[2]
    t=np.divide(plane_z-origin[2],directions[:,:,2],out=np.zeros((height,width),np.float32),where=directions[:,:,2]<-1e-6)
    ground=origin+t[:,:,None]*directions
    footprint=ground-np.asarray(region['reference_pose_in_base_xyz'],np.float32)
    xmin,xmax=region['x_range_m'];ymin,ymax=region['y_range_m']
    return (t>0)&(footprint[:,:,0]>=xmin)&(footprint[:,:,0]<=xmax)&(footprint[:,:,1]>=ymin)&(footprint[:,:,1]<=ymax)


def render_detection_overlay(rgb,labels,palette,effective,blind):
    """Blend classes in effective ROI and mark blind pixels; preserve all others."""
    if rgb.shape[:2]!=labels.shape or effective.shape!=labels.shape or blind.shape!=labels.shape:
        raise ValueError('Detection overlay shape mismatch')
    out=rgb.copy();valid=effective & ~blind
    out[valid]=np.rint(.55*rgb[valid]+.45*palette[labels[valid]]).astype(np.uint8)
    out[blind]=np.rint(.65*rgb[blind]+.35*np.array([255,175,55],dtype=np.float32)).astype(np.uint8)
    return out


def effective_ground_evaluation_mask(intrinsic,width,height,camera_to_base,lidar_settings,region,plane_z=0.):
    """Visible ground-plane footprint inside both sensor envelopes and front ROI.

    This is geometric coverage, independent of point-return sparsity and labels.
    It excludes the physical near-ground LiDAR blind region, never filling it.
    """
    k=np.asarray(intrinsic,dtype=np.float64).reshape(3,3)
    origin=transform_points(np.zeros((1,3),np.float32),camera_to_base)[0]
    basis=transform_points(np.eye(3,dtype=np.float32),camera_to_base)-origin
    u=(np.arange(width,dtype=np.float32)-k[0,2])/k[0,0]
    v=(np.arange(height,dtype=np.float32)-k[1,2])/k[1,1]
    directions=u[None,:,None]*basis[0]+v[:,None,None]*basis[1]+basis[2]
    t=np.divide(plane_z-origin[2],directions[:,:,2],out=np.zeros((height,width),np.float32),where=directions[:,:,2]<-1e-6)
    ground=origin+t[:,:,None]*directions
    reference=np.asarray(region['reference_pose_in_base_xyz'],np.float32)
    footprint=ground-reference
    xmin,xmax=region['x_range_m'];ymin,ymax=region['y_range_m']
    inside_roi=(footprint[:,:,0]>=xmin)&(footprint[:,:,0]<=xmax)&(footprint[:,:,1]>=ymin)&(footprint[:,:,1]<=ymax)
    offset=ground-np.asarray(lidar_settings['pose_xyz'],np.float32)
    roll,pitch,yaw=lidar_settings['pose_rpy'];cr,sr,cp,sp,cy,sy=np.cos(roll),np.sin(roll),np.cos(pitch),np.sin(pitch),np.cos(yaw),np.sin(yaw)
    rotation=np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],[sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]],np.float32)
    local=offset@rotation;horizontal=np.hypot(local[:,:,0],local[:,:,1]);azimuth=np.arctan2(local[:,:,1],local[:,:,0]);elevation=np.arctan2(local[:,:,2],horizontal);distance=np.linalg.norm(local,axis=2)
    hmin=float(lidar_settings.get('simulation_horizontal_min_rad',-np.deg2rad(lidar_settings['horizontal_fov_deg']/2)))
    hmax=float(lidar_settings.get('simulation_horizontal_max_rad',np.deg2rad(lidar_settings['horizontal_fov_deg']/2)))
    return (t>0)&inside_roi&(local[:,:,0]>0)&(azimuth>=hmin)&(azimuth<=hmax)&(elevation>=lidar_settings['vertical_min_rad'])&(elevation<=lidar_settings['vertical_max_rad'])&(distance>=lidar_settings['simulation_range_min_m'])&(distance<=lidar_settings['simulation_range_max_m'])
