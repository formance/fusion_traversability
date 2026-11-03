"""Independent evaluation scopes and a provenance-tracked front blind region."""
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

REGIONS = ('lidar_valid', 'evaluation_mask')
REGION_NAMES = {'lidar_valid': '雷达编码非黑色有效区', 'evaluation_mask': '原评测掩码区'}
REGION_DIRECTORIES = {'lidar_valid': 'evaluation_valid', 'evaluation_mask': 'evaluation_mask'}


def continuous_lidar_coverage(support):
    """Display envelope between per-column upper/lower encoded-return bounds.

    Fill sampling gaps using the model's dense predictions, not interpolated
    point labels. Missing interior columns interpolate their bounds; no
    horizontal extrapolation is made beyond columns containing returns.
    This estimated display envelope does not change the scoring mask.
    """
    support = np.asarray(support)
    if support.ndim != 2 or support.dtype != np.bool_ or not support.any():
        raise ValueError('Expected nonempty boolean LiDAR support')
    height, width = support.shape
    columns = np.flatnonzero(support.any(axis=0))
    upper = np.argmax(support, axis=0)
    lower = height-1-np.argmax(support[::-1], axis=0)
    x = np.arange(width)
    top = np.floor(np.interp(x, columns, upper[columns]))
    bottom = np.ceil(np.interp(x, columns, lower[columns]))
    y = np.arange(height)[:, None]
    return (y >= top) & (y <= bottom) & (x >= columns[0]) & (x <= columns[-1])


def rotation(rpy):
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]], dtype=np.float32)


def geometric_front_blind(camera, lidar, settings, width, height):
    """Same flat-ground near-field construction as the live/capture geometry."""
    if not settings.get('enabled', False):
        return np.zeros((height, width), dtype=bool)
    k = camera['simulation_intrinsics']
    origin = np.asarray(camera['pose_xyz'], dtype=np.float32)
    basis = rotation(camera['pose_rpy']) @ rotation(camera['optical_frame_rpy'])
    x = (np.arange(width, dtype=np.float32)-k['cx'])/k['fx']
    y = (np.arange(height, dtype=np.float32)-k['cy'])/k['fy']
    directions = x[None, :, None]*basis[:, 0]+y[:, None, None]*basis[:, 1]+basis[:, 2]
    t = np.divide(float(settings.get('ground_plane_z_base_m', 0))-origin[2], directions[:, :, 2],
                  out=np.zeros((height, width), dtype=np.float32), where=directions[:, :, 2] < -1e-6)
    ground = origin+t[:, :, None]*directions
    offset = ground-np.asarray(lidar['pose_xyz'], dtype=np.float32)
    local = offset @ rotation(lidar['pose_rpy'])
    elevation = np.arctan2(local[:, :, 2], np.hypot(local[:, :, 0], local[:, :, 1]))
    azimuth = np.arctan2(local[:, :, 1], local[:, :, 0])
    below = elevation < lidar['vertical_min_rad']
    close = np.linalg.norm(local, axis=2) < lidar['simulation_range_min_m']
    bounded = np.hypot(offset[:, :, 0], offset[:, :, 1]) <= settings.get('max_ground_radius_m', 2.)
    return (t > 0) & (local[:, :, 0] > 0) & (np.abs(azimuth) <= np.deg2rad(lidar['horizontal_fov_deg']/2)) & bounded & (below | close)


def _captured_geometry(world_path, camera, lidar):
    robot = ET.parse(world_path).getroot().find("world/model[@name='factory_robot']")
    if robot is None:
        raise ValueError('Capture world has no factory_robot geometry')
    camera_link = next(link for link in robot.findall('link') if link.find("sensor[@type='camera']") is not None)
    lidar_link = next(link for link in robot.findall('link') if link.find("sensor[@type='gpu_lidar']") is not None)
    for link, config in ((camera_link, camera), (lidar_link, lidar)):
        pose = list(map(float, link.findtext('pose').split()))
        config['pose_xyz'], config['pose_rpy'] = pose[:3], pose[3:]
    sensor = camera_link.find("sensor[@type='camera']/camera")
    camera['width'], camera['height'] = int(sensor.findtext('image/width')), int(sensor.findtext('image/height'))
    intrinsics = sensor.find('lens/intrinsics')
    camera['simulation_intrinsics'] = {key: float(intrinsics.findtext(key)) for key in ('fx', 'fy', 'cx', 'cy')}
    scan = lidar_link.find("sensor[@type='gpu_lidar']/lidar")
    lidar['vertical_min_rad'] = float(scan.findtext('scan/vertical/min_angle'))
    lidar['simulation_range_min_m'] = float(scan.findtext('range/min'))
    angles = [float(scan.findtext('scan/horizontal/'+key)) for key in ('min_angle', 'max_angle')]
    lidar['horizontal_fov_deg'] = float(np.rad2deg(2*max(map(abs, angles))))


def front_blind_mask(dataset, project, override=None):
    if override:
        path, source = Path(override).resolve(), 'explicit_front_blind_mask'
    else:
        filename = dataset.metadata.get('front_blind_mask_file', 'front_blind_mask.png')
        path = (dataset.directory/filename).resolve()
        if not path.is_relative_to(dataset.directory):
            raise ValueError('Front blind mask must be inside the capture directory')
        source = 'captured_front_blind_mask'
    if path.is_file():
        with Image.open(path) as image:
            pixels = np.array(image)
            if image.mode != 'L' or pixels.shape != (dataset.height, dataset.width) or not np.isin(pixels, [0, 255]).all():
                raise ValueError('Front blind mask must be a native-size binary grayscale image')
        return pixels > 0, {'source': source, 'path': str(path)}
    if override or 'front_blind_mask_file' in dataset.metadata:
        raise ValueError('Recorded/explicit front blind mask is missing: '+str(path))
    sensors_path = project/'src/factory_sim/config/sensors.json'
    sensors = json.loads(sensors_path.read_text())
    policy_path = project/'src/perception/config/inference.json'
    settings = json.loads(policy_path.read_text())['ground_prior']
    camera, lidar = sensors['camera'], sensors['lidar']
    info = {'source': 'current_sensor_geometry_fallback', 'geometry': str(sensors_path),
            'policy': settings, 'note': 'Flat-ground geometric display region; not a measured return or semantic label'}
    states = [Path.home()/'.local/state/fusion_traversability']
    if os.environ.get('FUSION_STATE_DIR'):
        states.insert(0, Path(os.environ['FUSION_STATE_DIR']))
    world = next((state/'captures'/dataset.directory.name/'evaluation.sdf' for state in states
                  if (state/'captures'/dataset.directory.name/'evaluation.sdf').is_file()), None)
    if world is not None:
        _captured_geometry(world, camera, lidar)
        info.update(source='capture_world_geometry_with_current_blind_policy', geometry=str(world))
    if (camera['width'], camera['height']) != (dataset.width, dataset.height):
        raise ValueError('Camera geometry dimensions differ; provide --front-blind-mask')
    mask = geometric_front_blind(camera, lidar, settings, dataset.width, dataset.height)
    info['pixels'] = int(mask.sum())
    return mask, info
