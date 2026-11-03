"""Validate capture records and read lossless semantic IDs one sample at a time."""
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image

from perception.io.label_io import load_ground_truth
from .regions import REGIONS


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name}: expected a finite number')
    return value


class CaptureDataset:
    def __init__(self, directory, allow_partial=False):
        self.directory = Path(directory).resolve()
        self.metadata = json.loads((self.directory/'metadata.json').read_text())
        self.result = json.loads((self.directory/'result.json').read_text())
        self.classes = self.metadata['classes']
        if not isinstance(self.classes, list) or len(self.classes) != 11 or not all(isinstance(c, str) and c for c in self.classes):
            raise ValueError('Expected 11 engineering class names')
        mapping = self.metadata['prediction_source_to_target']
        if not isinstance(mapping, list) or len(mapping) != 25 or any(type(i) is not int or not 0 <= i < 11 for i in mapping):
            raise ValueError('Expected 25 source-to-target IDs in 0..10')
        self.mapping = np.array(mapping, dtype=np.uint8)
        encoding = self.metadata.get('lidar_encoding', {})
        if encoding.get('name', 'projected_camera_log_depth_turbo') != 'projected_camera_log_depth_turbo':
            raise ValueError('Unsupported LiDAR validity encoding')
        missing = encoding.get('missing_pixel_rgb', [0, 0, 0])
        if not isinstance(missing, list) or len(missing) != 3 or any(type(c) is not int or not 0 <= c <= 255 for c in missing):
            raise ValueError('Invalid LiDAR missing-pixel RGB value')
        self.lidar_missing_rgb = np.array(missing, dtype=np.uint8)
        self.width, self.height = self.metadata['image_width'], self.metadata['image_height']
        if any(type(n) is not int or n <= 0 for n in (self.width, self.height)) or self.width*self.height > 8_000_000:
            raise ValueError('Invalid capture dimensions')
        if self.result.get('complete') is not True and not allow_partial:
            raise ValueError('Dataset is incomplete; use --allow-partial to evaluate committed samples explicitly')
        self.points = self.result['points']
        if not isinstance(self.points, list) or not self.points:
            raise ValueError('Dataset has no committed samples')
        with Image.open(self.directory/'evaluation_mask.png') as image:
            if image.mode != 'L' or image.size != (self.width, self.height):
                raise ValueError('Evaluation mask must be native-size grayscale PNG')
            mask = np.array(image)
        if not np.isin(mask, [0, 255]).all() or not mask.any():
            raise ValueError('Evaluation mask must contain nonempty binary 0/255 pixels')
        self.mask = mask > 0
        tolerance = self.metadata['pose_tolerance']
        for key in ('position_m', 'heading_deg', 'rgb_lidar_delta_ms'):
            if finite(tolerance[key], key) <= 0:
                raise ValueError('Pose/time tolerances must be positive')
        ids = set()
        folders = set()
        self.samples = []
        for point in self.points:
            ident = point['id']
            if not isinstance(ident, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', ident) or ident in ids:
                raise ValueError('Invalid or duplicate sample ID')
            ids.add(ident)
            folder = (self.directory/point['sample_dir']).resolve()
            if not folder.is_relative_to(self.directory/'samples') or folder in folders:
                raise ValueError(f'{ident}: invalid or duplicate sample directory')
            folders.add(folder)
            if point.get('pose_verified') is not True:
                raise ValueError(f'{ident}: pose was not verified')
            rgb_ns, lidar_ns = point['image_timestamp_ns'], point['lidar_timestamp_ns']
            if any(type(n) is not int or n < 0 for n in (rgb_ns, lidar_ns)):
                raise ValueError(f'{ident}: invalid timestamps')
            delta = abs(rgb_ns-lidar_ns)/1e6
            if delta > tolerance['rgb_lidar_delta_ms'] or abs(finite(point['sensor_delta_ms'], 'sensor_delta_ms')-delta) > 1e-6:
                raise ValueError(f'{ident}: RGB/LiDAR timestamp mismatch')
            for key, limit in (('position_error_max_m', 'position_m'), ('heading_error_max_deg', 'heading_deg')):
                value = finite(point[key], key)
                if not 0 <= value <= tolerance[limit]:
                    raise ValueError(f'{ident}: {key} outside tolerance')
            actual, target = point['actual_pose'], point['target_pose']
            x, y, yaw = (finite(actual[k], k) for k in ('x', 'y', 'yaw'))
            tx, ty, tyaw = (finite(target[k], k) for k in ('x_m', 'y_m', 'yaw_deg'))
            heading = abs(math.degrees(math.atan2(math.sin(yaw-math.radians(tyaw)), math.cos(yaw-math.radians(tyaw)))))
            if math.hypot(x-tx, y-ty) > point['position_error_max_m']+1e-6 or heading > point['heading_error_max_deg']+1e-6:
                raise ValueError(f'{ident}: recorded pose errors disagree with actual pose')
            after_ns = actual['timestamp_ns']
            if type(after_ns) is not int or after_ns < max(rgb_ns, lidar_ns):
                raise ValueError(f'{ident}: sensor timestamp falls after pose verification')
            self.samples.append((ident, folder))
        actual_folders = {p.resolve() for p in (self.directory/'samples').glob('point_*') if p.is_dir()}
        if actual_folders != folders:
            raise ValueError('Sample directories disagree with result.json')

    def read(self, folder):
        images = []
        for filename in ('rgb.png', 'lidar_encoding.png'):
            with Image.open(folder/filename) as image:
                if image.size != (self.width, self.height) or image.mode != 'RGB':
                    raise ValueError(f'{folder.name}/{filename}: expected native-size RGB image')
                images.append(image.copy())
        labels = load_ground_truth(folder/'ground_truth.png')
        if labels.shape != (self.height, self.width):
            raise ValueError(f'{folder.name}: ground truth dimensions disagree')
        lidar_valid = np.any(np.asarray(images[1]) != self.lidar_missing_rgb, axis=2)
        valid_gt = labels != 255
        regions = {'lidar_valid': lidar_valid & valid_gt, 'evaluation_mask': self.mask & valid_gt}
        for name, valid in regions.items():
            if not valid.any():
                raise ValueError(f'{folder.name}: no valid evaluation pixels in {name}')
        return images, labels, regions

    def check(self):
        counts = {name: np.zeros(11, dtype=np.int64) for name in REGIONS}
        for _, folder in self.samples:
            _, labels, regions = self.read(folder)
            for name, valid in regions.items():
                counts[name] += np.bincount(labels[valid], minlength=11)
        return {'samples': len(self.samples), 'declared_complete': self.result.get('complete') is True,
                'image_wh': [self.width, self.height],
                'regions': {name: {'valid_pixels': int(value.sum()), 'ground_truth_pixels_per_class': value.tolist()}
                            for name, value in counts.items()},
                'validity': 'Two independent regions: encoded LiDAR support AND GT != 255; stored evaluation mask AND GT != 255. No intersection between regions. Encoded support includes capture-time dilation.',
                'timestamp_check': 'Recorded RGB/LiDAR timestamps checked; a separate GT timestamp is not stored'}
