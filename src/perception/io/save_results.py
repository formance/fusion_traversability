from pathlib import Path
import cv2

def save_bundle(root, stamp, rgb, labels, color, overlay, lidar, aligned, evaluation_mask=None):
    arrays = {'rgb': rgb, 'semantic_labels': labels, 'semantic_color': color,
              'overlays': overlay, 'lidar_projection': lidar, 'aligned_pointcloud': aligned}
    if evaluation_mask is not None:
        arrays['evaluation_mask']=evaluation_mask.astype('uint8')*255
    for folder, array in arrays.items():
        path = Path(root) / folder
        path.mkdir(parents=True, exist_ok=True)
        output = array if array.ndim == 2 else cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(path / (stamp + '.png')), output):
            raise OSError('Cannot write result: ' + str(path))
