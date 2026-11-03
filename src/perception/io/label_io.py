"""Lossless semantic IDs with display colors and no text legend."""
from pathlib import Path
import numpy as np
from PIL import Image, PngImagePlugin

def save_ground_truth(path, labels, classes):
    """Keep semantic indices unchanged, without appending a legend."""
    labels=np.asarray(labels)
    colors=np.asarray(classes['palette'])
    if labels.ndim!=2 or labels.dtype!=np.uint8:
        raise ValueError('Ground truth must be a 2D uint8 label array')
    if colors.shape!=(11,3) or np.any(colors<0) or np.any(colors>255):
        raise ValueError('Expected the engineering 11-class RGB palette')
    if np.any((labels>10)&(labels!=255)):
        raise ValueError('Ground truth IDs must be 0..10 or 255')
    height,width=labels.shape
    table=np.zeros((256,3),dtype=np.uint8)
    table[:11]=colors
    table[255]=[255,255,255]
    image=Image.fromarray(labels)
    image.putpalette(table.ravel().tolist())
    info=PngImagePlugin.PngInfo()
    info.add_text('fusion_label_width',str(width))
    info.add_text('fusion_label_height',str(height))
    info.add_text('fusion_label_ids','0..10; 255 ignored')
    path=Path(path)
    temporary=path.with_name('.'+path.name+'.palette.tmp')
    try:
        image.save(temporary,format='PNG',pnginfo=info)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

def load_ground_truth(path):
    """Return HxW IDs, excluding any footer; support legacy P/L label PNGs."""
    with Image.open(path) as image:
        if image.mode not in ('P','L'):
            raise ValueError('Expected indexed or grayscale ground truth PNG')
        has_size=('fusion_label_width' in image.info,'fusion_label_height' in image.info)
        if any(has_size) and not all(has_size):
            raise ValueError('Incomplete ground truth label-area metadata')
        width=int(image.info.get('fusion_label_width',image.width))
        height=int(image.info.get('fusion_label_height',image.height))
        if width!=image.width or not 0<height<=image.height:
            raise ValueError('Invalid ground truth label-area dimensions')
        labels=np.array(image.crop((0,0,width,height)),dtype=np.uint8)
    if np.any((labels>10)&(labels!=255)):
        raise ValueError('Unexpected ground truth class ID')
    return labels
