from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def append_legend(rgb, classes, present_ids=None):
    """Append a single-row legend with equal spacing between detected classes."""
    width=rgb.shape[1]
    candidates=[Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),Path('/mnt/c/Windows/Fonts/msyh.ttc')]
    font_path=next((p for p in candidates if p.is_file()),None)
    if font_path is None: raise RuntimeError('A Chinese font is required for the prediction legend')
    ids=list(range(len(classes['classes']))) if present_ids is None else sorted({int(i) for i in present_ids})
    if any(i<0 or i>=len(classes['classes']) for i in ids): raise ValueError('Invalid legend class ID')
    size=max(8,min(28,width//65))
    def item_widths(font):return [size+6+int(np.ceil(font.getlength(classes['classes'][i]))) for i in ids]
    while size>8:
        font=ImageFont.truetype(str(font_path),size);widths=item_widths(font)
        if sum(widths)+12*(len(ids)+1)<=width:break
        size-=1
    font=ImageFont.truetype(str(font_path),size);widths=item_widths(font)
    if sum(widths)>width:raise ValueError('Image too narrow for a legible single-row legend')
    result=Image.new('RGB',(width,rgb.shape[0]+size+20),(245,245,245));result.paste(Image.fromarray(rgb),(0,0));draw=ImageDraw.Draw(result);gap=(width-sum(widths))/(len(ids)+1);x=gap;y=rgb.shape[0]+8
    for i,item_width in zip(ids,widths):
        xx=round(x);name=classes['classes'][i];color=classes['palette'][i];draw.rectangle((xx,y+2,xx+size,y+size+2),fill=tuple(color),outline=(60,60,60));draw.text((xx+size+6,y),name,font=font,fill=(20,20,20));x+=item_width+gap
    return np.asarray(result)
