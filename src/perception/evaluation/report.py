"""Standalone HTML report and per-sample full-resolution segmentation images."""
from html import escape
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, PngImagePlugin
from perception.visualization.legend import append_legend
from .regions import REGIONS, REGION_NAMES, REGION_DIRECTORIES, continuous_lidar_coverage

BLIND_FILL_RGB = (224, 224, 224)


def legend_font_path():
    candidates = (Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
                  Path('/mnt/c/Windows/Fonts/msyh.ttc'))
    font = next((p for p in candidates if p.is_file()), None)
    if font is None:
        raise RuntimeError('A Chinese font (Noto Sans CJK or Microsoft YaHei) is required for semantic labels')
    return font


def save_segmentation(output, ident, rgb, truth, prediction, palette, classes, regions, front_blind, lidar_support=None):
    coverage = continuous_lidar_coverage(regions['lidar_valid'] if lidar_support is None else lidar_support)
    for name in REGIONS:
        display = coverage if name == 'lidar_valid' else regions[name]
        _save_region(output, ident, rgb, truth, prediction, palette, classes, regions[name], front_blind, name, display)


def _save_region(output, ident, rgb, truth, prediction, palette, classes, valid, front_blind, region, display):
    stem = f'point_{ident}'
    directory = REGION_DIRECTORIES[region]
    if truth.shape != prediction.shape or valid.shape != prediction.shape or valid.dtype != np.bool_:
        raise ValueError('Segmentation display mask and label dimensions must match')
    if not valid.any() or front_blind.shape != valid.shape or display.shape != valid.shape:
        raise ValueError('No valid pixels or incompatible front blind mask')
    for name in ('semantic', 'comparisons'):
        (output/name/directory).mkdir(parents=True, exist_ok=True)
    colors = np.asarray(palette, dtype=np.uint8)
    if region == 'evaluation_mask':
        if (valid & front_blind).any():
            raise ValueError('Front blind region overlaps the stored evaluation mask; check geometry or --front-blind-mask')
        body = np.array(rgb, copy=True)
        body[front_blind] = BLIND_FILL_RGB
        invalid_name = '车前盲区（不统计）'
    else:
        body = np.full((*prediction.shape, 3), BLIND_FILL_RGB, dtype=np.uint8)
        invalid_name = '雷达覆盖区外（不统计）'
    body[display] = colors[prediction[display]]
    # Show captured ground truth independently of either prediction mask.
    # Only genuinely unlabelled GT (255) is white; no blind-zone display fill.
    ground = np.full_like(body, 255)
    labelled = truth != 255
    ground[labelled] = colors[truth[labelled]]
    legend_ids = np.unique(prediction[display]).tolist()+[len(classes)]
    legend = {'classes': list(classes)+[invalid_name],
              'palette': list(palette)+[list(BLIND_FILL_RGB)]}
    semantic = Image.fromarray(append_legend(body, legend, legend_ids))
    info = PngImagePlugin.PngInfo()
    info.add_text('fusion_prediction_width', str(prediction.shape[1]))
    info.add_text('fusion_prediction_height', str(prediction.shape[0]))
    info.add_text('fusion_blind_fill_rgb', ','.join(map(str, BLIND_FILL_RGB)))
    info.add_text('fusion_legend_class_ids', ','.join(map(str, legend_ids[:-1])))
    info.add_text('fusion_region', region)
    info.add_text('fusion_background', 'original_rgb_except_front_blind' if region == 'evaluation_mask' else 'fixed_gray')
    info.add_text('fusion_display_mask', 'continuous_lidar_column_envelope' if region == 'lidar_valid' else 'evaluation_mask_and_valid_gt')
    info.add_text('fusion_scoring_mask', 'encoded_nonblack_lidar_and_valid_gt' if region == 'lidar_valid' else 'evaluation_mask_and_valid_gt')
    semantic.save(output/'semantic'/directory/f'{stem}.png', pnginfo=info)
    ground_truth = Image.fromarray(ground)
    panels = []
    prediction_title = 'Prediction (continuous LiDAR coverage)' if region == 'lidar_valid' else 'Prediction (evaluation_mask)'
    for title, image in (('RGB', rgb), ('Ground truth (full captured labels)', ground_truth), (prediction_title, Image.fromarray(body))):
        preview = image.copy()
        preview.thumbnail((480, 300))
        panels.append((title, preview))
    width = max(image.width for _, image in panels)
    height = max(image.height for _, image in panels)
    comparison = Image.new('RGB', (width*3, height+24), 'white')
    draw = ImageDraw.Draw(comparison)
    for index, (title, image) in enumerate(panels):
        draw.text((index*width+8, 5), title, fill='black')
        comparison.paste(image, (index*width, 24))
    comparison = Image.fromarray(append_legend(np.asarray(comparison), legend, legend_ids))
    comparison.save(output/'comparisons'/directory/f'{stem}.jpg', quality=90)


def write_report(output, report, samples, classes, palette):
    def pct(value):
        return '—' if value is None else f'{float(value):.2%}'

    region_sections = []
    for name in REGIONS:
        score = report['metrics'][name]
        rows = ''.join('<tr>'+''.join(f'<td>{escape(str(value))}</td>' for value in
                                    (c['class'], c['gt_pixels'], c['predicted_pixels'],
                                     pct(c['iou']), pct(c['precision']), pct(c['recall']), pct(c['f1'])))+'</tr>'
                       for c in score['per_class'])
        legend = ''.join(f'<span><i style="background:rgb({",".join(map(str, palette[i]))})"></i>'
                         f'{escape(classes[i])}</span>' for i in score['class_ids'])
        region_sections.append(f'<div class="card"><h2>{REGION_NAMES[name]}</h2><div class="scores">'
                               f'<div>像素一致率（参考）<b>{pct(score["pixel_accuracy"])}</b></div>'
                               f'<div>类别召回均值（参考）<b>{pct(score["mean_accuracy_present_gt"])}</b></div></div>'
                               f'<p>有效像素 {score["valid_pixels"]:,} · '
                               f'<a href="confusion_matrix_{name}.csv">本区域混淆矩阵</a></p>'
                               f'<div class="legend">{legend}</div><div class="table"><table><thead><tr>'
                               '<th>类别</th><th>真值像素</th><th>预测像素</th>'
                               '<th>IoU</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>'
                               f'<tbody>{rows}</tbody></table></div></div>')
    grouped = {}
    for sample in samples:
        grouped.setdefault(sample['id'], {})[sample['region']] = sample
    sections = []
    for ident, group in grouped.items():
        stem = f'point_{ident}'
        content = [f'<section><h3>点位 {escape(ident)}</h3>']
        for name in REGIONS:
            sample = group[name]
            directory = REGION_DIRECTORIES[name]
            content.append(f'<h4>{REGION_NAMES[name]}</h4>'
                           f'<p>像素一致率（参考） {pct(sample["pixel_accuracy"])}'
                           f' · 有效像素 {int(sample["valid_pixels"]):,}</p>'
                           f'<img loading="lazy" src="comparisons/{directory}/{stem}.jpg" alt="{REGION_NAMES[name]}结果对比">'
                           f'<p><a href="semantic/{directory}/{stem}.png">{"连续雷达覆盖区" if name == "lidar_valid" else "掩码区"}彩色分割（下附类别图例）</a></p>')
        sections.append(''.join(content)+'</section>')
    document = f'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>离线语义分割评测报告</title><style>
body{{font:16px/1.6 system-ui,sans-serif;color:#243247;background:#f3f6fa;margin:0 auto;padding:28px;max-width:1400px}}
h1,h2,h3{{line-height:1.3}} section,.card{{background:white;border:1px solid #dce3ed;border-radius:10px;padding:20px;margin:20px 0}}
.scores{{display:flex;gap:32px;flex-wrap:wrap}}.scores b{{display:block;font-size:28px}}table{{width:100%;border-collapse:collapse}}
td,th{{padding:9px;text-align:left;border-bottom:1px solid #dce3ed}}img{{max-width:100%;height:auto}}a{{color:#165cb1}}
.legend{{display:flex;gap:14px;flex-wrap:wrap}}.legend i{{display:inline-block;width:14px;height:14px;margin-right:6px}}
.table{{overflow-x:auto}}code{{overflow-wrap:anywhere}}
</style><h1>离线语义分割评测报告</h1>
<p>模型 {escape(report['model']['name'])} · 已观察 {report['completed_samples']} 个样本 · 流程状态 {escape(report['status'])}</p>
<div class="card"><strong>公开预训练模型观察报告</strong><p>当前使用公开模型，未针对本工厂仿真场景微调。
本报告用于查看预测、核对数据和排查推理问题；参考统计不作精度达标判断，也不设置模型精度门槛。</p></div>
<p>两套区域独立统计，不取交集：雷达区为编码非黑色区域，掩码区为 evaluation_mask.png 非零区域；两者均排除真值 255。
雷达有效支持区包含采集时的膨胀，不能反推原始点云。
真值和预测均未出现的类别不进入逐类表、混淆矩阵及均值。
逐类指标仅作诊断参考，类别召回均值排除没有真值分母的类别。无分母指标用 — 表示。
雷达预测图展示连续覆盖区内的模型逐像素预测，不再只显示离散雷达点：根据编码有效点每列的上下边界形成显示包络，
填平内部采样空隙，中间缺失列插值边界，最左/最右有效列之外不外推。该包络是显示估计，不新增实测回波，雷达指标仍只统计非黑色编码点。
覆盖区外用 RGB(224,224,224) 填充。掩码区结果仅在掩码内显示预测，车前盲区填灰，其他位置逐像素保留原始 RGB。
彩色分割图保留原图像区域并在下方附加类别图例。</p>
<p>可视化图例只显示类别名称，不显示类别编号。真值面板显示完整采集标签，不应用雷达区、评测掩码或车前盲区填充；
仅未标注真值（255）显示白色。该显示调整不改变两套评测统计范围。</p>
{''.join(region_sections)}
<p><a href="summary.json">JSON 摘要</a> · <a href="per_class.csv">逐类 CSV</a> ·
<a href="per_sample.csv">逐点 CSV（含区域列）</a></p>
<h2>逐样本分割结果</h2>{''.join(sections)}
<div class="card"><h2>复现信息与限制</h2><p>数据：<code>{escape(report['dataset'])}</code></p>
<p>权重 SHA256：<code>{escape(report['model']['checkpoint_sha256'])}</code></p>
<p>车前盲区来源：<code>{escape(str(report['front_blind_region']))}</code></p>
<p>{escape(report['limitations'])}</p></div></html>'''
    temporary = output/'report.html.tmp'
    temporary.write_text(document, encoding='utf-8')
    temporary.replace(output/'report.html')
