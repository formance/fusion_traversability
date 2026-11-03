# 离线验证使用说明

入口为工程根目录下的 `scripts/val.sh`。验证直接读取采集时保存的 RGB、雷达编码和真值，不启动 ROS、Gazebo 或投影服务。默认使用未经本场景微调的公开 StitchFusion-B2 DELIVER RGB-L FP32 CUDA 模型。

## 运行

在工程根目录执行：

```bash
# 数据检查、推理、逐样本图像及评测报告
./scripts/val.sh data/patrol_capture

# 只检查数据
./scripts/val.sh data/patrol_capture --check-only

# 指定尚不存在的结果目录
./scripts/val.sh data/patrol_capture --output outputs/patrol_validation

# 评测未完成采集中的已保存完整点位
./scripts/val.sh data/未完成的数据集 --allow-partial

./scripts/val.sh --help
```

示例中的 `data/patrol_capture` 可由 `./run.sh --capture --output data/patrol_capture` 创建，也可替换成已有采集目录。

默认推理依赖现有镜像 `fusion_traversability_stitchfusion:latest`、权重 `models/stitchfusion/deliver_rgbl.pth`、支持 GPU 的 Docker Compose 和正常的 WSL 集成；不会自动下载或构建。宿主 Python 需要 NumPy、Pillow、PyYAML，图例需要 Noto Sans CJK 或 Windows 微软雅黑。

| 参数 | 用途 |
|---|---|
| `dataset` | 包含 `metadata.json`、`result.json`、`evaluation_mask.png` 和逐样本三张图的采集目录 |
| `--output` | 新的正式结果目录；默认在 `outputs/` 下自动创建独立子目录，工程内仅允许位于 `outputs/` |
| `--check-only` | 仅执行数据检查，不需要 Docker、模型权重、中文字体或 CUDA |
| `--allow-partial` | 允许处理采集完成标志不完整的数据集，仍检查所有已保存点位 |
| `--checkpoint` | 指定兼容 StitchFusion-B2 DELIVER RGB-L 的权重文件 |
| `--front-blind-mask` | 指定原图尺寸的 L 模式二值盲区 PNG，像素仅含 0 和 255 |
| `--resource-config` | 指定资源保护配置，默认 `config/resources.json` |
| `--local` | 在当前 Python 的 CUDA/PyTorch 和模型依赖环境推理，不经过 Docker 或宿主资源监控 |

`--local` 同样要求中文字体及 CUDA，不是 CPU 模式。输入数据格式见[数据说明](../../../data/README.md)。

## 查看报告

完成后终端打印“验证结果目录”和“评测报告”的绝对路径。浏览器直接打开该目录的 `report.html`；WSL 下可在结果目录运行 `explorer.exe .`，从文件管理器打开报告。报告中的图片和下载链接采用相对路径，复制或分享时保留整个结果目录。

| 文件或目录 | 内容 |
|---|---|
| `report.html` | 中文报告、两套区域参考统计、逐样本三格对比及彩色分割图链接 |
| `summary.json` | 状态、数据检查、统计、混淆矩阵、模型配置、文件校验信息、软件/GPU 信息和耗时 |
| `per_class.csv` | 带 `region` 列的逐类像素数、IoU、precision、recall、F1 |
| `per_sample.csv` | 每样本两行，分别记录两套区域的有效像素和像素一致率；模型每样本仅推理一次 |
| `confusion_matrix_lidar_valid.csv` | 雷达编码非黑色有效区的混淆矩阵 |
| `confusion_matrix_evaluation_mask.csv` | 评测掩码区的混淆矩阵 |
| `semantic/evaluation_valid/point_ID.png` | 连续雷达覆盖范围内的稠密预测，范围外灰色 |
| `semantic/evaluation_mask/point_ID.png` | 掩码内预测、车前盲区灰色，其余位置保留 RGB |
| `comparisons/evaluation_valid/point_ID.jpg` | RGB、完整真值和连续雷达覆盖预测三格预览 |
| `comparisons/evaluation_mask/point_ID.jpg` | RGB、完整真值和评测掩码预测三格预览 |

不输出 `predictions/`、`overlays/` 或 logits。`--check-only` 仅输出数据检查摘要 `summary.json`，不生成 HTML 和图像。

## 独立统计范围

| 统计标识 | 有效像素 |
|---|---|
| `lidar_valid` | 雷达编码非黑色区 AND 真值不为 255 |
| `evaluation_mask` | 保存的 `evaluation_mask.png` 非零区 AND 真值不为 255 |

两套区域独立累加，不取两者交集。掩码区不因雷达编码为黑色而被排除。当前编码缺失值为 RGB(0,0,0)，非黑色支持区含采集时的 3×3 扩张，不能作为原始雷达落点掩码。图像目录名 `evaluation_valid` 对应统计标识 `lidar_valid`。

混淆矩阵行为真值、列为预测。区域内真值与预测均未出现的类别不进入逐类表或导出的混淆矩阵；只在真值或预测一侧出现的类别仍统计。保留有效像素、像素一致率、真值中出现类别的召回均值及逐类 IoU/precision/recall/F1，**不计算 mIoU 或加权 IoU，不设置精度达标门槛**。

无分母指标记为 JSON `null`；JSON/CSV 比例为 0～1，HTML 和终端按相应格式显示。报告用于公开模型预测观察和诊断，`completed` 仅表示流程完成，不表示模型在本场景达标。

摘要格式为 `report_version: 5`，`assessment_mode` 为 `public_pretrained_model_observation`。`metrics`、`confusion_matrices` 按两套区域保存；各区域 `class_ids` 为实际参与统计的类别，`training_context` 说明公开模型未经本场景微调。

## 结果图显示

彩色分割图的原图区域保持采集尺寸 1920 × 1200，下方附本区域预测类别图例。图例和 HTML 仅显示类别名称；JSON、CSV 和 PNG 元数据保留机器处理所需的类别 ID。

连续雷达覆盖范围由非黑色编码像素逐列的最上、最下边界构成。中间无回波列的上下边界线性插值，不向最左、最右有效列之外延伸。包络内直接显示模型的稠密预测，采样空隙处也使用模型逐像素输出；包络外固定填充 RGB(224,224,224)。该范围只用于显示，**统计仍采用非黑色编码支持区与有效真值的交集**。

评测掩码图在有效掩码内显示预测，只把车前盲区填灰，其他位置保留原始 RGB 像素。真值面板使用完整采集标签着色，不应用统计掩码或盲区填充；只有原真值 255 显示白色。

PNG 元数据中的 `fusion_prediction_width`、`fusion_prediction_height` 记录原图区域尺寸，`fusion_region` 标明区域，`fusion_display_mask`、`fusion_scoring_mask` 说明显示与统计范围。

盲区优先读取 `--front-blind-mask` 或数据集的 `front_blind_mask.png`。未记录独立盲区文件的数据集，可从同机工程外采集会话的 `evaluation.sdf` 读取传感器几何并结合当前平地策略重建；没有会话几何时使用当前传感器配置。来源与假设记录在 `front_blind_region`。显式指定或元数据声明的文件缺失、尺寸不匹配，或盲区与评测掩码重叠时会失败，不静默替代。

## 数据检查与推理

检查采集完成标志、点位唯一性、目录与记录一致性、图像存在性、原始尺寸、标签范围、时间差、位姿确认及误差。只有显式添加 `--allow-partial` 才允许不完整批次。真值按类别 ID 读取，不将调色板转换成灰度；支持记录了原始尺寸的标签页脚。

采集数据没有单独记录 GT 时间戳，无法仅凭数据集事后独立复核 RGB/GT 同时间戳。记录的世界位姿是采后校验值，路线位姿误差不等于传感器标定误差。

输入按元数据中受支持的短边、两阶段双线性缩放和 32 倍数对齐处理；RGB 使用 ImageNet 归一化，雷达除以 255，FP32 推理。模型原生 25 类恢复原图尺寸后按 `prediction_source_to_target` 映射为工程 11 类，统计不应用可视化地面先验。

## 完成状态与退出码

已有结果目录不会被覆盖。失败、中断或资源不足时，终端显示原因及失败记录位置，不把部分结果作为成功报告输出。

| 退出码 | 含义 |
|---|---|
| 0 | 数据检查成功，或模型观察流程完成 |
| 1 | 数据异常、模型/容器失败、资源保护退出或中断 |
| 2 | 命令行参数错误 |

真值重采样、25→11 类映射、雷达编码及仿真资产均有工程近似；参考统计不能替代目标场景微调后的验证或真实工厂性能评估。
