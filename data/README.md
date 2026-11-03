# 采集数据格式

数据由自动巡检采集入口生成，默认在 `data/` 下创建独立子目录。使用 `./run.sh --capture --output data/patrol_capture` 可指定以下示例路径，已有目录不会被覆盖。启动方式见[工程使用说明](../docs/INSTRUCTIONS.md#实时仿真与自动采集)。

## 文件结构

```text
data/patrol_capture/
├── metadata.json
├── result.json
├── evaluation_mask.png
├── front_blind_mask.png
└── samples/
    ├── point_01/
    │   ├── rgb.png
    │   ├── lidar_encoding.png
    │   └── ground_truth.png
    └── point_02/ ...
```

| 文件 | 内容 |
|---|---|
| `metadata.json` | 类别、25→11 类映射、编码方式、输入预处理、尺寸、采集频率及校验容差 |
| `result.json` | 采集完成状态、点位、目标/实际位姿、时间戳及校验误差 |
| `evaluation_mask.png` | 整批共用的几何评测区域，L 模式，0 无效、255 有效 |
| `front_blind_mask.png` | 整批共用的车前平地几何盲区，L 模式，255 为盲区 |
| `rgb.png` | 原始 RGB，相机尺寸 1920 × 1200 |
| `lidar_encoding.png` | 同尺寸雷达投影编码，RGB(0,0,0) 表示无编码值 |
| `ground_truth.png` | 同尺寸索引标签，工程类别 0～10，255 为忽略 |

每点只保存三张图，不保存原始 XYZ 点云、logits、重复掩码或场景副本。默认路线包含 28 个点位。对于未保存独立 `front_blind_mask.png` 的数据，验证可读取工程外采集会话几何重建盲区，也支持显式指定掩码。

## 标签读取

真值 PNG 的调色板仅用于显示，图像本体存储类别 ID，没有类别名称或底部图例。使用配套读取函数：

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 - <<'PYTHON'
from perception.io.label_io import load_ground_truth
labels = load_ground_truth('data/patrol_capture/samples/point_01/ground_truth.png')
print(labels.shape, labels.dtype)
PYTHON
```

返回原始 H×W 的 `uint8` 标签，读取器也支持带尺寸元数据的标签页脚。索引 PNG 可直接通过 `np.array(PIL.Image.open(path))` 读取；不要先转换成 RGB 或灰度 L，也不要用 OpenCV 的彩色/灰度解码值作为类别 ID。255 显示白色，颜色不是类别编号。

## 采集与投影

采集 RGB、语义真值和雷达默认统一为 2 Hz，可通过 `--sensor-rate` 设为 1～10 Hz。相机分辨率、600 × 260 雷达方向、内外参及噪声保持配置值。停车稳定后才保存，RGB/真值严格同时间戳，RGB/雷达差不超过 50 ms，位置与朝向误差分别不超过 0.12 m 和 3°。中断时保留已完成点位，整批状态见 `result.json`；`--limit` 采样也可能留下非完整数据集。

雷达 XYZ 在内存中完成自身过滤、按时间查询 TF、变换到相机光学坐标系，再用 CameraInfo 内参投影。同像素选择最近深度，执行 3×3 扩张并用 TURBO 显示对数相机深度，编码上限为 120 m。非黑色编码支持区包含扩张，不等于原始点云落点。

仿真安装位姿与 TF 外参一致，未加入外参误差；测距噪声标准差为 0.05 m。保存的编码无法恢复精确 XYZ；修改内外参、编码公式或场景后应重新采集。真值由较低分辨率语义传感器映射到 RGB，轮廓保留采样误差。

## 离线使用

```bash
./scripts/val.sh data/patrol_capture
./scripts/val.sh data/patrol_capture --check-only
./scripts/val.sh data/未完成的数据集 --allow-partial
```

验证使用元数据中的类别映射，分别统计雷达非黑色编码区和原评测掩码区，二者均排除真值 255。区域内真值和预测均未出现的类别不统计。模型为未进行场景微调的公开权重，不统计 mIoU 或加权 IoU，不设精度门槛。

正式报告与逐样本图像保存到 `outputs/` 下本次运行的独立结果目录，打开其中的 `report.html` 查看。只检查模式仅输出 `summary.json`。结果图包含连续雷达覆盖范围预测和评测掩码范围预测；图例仅含类别名称，真值面板不填充盲区。完整参数、显示与统计定义见[离线验证说明](../src/perception/evaluation/README.md)。

