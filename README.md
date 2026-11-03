# fusion_traversability

面向工厂巡检无人车的相机与激光雷达融合语义分割工程。仿真使用 WSL2 Ubuntu 24.04、ROS 2 Jazzy 和 Gazebo Harmonic；感知使用 Docker 中的 NVIDIA GPU 执行 StitchFusion-B2 DELIVER RGB-L FP32 推理。

## 当前功能

- 工厂场景约 30 × 30 m，包含货架、生产设备、围栏、叉车、物料车、屋顶及 10 名使用网格和工装贴图的静态工作人员。
- 实时感知提供相机、原始点云、对齐投影和语义结果四格窗口，以及 W/A/D/X 步进控制。
- 自动采集沿 28 个巡检点行驶、停车，校验位姿及时间配对，保存 RGB、雷达编码和语义真值。
- 离线验证直接读取采集图像，生成逐样本彩色分割图、真值对比图、HTML 报告及 JSON/CSV 统计，不启动仿真。
- 资源不足时自动停止会话，保留已完成的采集样本。

实时启动、自动采集和离线验证统一使用 `config/docker/` 配置，默认巡检路线位于 `config/routes/patrol_plan.json`。

实时运行时，在另一终端执行 `./run.sh --ros2 topic list --no-daemon` 查看话题，执行 `./run.sh --ros2 topic echo /perception/status --once --no-daemon` 查看推理状态。

实时感知保留 `/perception/labels`、`/perception/semantic_color`、`/perception/lidar_aligned` 和 `/perception/status` 四个输出话题。完整的传感器、车辆控制及基础话题见[话题说明](docs/INSTRUCTIONS.md#实时窗口与话题)。

## 离线验证

在工程根目录运行：

```bash
# 检查数据、执行 GPU 推理并生成报告
./scripts/val.sh data/patrol_capture

# 仅检查数据，不需要 Docker、权重或 CUDA
./scripts/val.sh data/patrol_capture --check-only

# 指定尚不存在的正式结果目录
./scripts/val.sh data/patrol_capture --output outputs/patrol_validation
```

推理需要 Docker Desktop 已启用 Ubuntu 的 WSL 集成和 GPU 支持，存在镜像 `fusion_traversability_stitchfusion:latest`、权重 `models/stitchfusion/deliver_rgbl.pth`，宿主 Python 具备 NumPy、Pillow、PyYAML 和可用中文字体。

终端会打印结果目录和报告完整路径。默认在 `outputs/` 下创建独立子目录；指定上述 `--output` 后，打开 `outputs/patrol_validation/report.html` 查看。报告通过相对链接访问同目录中的分割图和对比图，分享时应保留整个结果目录。

示例中的 `data/patrol_capture` 可通过 `./run.sh --capture --output data/patrol_capture` 创建，也可替换成已有采集目录。本地数据、结果和模型权重不随 Git 仓库分发。

## 验证结果

| 内容 | 文件或目录 |
|---|---|
| 中文评测报告 | `report.html` |
| 完成状态、配置、统计和运行信息 | `summary.json` |
| 逐类、逐样本参考统计 | `per_class.csv`、`per_sample.csv` |
| 两套区域的混淆矩阵 | `confusion_matrix_lidar_valid.csv`、`confusion_matrix_evaluation_mask.csv` |
| 连续雷达覆盖范围内的稠密预测，范围外填灰 | `semantic/evaluation_valid/` |
| 评测掩码内预测，车前盲区填灰，其余位置保留原图 | `semantic/evaluation_mask/` |
| RGB、完整真值、对应预测的三格对比 | `comparisons/evaluation_valid/`、`comparisons/evaluation_mask/` |

彩色分割图在下方附纯类别名称图例；真值面板不填充盲区，未标注真值显示白色。离线验证不输出 `predictions/`、`overlays/`。

两套区域独立统计：雷达编码非黑色区与有效真值的交集，以及原评测掩码区与有效真值的交集。连续雷达覆盖包络仅用于显示，不改变统计范围。真值和预测中均未出现的类别不统计。

当前模型未经本场景微调，不统计 mIoU 或加权 IoU，不设置精度达标门槛；保留像素一致率、逐类 IoU、精确率、召回率和 F1 等诊断参考。`completed` 仅表示验证流程完成。详细参数和口径见[离线验证说明](src/perception/evaluation/README.md)。

## 目录

```text
fusion_traversability/
├── README.md
├── docs/INSTRUCTIONS.md       # 环境、仿真、采集、传感器及控制说明
├── run.sh                     # 实时仿真与采集入口
├── scripts/val.sh             # 离线验证入口
├── config/
│   ├── docker/                # Dockerfile 与 Compose
│   ├── routes/                # 布局图、巡检路线图和点位配置
│   ├── control.yaml           # 步进控制
│   ├── fastdds.xml            # ROS 通信
│   └── resources.json         # 资源保护阈值
├── src/
│   ├── factory_sim/           # 场景、传感器、采集与运行控制
│   ├── perception/            # 几何投影、模型适配、ROS、显示及离线验证
│   └── third_party/           # StitchFusion、CMNeXt 上游代码和许可证
├── models/                    # 本地模型权重
├── data/                      # 正式采集数据与格式说明
└── outputs/                   # 正式验证报告和结果图
```

## 模型与仿真边界

相机为 1920 × 1200、实时 15 Hz；雷达为前向 600 × 260 个采样方向、实时 10 Hz。采集模式将 RGB、真值和雷达统一设为 2 Hz，可配置为 1～10 Hz。传感器采用硬件视野的工程近似，参数见 [sensors.json](src/factory_sim/config/sensors.json)。

相机与雷达的仿真安装位姿和 TF 外参一致，未注入外参标定误差；雷达已配置标准差 0.05 m 的高斯测距噪声。雷达图为带 3×3 扩张的 TURBO 对数深度编码，未确认与 DELIVER 官方点云转换完全一致。

网络原生 25 类输出映射为工程 11 类。类别映射、真值重采样和简化场景均存在近似；图像平面分割结果不能直接代表真实工厂性能或三维通行安全。

## 文档与来源

- [工程使用说明](docs/INSTRUCTIONS.md)
- [采集数据格式](data/README.md)
- [离线验证说明](src/perception/evaluation/README.md)
- [巡检路线说明](config/routes/README.md)
- [StitchFusion 源码说明](src/third_party/stitchfusion/README.md)、[许可证](src/third_party/stitchfusion/LICENSE)
- [场景与资产来源](src/factory_sim/SOURCE.json)、[人员模型来源和许可](src/factory_sim/models/factory_worker/SOURCE.json)

上游项目：[StitchFusion](https://github.com/LiBingyu01/StitchFusion)、[DELIVER / CMNeXt](https://github.com/InSAI-Lab/DELIVER)、[fusion_perception](https://github.com/FakeItUtillMakeIt/fusion_perception)、[O3DE Human Worker](https://github.com/RobotecAI/o3de-humanworker-gem)。使用时遵循各自许可证与模型条款。
