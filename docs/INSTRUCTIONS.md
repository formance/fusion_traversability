# 工程使用说明

本工程提供工厂仿真、相机与激光雷达融合分割、自动巡检采集和离线验证。以下路径以工程根目录为基准。

## 环境与模型准备

| 项目 | 要求或默认配置 |
|---|---|
| 系统 | Windows、WSL2 Ubuntu 24.04；图形界面使用 WSLg |
| 仿真 | ROS 2 Jazzy、Gazebo Harmonic |
| 推理 | NVIDIA GPU、支持 GPU 的 Docker Desktop 和 Docker Compose |
| WSL 集成 | Docker Desktop 中启用当前 Ubuntu 发行版的集成 |
| 推理镜像 | `fusion_traversability_stitchfusion:latest` |
| 模型权重 | `models/stitchfusion/deliver_rgbl.pth` |
| 离线宿主依赖 | Python 3、NumPy、Pillow、PyYAML；中文字体 Noto Sans CJK 或 Windows 微软雅黑 |

镜像和权重不包含在 Git 仓库中。已有镜像可用以下命令检查：

```bash
docker image inspect fusion_traversability_stitchfusion:latest
```

如需构建，先准备本地基础镜像 `ubuntu24.04_py312_torch271_cu126:latest`，再从工程根目录运行：

```bash
docker build --network=none \
  -f config/docker/Dockerfile.stitchfusion \
  -t fusion_traversability_stitchfusion:latest .
```

离线验证不会自动构建镜像或下载权重。默认仅使用 GPU，没有纯 RGB 或 CPU 自动回退。

## 离线验证与报告

离线入口已支持当前目录布局，不需要启动 ROS 或 Gazebo：

```bash
./scripts/val.sh data/patrol_capture
./scripts/val.sh data/patrol_capture --check-only
./scripts/val.sh data/patrol_capture --output outputs/patrol_validation
./scripts/val.sh --help
```

推理结束后，终端打印“验证结果目录”和“评测报告”的绝对路径。浏览器打开对应 `report.html`，可查看两套区域的参考统计及每个样本的 RGB/真值/预测对比。WSL 下可在结果目录执行 `explorer.exe .`，从 Windows 文件管理器打开报告。分享时保留整个结果目录，避免报告的相对图片链接失效。

正式结果默认保存在 `outputs/` 下自动创建的独立子目录；指定的输出目录必须尚不存在。仅检查模式输出 `summary.json`，模型验证输出 HTML、JSON、CSV、`semantic/` 和 `comparisons/`。完整参数、区域含义、图像显示及退出码见[离线验证说明](../src/perception/evaluation/README.md)。

## 实时仿真与自动采集

实时仿真和采集使用 `config/docker/compose.inference.yaml`；默认路线使用 `config/routes/patrol_plan.json`。

实时入口及参数：

```bash
./run.sh
./run.sh --headless
./run.sh --headless --max-fps 1
./run.sh --headless --no-viewer
```

默认启动 Gazebo、小车步进控制、GPU 推理服务及独立四格窗口。`--headless` 关闭 Gazebo 图形界面，保留结果窗口；`--no-viewer` 关闭结果窗口；`--paused` 使仿真从暂停状态启动。推理频率默认上限为 1 Hz，`--viewer-width` 默认 512，`--startup-timeout` 默认 180 秒。

聚焦结果窗口，W/X 前进或后退，A/D 原地左转或右转，每次按键执行一个步进。`config/control.yaml` 默认直线步长 0.20 m、转角 15°、线速度 1.0 m/s、角速度 0.20 rad/s；实际运动受减速和仿真动力学影响。在窗口按 Q/Esc、关闭窗口，或在启动终端按 Q/Esc/Ctrl+C，停止整个会话。

自动采集入口及参数；`--capture` 必须放在第一项：

```bash
./run.sh --capture --output data/patrol_capture
./run.sh --capture --route config/routes/patrol_plan.json --gui
./run.sh --capture --route config/routes/patrol_plan.json --sensor-rate 5
./run.sh --capture --route config/routes/patrol_plan.json --limit 3
```

采集模式实际行驶到点位、停车并校验世界坐标位姿，不加载语义分割模型，只使用 GPU 投影服务生成雷达编码。默认后台运行，`--gui` 显示 Gazebo；`--output` 指定新的数据目录；`--limit` 限制本次采集点数；`--sensor-rate` 范围为 1～10 Hz，默认 2 Hz。Ctrl+C 停止，已完成样本保留，整批完成状态以 `result.json` 为准。

采集将 RGB、语义真值和雷达统一降为所选频率，保持图像分辨率、雷达采样密度、内外参及噪声配置。保存要求为 RGB/真值严格同时间戳、RGB/雷达相差不超过 50 ms、位置误差不超过 0.12 m、朝向误差不超过 3°，并通过停车稳定性检查。数据默认在 `data/` 下创建独立子目录；上述 `--output` 指定 `data/patrol_capture/`，格式见[数据说明](../data/README.md)。

## 场景与路线

厂房约 30 × 30 m，墙体及屋顶最高约 5 m。小车从世界坐标 `(0, 3.5)` 出发，朝世界 +Y；仓储、生产设备、装配维修及公用工程区通过绿色主通道连接。场景包含静态叉车、物料车及 10 名使用 O3DE Human Worker 网格和工装贴图的工作人员，人员采用六种静态姿态，不使用运行时骨骼动画。

- [工厂布局图](../config/routes/factory_layout.png)
- [巡检路线图](../config/routes/patrol_plan.png)与[路线说明](../config/routes/README.md)
- [场景尺寸配置](../src/factory_sim/config/layout_manifest.json)
- [人员位置和姿态](../src/factory_sim/config/workers.json)、[姿态预览](../src/factory_sim/models/factory_worker/preview.png)

Gazebo 图形界面中屋顶自动透明，传感器渲染与碰撞场景中的屋顶仍为实体。路线的静态净距不包含车辆转弯扫掠或动态避障，不能作为通行安全认证。场景或传感器配置变化后，需要重新采集匹配的数据。

## 传感器与外参

参数以 [sensors.json](../src/factory_sim/config/sensors.json)、[extrinsics.yaml](../src/factory_sim/config/extrinsics.yaml) 和[机器人模型](../src/factory_sim/models/factory_robot/model.sdf)为准。

| 项目 | 当前值 |
|---|---|
| 相机 | See3CAM_24CUG 视野近似，1920 × 1200，实时 15 Hz |
| 相机内参 | fx=741.9723、fy=1006.5094、cx=960、cy=600；畸变系数为 0 |
| 相机安装 | 相对 `base_link` 为 `(0.42, 0, 0.65)` m，向下俯视 15° |
| 雷达 | EM4-T 视野近似，600 × 260 方向，实时 10 Hz |
| 雷达安装 | 相对 `base_link` 为 `(0.42, 0, 0.40)` m，水平安装 |
| 雷达视野 | 水平约 120°，垂直 −12.5°～12.5° |
| 雷达量程 | 仿真设置 1～120 m |
| 雷达测距噪声 | 高斯噪声，均值 0、标准差 0.05 m |

仿真安装位姿与投影所用 TF 外参一致，未注入外参标定误差。相机和雷达高度差 0.25 m 已纳入变换。相机光学坐标系通过 RPY `(-π/2, 0, -π/2)` 从 `camera_link` 转换；投影使用相应时刻的 TF，并以 `odom` 为固定参考帧。

这些参数是硬件视野的工程近似，不是实机标定。仿真没有模拟真实镜头响应、逐点扫描时间、镜面多径等效应；像素取整、有限采样、3×3 编码扩张与两传感器视角遮挡差异仍可能使局部轮廓不完全重合。

## 模型输入与输出

模型使用公开 StitchFusion-B2 DELIVER RGB-L 权重，未经本工厂场景微调。RGB 与雷达编码按短边 376 缩放，再对齐到 32 的倍数，当前输入宽 × 高为 608 × 384。RGB 使用 ImageNet 归一化，雷达编码除以 255，推理和后处理采用 FP32。

网络输出 25 类 logits；当前输入对应形状为 `[1, 25, 384, 608]`。适配器先将 logits 双线性恢复到 `[1, 25, 1200, 1920]`，再 argmax 得到 0～24 的原生类别图，随后按[类别映射](../src/perception/config/class_mapping.json)生成工程 11 类。logits 不保存为验证结果，彩色分割、图例和盲区显示属于后处理。

| 工程 ID | 类别 |
|---:|---|
| 0 | 地面／地板 |
| 1 | 天空／天花板 |
| 2 | 墙体／围栏／立柱 |
| 3 | 固定设备 |
| 4 | 人 |
| 5 | 车辆 |
| 6 | 积水 |
| 7 | 坑洞 |
| 8 | 玻璃 |
| 9 | 其他障碍物 |
| 10 | 未知 |

Gazebo 场景语义标签为工程 ID 加 1，0 表示未标注；保存的真值使用工程 ID，255 表示忽略。DELIVER 缺少坑洞、玻璃的直接输出，部分宽泛原生类别映射为未知，映射不会增加模型的识别能力。

## 实时窗口与话题

实时运行期间，在另一个终端进入工程根目录，使用同一入口查看话题：

```bash
./run.sh --ros2 topic list --no-daemon
./run.sh --ros2 topic info /camera/image_raw --no-daemon
./run.sh --ros2 topic echo /perception/status --once --no-daemon
```

`--ros2` 只执行 ROS 命令，不启动仿真。查看话题时需保持实时会话运行。

四格窗口依次显示原始相机、原始点云、匹配点云投影和语义结果。点云采用原生 OpenGL 显示，拖动旋转、滚轮缩放；最多抽样显示 6 万点，不改变输入点云。

实时语义叠加采用 `front_perception` 前方 1～8 m、左右各 3 m 的平地投影区域，几何盲区显示淡琥珀色，区域外保留原图。实时可视化先验不修改用于统计的标签。离线图像的灰色盲区显示和两套统计范围见[验证说明](../src/perception/evaluation/README.md)。

实时感知输出：

| 话题 | 消息类型 | 内容 |
|---|---|---|
| `/perception/labels` | `sensor_msgs/msg/Image` | 原图尺寸的工程 11 类标签，不应用显示用盲区先验 |
| `/perception/semantic_color` | `sensor_msgs/msg/Image` | 检测区语义显示图，供四格窗口使用 |
| `/perception/lidar_aligned` | `sensor_msgs/msg/Image` | 雷达在 RGB 图像上的投影显示图，供四格窗口使用 |
| `/perception/status` | `std_msgs/msg/String` | JSON 格式的推理状态 |

传感器和车辆控制：

| 话题 | 消息类型 | 内容 |
|---|---|---|
| `/camera/image_raw` | `sensor_msgs/msg/Image` | 原始相机图像，供推理、显示和采集使用 |
| `/camera/camera_info` | `sensor_msgs/msg/CameraInfo` | 原始相机内参，供投影和采集使用 |
| `/lidar/points_raw` | `sensor_msgs/msg/PointCloud2` | 原始雷达点云，供投影、显示和采集使用 |
| `/robot/step_command` | `std_msgs/msg/String` | W/X/A/D 步进控制指令 |
| `/robot/step_status` | `std_msgs/msg/String` | 步进执行状态，供窗口和自动采集使用 |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | 步进控制器发送给仿真小车的速度指令 |
| `/odom` | `nav_msgs/msg/Odometry` | 小车里程计，供步进控制和采集使用 |

仿真与 ROS 基础话题：

| 话题 | 消息类型 | 内容 |
|---|---|---|
| `/clock` | `rosgraph_msgs/msg/Clock` | 仿真时间 |
| `/tf` | `tf2_msgs/msg/TFMessage` | 里程计到车体的动态坐标变换 |
| `/tf_static` | `tf2_msgs/msg/TFMessage` | 车体、相机和雷达之间的静态坐标变换 |
| `/rosout` | `rcl_interfaces/msg/Log` | ROS 节点日志 |
| `/parameter_events` | `rcl_interfaces/msg/ParameterEvent` | ROS 参数变更事件 |

实时模式中，`ros2 topic list` 会显示感知、传感器、控制及基础话题。基础话题由仿真和 ROS 节点使用。

自动采集模式另外使用 `/evaluation/labels_raw`（`sensor_msgs/msg/Image`）接收语义真值，不启动实时分割节点，因此不发布上述四个感知输出。离线验证直接读取采集文件，不发布 ROS 话题。

感知输出统一使用 `/perception` 命名空间，话题接口不依赖具体模型。实时 `--save-every N` 可每 N 个推理结果向 `outputs/` 保存 RGB、标签、显示图、雷达编码和掩码，默认 0 表示不保存；这些实时文件不属于 `val.sh` 的离线结果格式。

## 中断与结果

资源不足时会自动停止会话。采集中断后保留已完成样本，可用 `--allow-partial` 验证；离线验证失败或中断时，终端显示原因及失败记录位置，不把部分结果作为成功报告输出。

采集数据位于 `data/`，正式验证报告、分割图及显式保存的实时结果位于 `outputs/`。已有采集或验证目录不会被覆盖。
