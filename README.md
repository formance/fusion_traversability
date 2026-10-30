# 多模态工厂巡检仿真工程

本工程在 WSL Ubuntu 24.04 中运行 ROS 2 Jazzy 与 Gazebo Harmonic，使用 Docker 执行 CMNeXt-B2 DELIVER RGB-L 推理，输出相机和激光雷达融合的语义分割结果。

## 启动与退出

```bash
cd /home/dministrator/Project/fusion_traversability
./run.sh
```

启动脚本启动 Gazebo、小车步进控制、GPU 推理容器和独立结果窗口。聚焦结果窗口：W/X 前进/后退、A/D 左转/右转；每次按键执行一个步进，参数在 `config/control.yaml`。仿真线速度上限为 1.5 m/s，默认步进配置为 1.0 m/s；实际速度受步长、控制器减速和仿真动力学约束。在结果窗口按 Q/Esc、关闭结果窗口，或在启动终端按 Q/Esc/Ctrl+C，退出整个会话。

长时间运行建议使用 `./run.sh --headless`：Gazebo 在后台运行，保留独立四格结果窗口。默认模型推理上限为 1 Hz；相机仍为 15 Hz，雷达仍为 10 Hz。可用 `--max-fps` 调整推理频率；`--no-viewer` 关闭结果窗口。其他参数参见 `./run.sh --help`。

## 工厂场景

场景位于 `src/factory_sim/worlds/factory.sdf`，厂房边界约 30×30 m，墙体高 5 m。小车从世界坐标 `(0, 3.5)` 出发，朝世界 +Y。左侧为带小推车的仓储区，右侧是有围栏的生产设备区，中央通道有静态叉车。生产区只有通道侧一个入口，按碰撞体净宽 2 m 布置；货架间净距 2.15 m。拟定路线的静态包络和设备尺寸见 `src/factory_sim/config/layout_manifest.json`。

下方左区为装配、维修和包装工位，包含维修台、零件料架、托盘、物料车及工具柜；下方右区为空压、泵组和电控公用工程区，包含空压机、储气罐、泵组撬装、过滤单元和配电柜。地面为中性灰，墙面为浅色加蓝灰下墙裙，立柱采用深蓝与黄黑底部标识，设备采用青蓝、绿色及金属灰；功能区使用低饱和度地坪和黄色边界线。输送台、护栏、叉车护顶架及小推车均有连接支撑。下方设备包围盒间距至少约2.11 m，现有巡检路线中心线到障碍物静态包围盒最小距离1.00 m。全部物体高度不超过5 m。

[工厂布局图](routes/factory_layout.png)与[实际相机预览](docs/scene_preview/factory_preview.jpg)展示当前场景。模型为基于实际设备资料的简化工程几何，资料来源见 src/factory_sim/SOURCE.json，不能作为厂家CAD或真实材质标定数据。

静态尺寸及净距只描述当前仿真模型，并非真实工厂安全认证；移动、转向、动态障碍物及资产修改后需要重新校核。叉车右侧不足2 m，不作为巡检通道。场景物体设置 Gazebo 语义标签，类别值为以下工程类别 ID 加 1；0 保留给未标注对象。RGB 与模型输出本身使用 0～10 的工程类别 ID。场景语义标签不意味着模型能够识别全部类别。

巡检采样路线配置和点位顺序见 routes/patrol_plan.json；路线图和检测范围检查说明见 routes/README.md。该路线为规划，执行时需用世界真值位姿及同步时间戳核验采样。

## 传感器和模型

厂区包含 4.8～5.0 米高的屋顶和 10 名静态工作人员，人员位置及对应采集点见 `src/factory_sim/config/layout_manifest.json`；人物为工服、安全帽的简化几何模型。

传感器参数在 `src/factory_sim/config/sensors.json`。相机为 See3CAM_24CUG 视野近似，1920×1200、15 Hz；雷达为水平安装的 EM4-T 视野近似，600×260 采样、10 Hz。两者为工程仿真配置，不替代实际硬件标定。

CMNeXt 输入为同步 RGB 和雷达投影图，权重为 `models/cmnext/cmnext_b2_deliver_rgbl.pth`，使用短边 376 对齐、当前相机输入 384×608，双线性缩放关闭抗锯齿以与本工程离线评测保持一致。投影图采用工程近似的 TURBO 对数深度编码，尚未确认与训练用雷达编码一致；没有纯 RGB 自动回退。 ROS 侧完成时间配对和对应时刻的 TF 查询，通过二进制数组接口发送 RGB、XYZ、内参及变换矩阵。容器内点云自身过滤、坐标变换、相机投影、最近深度选择、雷达编码、图像缩放、模型推理及标签后处理均在 CUDA 上执行，保留 FP32。ROS 消息收发、评估区域及盲区掩码、语义叠加和最终图像保存仍由 CPU 完成；旧 PNG 接口仅保留兼容使用。

## 语义输出类别

| ID | 类别 |
|---:|---|
| 0 | 地面／地板 |
| 1 | 天空／天花板 |
| 2 | 墙体／围栏／立柱 |
| 3 | 固定设备 |
| 4 | 行人 |
| 5 | 车辆 |
| 6 | 积水 |
| 7 | 坑洞 |
| 8 | 玻璃 |
| 9 | 其他障碍物 |
| 10 | 未知 |

模型原生 25 类到工程 11 类的近似映射见 `src/cmnext_inference/config/class_mapping.json`。DELIVER 没有坑洞、玻璃的直接输出；Water 类不能保证识别工厂积水，Static 等宽泛类别映射为未知。类别映射不赋予模型新的识别能力。

## 显示和话题

独立窗口的 2×2 面板依次显示原始相机、原始点云、匹配点云投影和语义分割。 原始点云面板使用嵌入式原生 OpenGL 子窗口，由 GPU 进行投影、深度遮挡和距离着色；最多显示抽样 6 万点，原始雷达数据不变。拖动旋转、滚轮缩放，W/A/D/X 和 Q/Esc 继续控制步进与退出。仅在订阅四格面板话题时回读点云画面。语义面板仅在前方 1～8 m、左右各 3 m 的平地投影检测区中叠加语义颜色；几何盲区显示淡琥珀色，检测区以外保留原始 RGB 像素。近地盲区只是几何假设，不代表传感器观测或可通行判断。

| 话题 | 内容 |
|---|---|
| `/camera/image_raw`、`/lidar/points_raw` | 原始相机、雷达 |
| `/cmnext/rgb`、`/cmnext/lidar_encoding` | 配对的 RGB 和模型雷达编码图 |
| `/cmnext/labels`、`/cmnext/model_labels` | 未应用盲区补全的 11 类标签 |
| `/cmnext/semantic_color`、`/cmnext/lidar_aligned` | 检测区语义叠加图和点云投影图；叠加图在检测区外与原图像素相同 |
| `/cmnext/evaluation_mask` | 几何有效评估掩码，255 为有效 |
| `/cmnext/ground_prior_mask` | 可视化先验实际应用区域 |
| `/cmnext/status`、`/robot/step_status` | 推理及步进状态 |

前向区域掩码基于平地射线，不是对障碍物表面进行三维区域裁剪；语义精度需使用同步真值，不能仅凭预测地面面积计算。

## 环境与资源

推理镜像为 `fusion_traversability_cmnext:latest`，使用 GPU、4 GiB 容器内存上限和 4 个 CPU。后台资源保护参数位于 `config/resources.json`；资源持续超过停止阈值时会停止本工程会话。Windows 可用内存低于 4 GiB 持续 4 秒会停止仿真，长期运行前应留出充足整机内存余量。静态坐标变换由单进程发布；模型输入配对队列上限为 4，无订阅的显示图像不做消息转换，点云投影显示按需生成。结果窗口仅缓存面板分辨率图像，不影响原始传感器数据或保存的评测原图。模型和传感器输入配置分别位于 `src/cmnext_inference/config/inference.json` 与 `src/factory_sim/config/`。

运行日志、临时评测文件和旧场景备份放在 `~/.local/state/fusion_traversability/`，不写入工程源代码目录。工程源码及资产来源见 `src/factory_sim/SOURCE.json`，权重来源见 `models/cmnext/model_info_deliver_rgbl.json`。
