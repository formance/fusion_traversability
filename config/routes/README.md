# 巡检路线

本目录包含当前场景的布局与 28 个巡检采样点，坐标单位为米，偏航角单位为度。

| 文件 | 用途 |
|---|---|
| [factory_layout.png](factory_layout.png) | 工厂布局 |
| [patrol_plan.png](patrol_plan.png) | 点位、采样顺序和车头朝向 |
| [patrol_plan.json](patrol_plan.json) | 点位坐标、偏航角、顺序及相邻段几何检查 |

世界坐标图中 +X 向右、+Y 向上，yaw 从 +X 逆时针计算。点位按 01～28 编号，同一坐标可用不同朝向重复采样，路线图中的箭头可能重合。起点为 `(0, 3.5)`，朝向 +Y。

## 采集使用

采集参数显式指定当前配置路径：

```bash
./run.sh --capture --route config/routes/patrol_plan.json
```

不传 `--route` 时也默认使用本目录的 `patrol_plan.json`。使用 `--output data/patrol_capture` 指定采集目录，完整参数见[工程使用说明](../../docs/INSTRUCTIONS.md#实时仿真与自动采集)。

采集程序实际行驶并在停车后校验世界真值位姿和传感器时间，不通过移动传感器或修改图像生成样本。采集完成状态以输出数据集 `result.json` 为准；路线配置中的 `status` 是路线规划属性，不是执行结果。

## 几何检查范围

相邻点的规划检查以此前点位的 `front_perception` 为参考，前向 1～8 m、左右各 3 m，并检查下一点地面位置位于理想相机和雷达视锥内。配置中 27 段均通过，静态路线中心线到保守障碍物包围盒的最小距离为 1.00 m。

这些检查不包含遮挡、回波密度、物体高度、车身转弯扫掠或动态障碍物，不能证明整条路线可安全通行。场景资产或点位变化后需要重新检查并采集。场景与人员配置分别见 [layout_manifest.json](../../src/factory_sim/config/layout_manifest.json) 和 [workers.json](../../src/factory_sim/config/workers.json)。
