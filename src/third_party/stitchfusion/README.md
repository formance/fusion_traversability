# StitchFusion DELIVER RGB-L 推理源码

本目录保留 DELIVER RGB-L 推理所需的网络结构、配置、预处理参考及[许可证](LICENSE)。源码来源为 [LiBingyu01/StitchFusion](https://github.com/LiBingyu01/StitchFusion)，版本 `f04ecb75bd7391a047de47f5d3566bf249e2fdd7`。

工程默认使用公开 StitchFusion-B2 DELIVER RGB-L 权重，位于 `models/stitchfusion/deliver_rgbl.pth`，未经本工厂场景微调。权重不随仓库分发。

## 工程接入

- [模型适配器](../../perception/models/model_adapter.py)按 DELIVER 已发布权重的结构使用各阶段模态特征相加，并严格加载权重；该权重不使用默认 FeatureCross 结构。
- [离线验证入口](../../../scripts/val.sh)读取采集的 RGB、雷达编码和真值，生成正式报告与分割图，使用方式见[离线验证说明](../../perception/evaluation/README.md)。
- [推理服务](../../perception/services/server.py)和[Docker 配置](../../../config/docker/)提供 GPU 部署。

`configs/deliver.yaml` 和 `semseg/augmentations_mm.py` 作为上游配置与预处理参考，其中的训练数据路径不适用于本工程。雷达输入采用工程近似的 TURBO 对数深度编码，未确认与 DELIVER 官方编码完全一致；模型输出 25 类，工程按映射转换成 11 类。
