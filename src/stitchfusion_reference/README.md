# StitchFusion DELIVER RGB-L 推理源码

本目录保留 DELIVER RGB-L 推理所需的网络结构、DELIVER 配置、预处理参考及许可证。源码来源：https://github.com/LiBingyu01/StitchFusion ，版本 f04ecb75bd7391a047de47f5d3566bf249e2fdd7。

权重位于工程 models/stitchfusion/deliver_rgbl.pth。DELIVER RGB-L 推理按作者的结构提示使用模态特征相加；官方默认网络中的 FeatureCross 与该权重不匹配，不能直接将默认网络用于该权重。评测推理入口及资源保护脚本位于 ~/.local/state/fusion_traversability/stitch_evaluate.py 和 stitch_eval_supervisor.py。

configs/deliver.yaml 和 semseg/augmentations_mm.py 保留作官方预处理参考；配置中的训练路径不适用于本机。雷达输入使用本工程的近似编码，尚未验证为 DELIVER 官方编码。
