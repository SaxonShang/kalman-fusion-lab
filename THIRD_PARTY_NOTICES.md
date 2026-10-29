# 代码、数据与模型来源

此工作区基于 [lihongliang0203/SIO_SLAM](https://github.com/lihongliang0203/SIO_SLAM)，原始 Git 基点与本地修改记录见发布快照的 `results/provenance/`。主源码中保留原有作者和版权说明。本文件记录来源，不为上游代码重新授予许可证。

| 组件 | 来源与版本 | 本仓库中的位置 |
|---|---|---|
| SIO_SLAM | 原项目，导出时 HEAD 记录在 workspace.json | apps、include、src、tests 等 |
| GICP / Stereo-GICP | Ben-cpy/robotic-front，744ac2d993ae629b0cc44a3b1d995e04ab4fad48 | modules/registration；适配和原始文件哈希见 UPSTREAM.json |
| small_gicp | koide3/small_gicp，fa0cfc983417c044b6036f71dfa448038bb9c820 | 构建脚本下载，不复制安装产物 |
| Gaussian rasterizer 与附带第三方头文件 | 原工程随附版本 | submodules/；保留已有 LICENSE.md 与文件头 |
| FR5 录制格式与采集流程 | SaxonShang/Fr5_recording，high-speed 分支 | docs/FR5_DATASET.md；数据摘要在 results/ |
| 本次六条录制与两个 ONNX | 用户提供的实验资产 | 结果、哈希和约定见 results/inputs、results/models |

已有第三方授权文件随源码保留。主工程未发现顶层 LICENSE；本次整理不补写一个覆盖全部源码、数据和权重的新许可证。模型权重与原始图像需另行取得，仓库提供复现实验所需的身份校验和准备步骤。
