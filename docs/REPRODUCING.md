# 复现实验

本文是可选技术附录。理解实验和提出修改建议不需要执行以下命令；请先读 [实验审阅说明](EXPERIMENT_REVIEW_BRIEF.md)、[实验设计](EXPERIMENT_DESIGN.md) 与 [协方差诊断](FUSION_COVARIANCE_DIAGNOSIS.md)。

新仓库以 `results/` 保存不可覆盖的历史证据，新实验写入 Git 忽略的 `artifacts/`。历史 YAML 中的绝对路径保留用于追踪；跨机器执行请使用 `scripts/reproduce_published.py`，它在新配置中重建路径，不改写历史文件。

## 1. 只查看和重建汇总报告

图片、Markdown 和 CSV 可在 GitHub 直接阅读。HTML 报告需下载后在浏览器打开，或在仓库根目录启动 `python3 -m http.server 8000 --bind 127.0.0.1`，访问 `http://localhost:8000/results/diagnostics/20261029_105246/report.html`。

Ubuntu 22.04 的 Python 报告依赖与命令：

```bash
sudo apt update
sudo apt install python3-yaml python3-numpy python3-opencv python3-matplotlib
python3 scripts/reproduce_published.py verify
python3 scripts/reproduce_published.py audit
```

`verify` 校验导出快照中的 SHA256，不需要 Python 第三方包。`audit` 从六条历史对照的矩阵和 CSV 重建归一化 K 校验、P/R 表、图和 HTML；不执行滤波、不读取原图或 GT，不需要 ROS/GPU。新报告路径在终端打印。

## 2. 从冻结观测回放 ESKF

已装好 ROS 2 Humble 的 Ubuntu 22.04 上：

```bash
sudo apt install build-essential cmake ninja-build git curl ca-certificates \
  libeigen3-dev libyaml-cpp-dev libpcl-dev libopencv-dev libgtest-dev \
  ros-humble-rclcpp ros-humble-sensor-msgs ros-humble-geometry-msgs \
  ros-humble-pcl-conversions ros-humble-ament-cmake-gtest
bash scripts/build_deps.sh cpu
python3 scripts/fusion_lab.py build
python3 scripts/fusion_lab.py test
python3 scripts/reproduce_published.py replay --sequence up --source repropagated
```

ROS 未安装时先按 [ROS Humble 官方安装步骤](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html) 配置 ROS 软件源并安装；本节 CPU 实验不需要执行安装 CUDA/TensorRT 的系统脚本。

ESKF 需要的完整先验、P、GICP 位姿和 R 均在 `observations.yml.gz`；报告使用归档 GT 和标定，所以不依赖原始图像、点云缓存、ONNX、TensorRT 或 GPU。

```bash
# 同一旧 P 与观测，仅运行当前已修复的 ESKF 求解器
python3 scripts/reproduce_published.py replay --sequence all --source reference
# 已修正初始 P 的冻结观测，检查预先指定的 P/R 敏感性
python3 scripts/reproduce_published.py replay --sequence all --source repropagated \
  --prior-scale 0.0001 --scales 0.1 1 10 1000 100000
```

当前程序不会恢复旧的错误求解器；旧输出原样保存在 `results/fusion/`。历史源码片段用于审查修复依据，不是独立可编译工程。CPU/PCL/OpenMP 版本不同可能造成浮点差异；新运行会记录实际代码和程序哈希。

## 3. 从录制图像重跑 GICP 与 ESIKF

取得本次六条录制，按下面结构放置原图及原始 CSV/报告。采集格式和说明来自 [Fr5_recording high-speed](https://github.com/SaxonShang/Fr5_recording/tree/high-speed)；该代码仓库链接不保证含本次约 4.5 GB 图像，原图需向实验提供者取得。

```text
data/recordings/recordings_highspeed/
├── calib/                       与 results/recording/calib/ 对照
├── 20261015_112553_up/save_data/
│   ├── left/  right/
│   ├── img_timeline.csv  zed_imu.csv  groundtruth_tum.txt
│   └── frame_lag_report.json  groundtruth_report.json
└── 其余五条序列/save_data/...
```

```bash
python3 scripts/prepare_fr5_dataset.py
python3 scripts/reproduce_published.py run-cache --sequence up
# --sequence all 可执行六条；--case main_motion 可先检查单组
```

`run-cache` 读取归档的同一批 54 帧模型视差，以当前代码重新准备点云和纯惯性先验，独立运行 GICP、ESKF 与 ESIKF。它不调用模型推理，不使用旧的点云缓存，也不需要 GPU。会校验原录制索引、IMU 和所选图像的哈希；输入不一致时停止，避免把不同数据当作本次复现。

只想验证数据接口而没有同一模型视差时，可显式运行 SGBM：

```bash
python3 scripts/fusion_lab.py run --config experiments/fusion/configs/baseline.yaml \
  --sequence up --mode all
```

SGBM 与模型视差是不同实验条件，精度结论不可直接混合。

## 4. 重新运行原始模型

模型权重由实验提供者另行提供。`results/models/manifest.json` 记录原文件 SHA256、张量形状和输入输出约定，`disp_cost_volume.verified.json` 记录派生图及子图验证。将 `disp.onnx` 放入 `models/`，按 [LOCAL_SETUP.md](LOCAL_SETUP.md) 准备 GPU 依赖，再执行：

```bash
bash scripts/build_deps.sh gpu
bash scripts/build.sh gpu
bash scripts/build_models.sh disp
python3 scripts/fusion_lab.py run --sequence all --mode all
```

参考机是 RTX 4060 Laptop 8 GB、CUDA 12.9、TensorRT 10.14.1、CUDA OpenCV 4.12。`.plan` 在目标机器本地构建。RAFT 不参与本次滤波实验；完整后端和轨迹验证不属于已有结果。

## 版本与结果身份

主源码是发布时已修复的版本。`results/provenance/workspace.json` 保存原 SIO_SLAM HEAD、工作区差异和导出统计；实际实验以每次 `resolved.yaml` 的 `source_hashes`、输入和程序 SHA256 为准。旧记录中的 `git_commit` 只是当时的基点，工作区当时包含未提交修改。

所有历史记录保留原始字节和路径。`results/SHA256SUMS` 同时覆盖代码、数据摘要、模型视差、报告和历史源码；新增实验与重建报告不会覆盖它们。
