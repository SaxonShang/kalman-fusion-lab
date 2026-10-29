# 本机工作区与环境

本机是 Ubuntu 22.04 x86_64，显卡为 RTX 4060。原工作区保留本地。正式实验仓库为 `SaxonShang/kalman-fusion-lab`，本机位于 `/home/ubuntu22/slam/SIO_SLAM/build/github-export/kalman-fusion-lab`；后续代码同步和结果上传使用此仓库，原仓库不推送。

2026-10-29：在新仓库代码 `b9804c7` 上完成 ESIKF E0–E7 三阶段共 1236 条记录，111 项单测通过。模型和依赖通过被 Git 忽略的本机符号链接复用原工作区，CPU 实验程序及 GPU 视差导出工具从新仓库源码重新编译。完整 SLAM 未在该提交上重跑。结果见 [本轮审阅包](../results/esikf/20261029_b9804c7/README.md)。

## 本机验证进度（2026-10-28）

当前优先测试卡尔曼前端融合，专用入口见 [前端融合测试工作区](../experiments/fusion/README.md)。IDE 默认构建目录已切换为 `build/local/fusion/`，它独立于完整 GPU 前端。

- CUDA 12.9、TensorRT 10.14.1 已安装；本地 CUDA OpenCV、LibTorch 和 3DGS 光栅化库已就绪。
- `bash scripts/build_deps.sh gpu` 和 `bash scripts/build.sh gpu` 已成功执行。
- `run_frontend`、`run_demo` 和集成测试目标已编译，70 项单元测试全部通过；两个主程序的共享库均可解析。
- 已补齐 OpenCV 的 `cudev` 模块，并修正 `torch_pose_optimize.cpp` 中三处 `options()` 接口调用。
- 重启后显卡已由 NVIDIA 驱动接管，已验证 RTX 4060 Laptop GPU、驱动 595.99.02、显存 8188 MiB。
- 实际 GPU 检查全部通过：CUDA 设备访问，OpenCV CUDA 颜色转换及通道拆分/合并，LibTorch CUDA 矩阵运算、反向传播及 Adam 更新，TensorRT 微型网络构建/反序列化/推理，3DGS CUDA 可见性内核。
- FR5 六条高速序列已提供；兼容索引、时间补偿和 ZED X 标定的准备方法见 [FR5_DATASET.md](FR5_DATASET.md)。
- 已用实际 C++ 加载器读取全部 3826 对保留图像与 23884 个 IMU 样本，六条序列静止初始化均通过；日志为 `build/fr5-loader-probe.log`。
- 已提供 `models/disp.onnx` 和 `models/raft_large_896x448_iter6.onnx`，分辨率均为 896×448；模型转换与实际推理验证状态见下方及融合实验记录。

2026-10-29：两个本机 TensorRT 引擎均构建成功，环境检查已识别两个 `.plan`。视差模型已在录制图像上产生有限的像素视差，并接入默认融合实验。RAFT 已验证六个输出张量及最后一轮 `2206` 可推理、输出有限；尚未进行真实光流精度评估。完整 SLAM 数据流程与后端仍属于后续验证范围。

日常环境检查：

```bash
cd /home/ubuntu22/slam/SIO_SLAM
source env.sh
nvidia-smi
bash scripts/check_env.sh gpu
```

成功构建日志：`build/gpu-dependencies-final.log`、`build/gpu-build.log`。实际 GPU 检查日志：`build/gpu-runtime-probe.log`；验证程序保存在被 Git 忽略的 `build/gpu-runtime-probe/` 中。

## 目录约定

```text
SIO_SLAM/
├── apps/、include/、src/、tests/     项目源码
├── config/                         共享算法配置
├── project_config.local.yaml       本机路径覆盖，Git 忽略
├── env.sh                          当前终端加载 ROS / CUDA / 本地依赖
├── scripts/                        安装、构建与环境检查
├── models/*.onnx                    本机用户提供的视差及光流模型
├── models/tensorrt/                 本机生成的 .plan
├── data/datasets/                   离线数据集
├── data/calibration/                相机与 IMU 标定文件
├── data/recordings/                 原始采集数据
├── artifacts/                      日志、轨迹和地图输出
├── build/dependencies/src/          第三方源码
├── build/dependencies/downloads/    下载缓存
├── build/dependencies/build/        第三方编译目录
├── build/dependencies/install/      第三方本地安装前缀
├── build/local/cpu/                 CPU 单元测试构建
├── build/local/gpu/                 完整 GPU 构建
└── install/local/                   项目本地安装目录
```

保留仓库原有 `build/` 中从其他机器带来的文件；新的构建使用 `build/local/`，避免旧 CMake 缓存中的绝对路径。模型、数据、依赖和产物目录均已加入 Git 忽略规则。

构建脚本更新根目录的 `compile_commands.json` 符号链接，clangd 从这里读取最近构建模式的编译参数。VS Code / Cursor 当前默认构建目录指向 `build/local/fusion/`，并提供 `Fusion:` 任务与调试入口。完整 GPU 构建使用终端脚本，产物保留在 `build/local/gpu/`。

## 环境版本

| 组件 | 配置 |
|---|---|
| ROS 2 | 已安装的 Humble |
| 编译器 | 系统 GCC 11 / C++17 |
| NVIDIA 驱动 | Ubuntu 推荐版本，已验证 595.99.02 |
| CUDA Toolkit | 12.9，安装到 `/usr/local/cuda-12.9` |
| TensorRT | `10.14.1.48-1+cuda12.9`，系统 deb 包 |
| OpenCV + contrib | 4.12.0，启用 CUDA，`CUDA_ARCH_BIN=8.9` |
| small_gicp | 固定提交 `fa0cfc983417c044b6036f71dfa448038bb9c820` |
| LibTorch | 2.8.0 / CUDA 12.9 / C++11 ABI |
| 3DGS 光栅化器 | 仓库自带源码，`CMAKE_CUDA_ARCHITECTURES=89` |

OpenCV、small_gicp 和 LibTorch 安装到工作区内，不替换系统库。默认最多使用 4 个构建进程，可通过 `SIO_BUILD_JOBS=2` 降低内存和 CPU 占用。

`nvidia-smi` 中显示的 CUDA 13.2 是驱动支持的 CUDA 版本；项目安装的 Toolkit 为 CUDA 12.9，以 `nvcc --version` 为准。两者配合已通过上述 GPU 运算验证。版本字段含义见 [NVIDIA 官方说明](https://docs.nvidia.com/deploy/nvidia-smi/index.html#cuda-version)。

## 系统安装与 GPU 构建

第一步需要在本机终端输入 sudo 密码：

```bash
cd /home/ubuntu22/slam/SIO_SLAM
bash scripts/install_system_deps.sh
```

脚本安装基础开发包、Ubuntu 推荐驱动、CUDA 12.9 和指定版本 TensorRT，不自动重启。GPU 依赖会下载数 GB 软件包，请等待安装完成。

如果基础包和驱动已安装，仅 CUDA / TensorRT 阶段失败，可从该阶段继续：

```bash
bash scripts/install_system_deps.sh --simulate-gpu       # 仅检查 APT 依赖解析，无需 sudo
bash scripts/install_system_deps.sh --gpu-packages-only  # 继续安装 CUDA / TensorRT
```

脚本为 TensorRT 的运行库、插件、解析器与头文件统一指定 `10.14.1.48-1+cuda12.9`，避免 APT 为间接依赖选取较新的 CUDA 13 软件包。

本机开启了 Secure Boot。若驱动安装提示创建 MOK 密码，请在重启后出现的蓝色 MOK 界面选择 Enroll MOK，按提示完成注册。无需关闭 Secure Boot。保存工作后自行重启，再验证：

```bash
nvidia-smi
```

第二步以普通用户构建本地依赖和项目：

```bash
cd /home/ubuntu22/slam/SIO_SLAM
bash scripts/build_deps.sh gpu
bash scripts/build.sh gpu
source env.sh
bash scripts/check_env.sh gpu
```

`build_deps.sh gpu` 会下载并编译 CUDA OpenCV、下载 LibTorch、编译 3DGS 光栅化器。首次编译较久，进度显示在终端。`build.sh gpu` 启用 `SIO_ENABLE_LIBTORCH=ON` 并运行单元测试；集成测试依赖模型和数据，不自动运行。

依赖构建成功后再执行项目构建；项目脚本会先检查 OpenCV、LibTorch 和光栅化库是否存在。CUDA OpenCV 的模块列表必须包含 contrib 中的 `cudev`，脚本已显式启用。

如果安装驱动后 `nvidia-smi` 仍无法连接，用 `lspci -k -s 01:00.0` 查看显卡实际使用的驱动。本机安装过程中曾仍显示 `Kernel driver in use: nouveau`，此时先保存工作并重启，再检查驱动是否切换成功。不要在图形桌面运行时强行卸载当前显示驱动。

后续每次打开终端只需执行 `source /home/ubuntu22/slam/SIO_SLAM/env.sh`。无需修改全局 `.bashrc`。

## 当前可先使用的 CPU 开发环境

驱动、CUDA 未安装时，也可构建和测试前端的 CPU 算法模块：

```bash
bash scripts/build_deps.sh cpu
bash scripts/build.sh cpu
bash scripts/check_env.sh cpu
```

此模式设置 `SIO_BUILD_RUNTIME=OFF`，不生成 GPU 主程序或集成测试。默认 CMake 行为仍为完整 GPU 构建。

## 模型与实验数据

本机路径由 `project_config.local.yaml` 覆盖。默认约定：

```text
models/tensorrt/disp_896x448_fp16.plan
models/tensorrt/raft_large_896x448_iter6.plan
data/datasets/fr5_highspeed/20261015_112553_up/imu.csv
```

FR5 原始数据位于 `data/recordings/recordings_highspeed/`。执行 `python3 scripts/prepare_fr5_dataset.py --activate up` 生成加载器兼容数据并设置本机路径；切换其他运动序列、时间与标定约定见 [FR5_DATASET.md](FR5_DATASET.md)。

ONNX 原文件保持在 `models/` 下；在本机 RTX 4060 上生成 TensorRT engine，原文件不改写。其他显卡或 TensorRT 版本生成的 engine 可能无法直接使用。

```bash
bash scripts/build_models.sh all
python3 scripts/fusion_lab.py run --sequence up --mode all
```

构建脚本使用固定 batch=1、FP32 输入输出、FP16 内部计算和默认 3 GiB TensorRT workspace 上限（不是总显存上限）。可通过 `SIO_TRT_WORKSPACE_MB` 和 `SIO_TRT_OPT_LEVEL` 指定工作区和优化级别。首次原图构建曾因 1 GiB workspace 不足而失败；4 GiB 重试出现总显存分配失败提示后已停止。日志分别在 `models/tensorrt/logs/disp-workspace1024-failed.log` 和 `disp-workspace4096-stopped.log`；成功引擎只在完成后写入正式路径，零字节 `.building` 不能用于推理。

原始视差导出含 312 个 `ScatterND`，TensorRT 将其融合为巨大的代价体节点。详细日志中该节点需要约 5.18 GiB（FP32）或 3.25 GiB（FP16）的临时空间；单纯增大 workspace 仍可能超过整机显存。因此默认先调用 [prepare_disp_onnx.py](../scripts/prepare_disp_onnx.py)，在保持原始更新张量、相关计算和精度转换边界的前提下，将逐平面写入改成 `Pad/Concat`。原文件不修改，派生文件位于 `models/onnx/disp_cost_volume.onnx`；只有通过 ONNX Runtime 原始子图/派生子图数值比较，才会生成 `.verified.json` 并用于构建。

本机独立工具环境为 `build/venv/model-tools/`，依赖固定在 [requirements-model-tools.txt](../requirements-model-tools.txt)，不改系统 Python。构建优化级别默认 0，优先完成模块验证；可用 `SIO_TRT_OPT_LEVEL` 调整。若要在其他具备足够资源的环境直接尝试原图，可显式设置 `SIO_DISP_ONNX=disp.onnx`。

派生代价体与原始子图在固定随机 FP16 特征输入上逐元素一致（最大误差 0）。图节点数从 48,651 减至 14,202；派生路径的 FP16 临时空间需求约 2.12 GiB，因此默认 workspace 设为 3 GiB。该比较验证代价体改写，不代替完整模型和数据集精度评估。

视差张量为 `left/right → disp`；RAFT 张量为 `img1/img2`，此导出包含六轮输出，最后一轮名称为 `2206`。完整前端已按名称选择最后一轮；独立融合测试只生成视差缓存，不启用 RAFT。原始模型与生成引擎均在 Git 忽略范围内。

## 官方参考

- [Ubuntu NVIDIA 驱动安装](https://ubuntu.com/server/docs/nvidia-drivers-installation/)
- [CUDA 12.9 Linux 安装指南](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-installation-guide-linux/index.html)
- [TensorRT 10.14.1 说明](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.14.1.html)
- [OpenCV 构建选项](https://docs.opencv.org/4.12.0/db/d05/tutorial_config_reference.html)
- [LibTorch 安装](https://docs.pytorch.org/cppdocs/installing.html)
