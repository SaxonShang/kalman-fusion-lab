# SIO_SLAM

当前前端融合测试入口见 [experiments/fusion/README.md](experiments/fusion/README.md)：独立检查 GICP 及观测协方差、固定观测 ESKF 回放和关键帧 ESIKF，支持本机 FR5 静态场景数据与 IDE 断点调试。

实验介绍和目前最好的结果见 [README.md](README.md)；实验过程、对照、全部数值和运行方法见 [实验记录.md](实验记录.md)。

已接入本机 `models/disp.onnx` 与 RAFT 模型。融合实验默认先导出模型视差缓存，再独立测试滤波；SGBM 保留为显式对照配置。使用方法与模型转换说明见 [本机配置](docs/LOCAL_SETUP.md#模型与实验数据)。

GICP 已采用 `Ben-cpy/robotic-front` 的 main 提交 `744ac2d`；源码、版本记录和滤波接口适配见 [modules/registration/README.md](modules/registration/README.md)。

初始协方差与 ESKF 数值求解的修复、固定观测对照和 P/R 敏感性报告见 [实验记录.md 第 4 节](实验记录.md#4-2026-10-29-初始-p-与-eskf-求解修复)。

独立卡尔曼实验仓库的整理、导出和首次上传步骤见 [PUBLISH_NEW_REPOSITORY.md](docs/PUBLISH_NEW_REPOSITORY.md)。新仓库使用面向实验的首页，并包含历史结果与跨机器复现入口；原工作区的远程配置保持不变。

## 项目简介

SIO_SLAM 是一个面向双目视觉与 IMU 融合的 SLAM 研究工程。项目当前由两条主线组成：

- 前端：以双目图像和 IMU 为输入，完成深度恢复、动态点剔除、GICP + ESKF 松耦合更新，以及关键帧上的几何-光度 ESIKF 紧耦合优化。
- 后端：以关键帧、IMU 预积分和 3D Gaussian Splatting 为基础，在滑窗内执行光度、深度与惯性的联合优化，用于提升位姿精度并维护高斯地图。

项目整体更接近“研究原型 + 论文工程化实现”，因此代码中既包含日常运行入口，也包含方法说明文档、论文拆分框架和若干实验性模块。

## 系统目标

本项目主要解决以下问题：

1. 在双目 + IMU 输入下稳定估计载体位姿。
2. 在动态场景中通过观测筛选降低运动物体干扰。
3. 在关键帧阶段利用几何和光度联合残差提升局部配准精度。
4. 在后端通过 3DGS 与 IMU 融合进一步提升位姿和地图质量。

## 总体架构

按数据流，系统可概括为：

```text
传感器 / ROS 节点 (SVIO_SLAM)
        │
        ▼
SIOFrontend (frontend.h / frontend.cpp)
  ├─ ImagePreprocessor / ImuPreprocessor
  ├─ DynamicCloudProcessor     双目视差 → 点云 → 动态滤除
  ├─ FeatureProcessor          法向 / 颜色梯度 / 立体点协方差
  ├─ registration             普通帧配准（默认 GICP，可显式选择 Stereo-GICP）
  ├─ ImuIntegrator             初始化 + 预积分
  ├─ StateEstimator            关键帧 ESIKF / 普通帧 ESKF
  └─ KeyframeManager           前后端关键帧判定与创建
        │
        ▼
SIOBackend (backend.h / backend.cpp)   滑动窗口 3DGS + IMU 联合优化
        │
        ▼
后验回传 → Frontend.ApplyBackendStateUpdates
```

### 前端链路

前端主流程可概括为：

1. 读取双目图像与对应时间段 IMU 数据。
2. 图像/IMU 预处理后，通过视差恢复与点云生成得到当前帧稠密观测。
3. 基于 IMU 预积分获得当前帧先验状态。
4. 基于预测相对位姿进行动态点剔除。
5. 非关键帧：默认欧氏 GICP，可显式选择投影 Stereo-GICP；位姿与协方差作为 ESKF 观测，失败时保留 IMU 先验。
6. 关键帧：投影关联下的几何-光度 ESIKF 紧耦合更新，并推进关键帧管理、推送后端。

前端主控入口位于 [apps/run_frontend.cpp](apps/run_frontend.cpp)，编排实现位于 [src/frontend.cpp](src/frontend.cpp)；算法模块声明见 [include/sio_frontend_modules.h](include/sio_frontend_modules.h)、[include/color_icp.h](include/color_icp.h)、[include/stereo_gicp.h](include/stereo_gicp.h)。

### 后端链路

后端主要围绕滑窗优化与高斯地图维护展开：

1. 接收前端关键帧和 IMU 预积分结果。
2. 在滑窗内维护节点状态、观测图像与深度。
3. 通过 3DGS 渲染计算光度误差及其位姿梯度。
4. 结合 IMU 约束进行联合优化。
5. 更新高斯地图并执行必要的增密或裁剪，后验回传前端。

后端入口和核心实现主要位于 [src/backend.cpp](src/backend.cpp)、[src/gaussian_map_manager.cpp](src/gaussian_map_manager.cpp)、[src/pose_3dgs_optimizer.cpp](src/pose_3dgs_optimizer.cpp) 和 [src/torch_pose_optimize.cpp](src/torch_pose_optimize.cpp)。

## 目录说明

```text
.
├── apps/                 可执行程序入口（run_frontend、run_demo 等）
├── config/               运行参数示例与算法参数
├── docs/                 方法说明、模块文档、论文框架
├── include/              对外头文件与模块声明
│   ├── frontend.h / backend.h
│   ├── sio_frontend_modules.h   双目、特征、状态估计、关键帧
│   ├── stereo_gicp.h / color_icp.h
│   ├── imu_integrator.h / imu_preprocessor.h / image_preprocessor.h
│   └── data.h / sio_config.h / ...
├── src/                  主要实现代码（与 include 一一对应）
├── modules/registration/ robotic-front main 配准模块与上游版本记录
├── tests/                单元测试与集成测试
├── submodules/           第三方子模块（含 diff-gaussian-rasterization）
├── CMakeLists.txt        构建脚本
├── package.xml           ROS2/ament 包描述
└── project_config.yaml   顶层运行配置
```

## 核心模块对照

| 模块 | 主要文件 | 说明 |
| --- | --- | --- |
| 数据读取 | [src/dataset_loader.cpp](src/dataset_loader.cpp) | 读取图像序列与 IMU 数据 |
| 图像预处理 | [src/image_preprocessor.cpp](src/image_preprocessor.cpp) | 双目 Y 通道自适应光照增强 |
| IMU 预处理 | [src/imu_preprocessor.cpp](src/imu_preprocessor.cpp) | 可选低通与野值剔除 |
| IMU 预积分 | [src/imu_integator.cpp](src/imu_integator.cpp) | 静止初始化、位姿增量与协方差传播 |
| 稠密视觉处理 | [src/dynamic_cloud_processor.cpp](src/dynamic_cloud_processor.cpp) | 视差、光流、点云及动态点滤除 |
| 特征预处理 | [src/feature_processor.cpp](src/feature_processor.cpp) | 法向、颜色梯度、立体点协方差 |
| 投影 Stereo-GICP | [modules/registration/src/stereo_gicp.cpp](modules/registration/src/stereo_gicp.cpp) | 上游投影关联、立体 Σ、独立配准 |
| Color ICP / GICP | [modules/registration/src/color_icp.cpp](modules/registration/src/color_icp.cpp) | 上游 Colored ICP、欧氏 GICP 及 Stereo 入口 |
| 前端状态估计 | [src/state_estimater.cpp](src/state_estimater.cpp) | 投影/欧氏关联、ESIKF、ESKF |
| 前端主控 | [src/frontend.cpp](src/frontend.cpp) | 组织数据流、调度前端各模块 |
| 关键帧管理 | [src/keyframe_manager.cpp](src/keyframe_manager.cpp) | 前后端关键帧触发与创建 |
| 后端主控 | [src/backend.cpp](src/backend.cpp) | 滑窗管理、优化调度、状态回传 |
| 高斯地图 | [src/gaussian_map_manager.cpp](src/gaussian_map_manager.cpp) | 高斯地图维护、融合、稠密化 |
| 位姿优化 | [src/pose_3dgs_optimizer.cpp](src/pose_3dgs_optimizer.cpp) | 3DGS 相关位姿优化 |
| Torch 联合优化 | [src/torch_pose_optimize.cpp](src/torch_pose_optimize.cpp) | 位姿、IMU 与渲染的联合优化 |

## 环境依赖

根据当前 CMake 配置，项目默认依赖如下环境：

- Linux
- C++17
- ROS2 / ament_cmake
- OpenCV
- PCL
- Eigen3
- CUDA Runtime
- TensorRT
- small_gicp
- OpenMP（可选，但建议开启）

完整安装步骤、CMake 选项与排错说明见 [docs/BUILD_DEPENDENCIES.md](docs/BUILD_DEPENDENCIES.md)。

本机 Ubuntu 22.04 / RTX 4060 的目录约定、安装脚本和构建步骤见 [docs/LOCAL_SETUP.md](docs/LOCAL_SETUP.md)。

FR5 + ZED X 高速实验数据接入见 [docs/FR5_DATASET.md](docs/FR5_DATASET.md)。原始数据放入 `data/recordings/recordings_highspeed/` 后，执行 `python3 scripts/prepare_fr5_dataset.py --activate up` 生成兼容索引、时间补偿与相机标定，并选择本机运行序列。

默认构建会检查 TensorRT、CUDA 和 small_gicp。仅开发 CPU 算法模块时可使用 `-DSIO_BUILD_RUNTIME=OFF`，或执行 `bash scripts/build.sh cpu`。

## 运行前需要修改的配置

### 1. 顶层运行配置

[project_config.yaml](project_config.yaml) 是当前工程的硬依赖配置文件，至少需要检查这些字段：

- `dataset_dir`：数据集目录
- `imu_file`：IMU 数据文件路径
- `plan_disp`：双目视差 TensorRT engine 路径
- `plan_raft`：光流 TensorRT engine 路径
- `params_file`：算法参数文件路径

### 2. 本机 TensorRT 提示配置

如果本机 TensorRT 路径与默认搜索路径不同，建议新增或维护 `project_config.local.yaml`，用于覆盖：

- `tensorrt_root`
- `nvinfer_include_dir`
- `nvinfer_library_dir`

### 3. 算法参数

[config/params.yaml](config/params.yaml) 主要负责：

- 相机内参
- IMU 噪声参数
- 相机/IMU 外参
- 前后端关键帧阈值
- 追踪点数量与深度范围

## 构建方式

推荐使用 ROS2 的 `colcon` 进行构建。

```bash
colcon build --packages-select sio_slam --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

如果需要前端 profiling，可额外打开：

```bash
colcon build --packages-select sio_slam --cmake-args -DCMAKE_BUILD_TYPE=Release -DSIO_ENABLE_PROFILING=ON
```

本机首次构建可使用以下脚本，依赖与构建产物存放在工作区内：

```bash
bash scripts/install_system_deps.sh  # 本机终端输入 sudo 密码；完成后保存工作并重启
bash scripts/build_deps.sh gpu       # 普通用户执行
bash scripts/build.sh gpu
source env.sh
```

脚本会构建 3DGS 光栅化器，并开启 LibTorch 后端优化。模型和数据需要另外提供。

## 运行入口

### 前端主程序

```bash
ros2 run sio_slam run_frontend
```

### 运行演示入口

```bash
ros2 run sio_slam run_demo
```

## 测试

当前 CMake 中配置了以下测试入口：

- 单元测试目标：`sio_unit_tests`
- 集成测试目标：`pipeline_smoke_test`

推荐执行方式：

```bash
colcon test --packages-select sio_slam --event-handlers console_direct+
colcon test-result --verbose
```

## 文档地图

### 核心方法文档

- [docs/SIO_SLAM_前端说明.md](docs/SIO_SLAM_前端说明.md)：前端状态定义、坐标系、GICP/ESKF、ESIKF 与动态点剔除的完整说明。
- [docs/SIO_SLAM_后端说明.md](docs/SIO_SLAM_后端说明.md)：后端滑窗、3DGS 渲染、IMU 联合优化与高斯地图管理说明。

### 模块使用文档

- [docs/BUILD_DEPENDENCIES.md](docs/BUILD_DEPENDENCIES.md)：编译依赖、构建选项与环境配置说明。
- [docs/IMAGE_PREPROCESSOR.md](docs/IMAGE_PREPROCESSOR.md)：图像预处理模块说明和集成方式。
- [docs/LEARNING_RATE_SCHEDULER.md](docs/LEARNING_RATE_SCHEDULER.md)：学习率调度模块说明。

### 论文规划文档

- [docs/SCI_五篇论文索引.md](docs/SCI_五篇论文索引.md)：五篇 SCI 总索引（理论与雅可比在各篇 §3）。
- [docs/JOURNAL_RECOMMENDATIONS.md](docs/JOURNAL_RECOMMENDATIONS.md)：候选期刊 IF/JCI/WJCI 与 Top 标记、按论文方向的投稿建议。
- [docs/SCI_动态观测滤波.md](docs/SCI_动态观测滤波.md)：动态观测滤波。
- [docs/SCI_投影StereoGICP.md](docs/SCI_投影StereoGICP.md)：投影 Stereo-GICP。
- [docs/SCI_ESKF与ESIKF.md](docs/SCI_ESKF与ESIKF.md)：ESKF / ESIKF。
- [docs/SCI_3DGS_IMU后端.md](docs/SCI_3DGS_IMU后端.md)：以 IMU 运动先验稳定的 3DGS 位姿优化。
- [docs/SCI_系统集成.md](docs/SCI_系统集成.md)：系统集成与协同收益。

## 推荐阅读顺序

### 路径 A：第一次接触项目

适合先建立整体认知：

1. 本文档 [README.md](README.md)
2. [docs/SIO_SLAM_前端说明.md](docs/SIO_SLAM_前端说明.md)
3. [docs/SIO_SLAM_后端说明.md](docs/SIO_SLAM_后端说明.md)
4. [docs/IMAGE_PREPROCESSOR.md](docs/IMAGE_PREPROCESSOR.md)
5. [docs/LEARNING_RATE_SCHEDULER.md](docs/LEARNING_RATE_SCHEDULER.md)

### 路径 B：以工程跑通为目标

适合先把程序运行起来，再逐步深入：

1. 本文档 [README.md](README.md)
2. [project_config.yaml](project_config.yaml)
3. [config/params.yaml](config/params.yaml)
4. [apps/run_frontend.cpp](apps/run_frontend.cpp)
5. [src/frontend.cpp](src/frontend.cpp)
6. [src/dynamic_cloud_processor.cpp](src/dynamic_cloud_processor.cpp)
7. [src/state_estimater.cpp](src/state_estimater.cpp)

### 路径 C：以前端算法为主

适合重点理解状态估计与配准：

1. [docs/SIO_SLAM_前端说明.md](docs/SIO_SLAM_前端说明.md)
2. [docs/SCI_投影StereoGICP.md](docs/SCI_投影StereoGICP.md) / [docs/SCI_ESKF与ESIKF.md](docs/SCI_ESKF与ESIKF.md)
3. [src/frontend.cpp](src/frontend.cpp)
4. [modules/registration/README.md](modules/registration/README.md) 及其中的配准源码
5. [src/state_estimater.cpp](src/state_estimater.cpp)
6. [src/keyframe_manager.cpp](src/keyframe_manager.cpp)
7. [docs/IMAGE_PREPROCESSOR.md](docs/IMAGE_PREPROCESSOR.md)

### 路径 D：以后端与 3DGS 为主

适合重点理解滑窗和优化：

1. [docs/SIO_SLAM_后端说明.md](docs/SIO_SLAM_后端说明.md)
2. [src/backend.cpp](src/backend.cpp)
3. [src/gaussian_map_manager.cpp](src/gaussian_map_manager.cpp)
4. [src/pose_3dgs_optimizer.cpp](src/pose_3dgs_optimizer.cpp)
5. [src/torch_pose_optimize.cpp](src/torch_pose_optimize.cpp)
6. [docs/LEARNING_RATE_SCHEDULER.md](docs/LEARNING_RATE_SCHEDULER.md)

### 路径 E：以论文整理和写作为主

适合从工程抽象到论文结构：

1. [README.md](README.md)
2. [docs/SCI_五篇论文索引.md](docs/SCI_五篇论文索引.md)
3. [docs/SCI_动态观测滤波.md](docs/SCI_动态观测滤波.md)
4. [docs/SCI_投影StereoGICP.md](docs/SCI_投影StereoGICP.md)
5. [docs/SCI_ESKF与ESIKF.md](docs/SCI_ESKF与ESIKF.md)
6. [docs/SCI_3DGS_IMU后端.md](docs/SCI_3DGS_IMU后端.md)
7. [docs/SCI_系统集成.md](docs/SCI_系统集成.md)

## 建议的代码阅读顺序

如果你希望快速建立“代码调用关系”而不是先读公式，建议按下面顺序看：

1. [apps/run_frontend.cpp](apps/run_frontend.cpp)
2. [src/frontend.cpp](src/frontend.cpp)
3. [src/dataset_loader.cpp](src/dataset_loader.cpp)
4. [src/dynamic_cloud_processor.cpp](src/dynamic_cloud_processor.cpp)
5. [modules/registration/src/stereo_gicp.cpp](modules/registration/src/stereo_gicp.cpp) / [modules/registration/src/color_icp.cpp](modules/registration/src/color_icp.cpp)
6. [src/state_estimater.cpp](src/state_estimater.cpp)
7. [src/keyframe_manager.cpp](src/keyframe_manager.cpp)
8. [src/backend.cpp](src/backend.cpp)
9. [src/gaussian_map_manager.cpp](src/gaussian_map_manager.cpp)
10. [src/pose_3dgs_optimizer.cpp](src/pose_3dgs_optimizer.cpp)

## 当前阅读建议

如果你的目标是尽快把项目理解清楚，最省时间的顺序是：先看 [README.md](README.md)，再看 [docs/SIO_SLAM_前端说明.md](docs/SIO_SLAM_前端说明.md) 和 [docs/SIO_SLAM_后端说明.md](docs/SIO_SLAM_后端说明.md)，最后再回到 [src/frontend.cpp](src/frontend.cpp) 和 [src/backend.cpp](src/backend.cpp) 对照实现。
