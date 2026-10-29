# SIO_SLAM 编译依赖说明

本文档根据 [`CMakeLists.txt`](../CMakeLists.txt)、[`package.xml`](../package.xml) 与项目源码整理，说明从零配置开发环境到成功编译本项目的全部依赖、构建选项与常见排错方法。

## 1. 概述

SIO_SLAM 是一个 **ROS 2 ament_cmake** 包（包名 `sio_slam`），使用 **C++17** 编写。推荐通过 **colcon** 在工作空间根目录构建。

构建产物分为三类目标，依赖层级不同：

| 目标 | 类型 | 主要用途 |
|------|------|----------|
| `run_frontend` | 完整运行时 | 前端 + 后端主程序（含 3DGS 光栅化） |
| `run_demo` | 运行时（无 3DGS 静态库） | 演示 / 稠密视觉链路 |
| `sio_unit_tests` | 轻量单测 | 不链接 TensorRT / CUDA 运行库；`SIO_BUILD_RUNTIME=OFF` 时可独立构建 |
| `pipeline_smoke_test` | 集成测试 | 与 `run_demo` 同级运行时依赖 |

> **重要**：默认 `SIO_BUILD_RUNTIME=ON`，配置阶段会检查 TensorRT、CUDA、CUDA OpenCV、small_gicp 等依赖。仅构建 CPU 单元测试时设置 `SIO_BUILD_RUNTIME=OFF`，仍需 ROS、OpenCV、PCL、Eigen、yaml-cpp 和 small_gicp。本机脚本见 [LOCAL_SETUP.md](LOCAL_SETUP.md)。

---

## 2. 基础环境

| 项目 | 要求 | 说明 |
|------|------|------|
| 操作系统 | Linux（推荐 Ubuntu 22.04 / 24.04） | 项目面向 Linux + NVIDIA GPU 环境 |
| 编译器 | GCC / Clang，支持 **C++17** | `CMAKE_CXX_STANDARD 17` |
| CMake | ≥ **3.10**（主项目）；子模块 3DGS 光栅化器要求 ≥ **3.20** | 主项目与子模块分别配置 |
| ROS 2 | 已 source 的 ROS 2 发行版（如 **Jazzy** / **Humble**） | 需 `ament_cmake`、`rclcpp` 等 |
| NVIDIA GPU | 支持 CUDA 的显卡 | 视差推理、光流、3DGS 光栅化均依赖 GPU |
| 构建工具 | `colcon`（推荐）或手动 `cmake` + `make` | README 推荐 colcon |

---

## 3. 依赖总览

### 3.1 硬依赖（缺失则 CMake 配置失败）

| 依赖 | 查找方式 | 使用模块 |
|------|----------|----------|
| **ament_cmake** | `find_package(ament_cmake)` | ROS 2 构建系统 |
| **rclcpp** | `ament_target_dependencies` | 日志、节点、参数 |
| **sensor_msgs** | 同上 | IMU / 图像消息 |
| **geometry_msgs** | 同上 | 几何消息 |
| **pcl_conversions** | 同上 | PCL ↔ ROS 消息转换 |
| **OpenCV** | `find_package(OpenCV)` | 图像处理、预处理；**运行时链路还需 CUDA 模块**（`opencv2/cuda*`） |
| **PCL** | `find_package(PCL)` | 点云、KD-Tree、法线估计 |
| **Eigen3** | `find_package(Eigen3)` | 矩阵、位姿、滤波 |
| **yaml-cpp** | 手动 `find_path` / `find_library` | 配置解析（`image_preprocessor`、`project_config`） |
| **CUDA Runtime** (`libcudart`) | `find_package(CUDAToolkit)` 或手动查找 | `TrtEngine`、`DynamicCloudProcessor` GPU 预处理 |
| **TensorRT** (`libnvinfer`) | 手动查找 `NvInfer.h` | 视差 / 光流 TensorRT 推理 |
| **small_gicp** | `find_package(small_gicp)` 或 `find_library` | `DynamicCloudProcessor`、`ColorICP` 配准 |
| **project_config.yaml** | 构建时文件存在性检查 | 编译期注入 `SIO_SLAM_PROJECT_CONFIG_PATH` |
| **color_icp 内嵌 yaml** | 文件存在性检查 | `include/color_icp/yaml.h`、`src/yaml.cpp` |

### 3.2 仅 `run_frontend` 额外需要

| 依赖 | 说明 |
|------|------|
| **CUDA Toolkit**（完整） | `find_package(CUDAToolkit REQUIRED)`，用于链接 3DGS 光栅化 |
| **libCudaRasterizer.a** | 需预先编译子模块 `submodules/diff-gaussian-rasterization-w-pose-main` |
| **GLM**（子模块自带） | 头文件位于 `third_party/glm`，随子模块初始化 |

### 3.3 软依赖（可选）

| 依赖 | 说明 |
|------|------|
| **OpenMP** | `find_package(OpenMP)`；未找到时 `ColorPointObserver` 串行运行 |
| **nvinfer_plugin** | TensorRT 插件库；找到则链接，未找到不报错 |
| **LibTorch** | 通过 `SIO_ENABLE_LIBTORCH=ON` 查找和链接；启用 `SIO_USE_LIBTORCH` 宏。完整后端联合优化需要开启，本机 GPU 构建脚本默认开启 |

### 3.4 运行时资源（非编译依赖，但运行必需）

| 资源 | 配置位置 | 说明 |
|------|----------|------|
| 视差 TensorRT Engine | `project_config.yaml` → `plan_disp` | 默认 `foundation_stereo_896x448.plan` |
| 光流 TensorRT Engine | `project_config.yaml` → `plan_raft` | 默认 `raft_large_896x448_iter6.plan` |
| 数据集与 IMU | `dataset_dir`、`imu_file` | 数据读取 |
| 算法参数 | `params_file`（默认 `config/params.yaml`） | 内外参、噪声、关键帧阈值等 |

---

## 4. ROS 2 包依赖

[`package.xml`](../package.xml) 声明的依赖：

```xml
buildtool_depend>ament_cmake</buildtool_depend>
buildtool_depend>ament_cmake_gtest</buildtool_depend>

depend>rclcpp</depend>
depend>sensor_msgs</depend>
depend>geometry_msgs</depend>
depend>pcl_conversions</depend>

test_depend>ament_cmake_gtest</test_depend>
```

构建前需已安装并 source 对应 ROS 2 发行版，例如：

```bash
source /opt/ros/jazzy/setup.bash   # 按本机发行版替换
```

---

## 5. 系统级依赖安装参考

以下命令以 **Ubuntu + ROS 2** 为例，包名随发行版略有差异，请按本机 `apt` 搜索结果调整。

### 5.1 基础构建与 ROS 依赖

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake git \
  libeigen3-dev \
  libyaml-cpp-dev \
  libomp-dev
```

ROS 2 相关包通常随桌面版或 `ros-dev-tools` 安装，需包含：

- `ros-<distro>-rclcpp`
- `ros-<distro>-sensor-msgs`
- `ros-<distro>-geometry-msgs`
- `ros-<distro>-pcl-conversions`
- `ros-<distro>-ament-cmake-gtest`（启用测试时）

### 5.2 OpenCV

```bash
sudo apt install -y libopencv-dev
```

**注意**：`dynamic_cloud_processor.cpp` 使用 `cv::cuda::*`（`cudaimgproc`、`cudawarping` 等），需要 **带 CUDA 模块的 OpenCV 构建**。若系统 `libopencv-dev` 不含 CUDA，需自行编译 OpenCV（`-DWITH_CUDA=ON`）或使用已集成 CUDA 的预编译包。

### 5.3 PCL

```bash
sudo apt install -y libpcl-dev
```

PCL 间接依赖 FLANN。若编译 `color_icp.cpp` 时出现 `std::unordered_map::serialize` 相关错误，属于 FLANN 与 C++17 兼容性问题，项目已在 `color_icp.cpp` 中通过宏规避，一般无需额外操作。

---

## 6. NVIDIA 相关依赖

### 6.1 CUDA Toolkit

- **最低需求**：CUDA Runtime（`libcudart`，头文件 `cuda_runtime_api.h`）
- **`run_frontend` 额外需求**：完整 CUDA Toolkit（编译并链接子模块光栅化器）

推荐安装与显卡驱动匹配的 [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)，并确保：

```bash
nvcc --version
nvidia-smi
```

CMake 搜索路径包括：

- `/usr/local/cuda/include`、`/usr/local/cuda/lib64`
- 环境变量 `CUDA_ROOT`、`CUDA_PATH`
- CMake 缓存 `-DCUDA_TOOLKIT_ROOT_DIR=<cuda_root>`

### 6.2 TensorRT

CMake 在配置阶段必须找到：

- 头文件：`NvInfer.h`
- 库：`libnvinfer.so`（可选 `libnvinfer_plugin.so`）

**配置方式（任选其一，优先级从高到低）：**

#### 方式 A：本机覆盖文件（推荐）

在工作空间根目录创建 `project_config.local.yaml`（已在 `.gitignore` 中，不提交）：

```yaml
tensorrt_root: /path/to/TensorRT-10.x.x.x
nvinfer_include_dir: /path/to/TensorRT-10.x.x.x/include
nvinfer_library_dir: /path/to/TensorRT-10.x.x.x/lib
```

#### 方式 B：CMake 缓存变量

```bash
colcon build --packages-select sio_slam --cmake-args \
  -DTENSORRT_ROOT_HINT=/path/to/TensorRT-10.x.x.x
```

或分别指定：

```bash
-DNVINFER_INCLUDE_HINT=/path/to/include \
-DNVINFER_LIBRARY_HINT=/path/to/lib
```

#### 方式 C：环境变量

```bash
export TENSORRT_ROOT=/path/to/TensorRT
# 或
export TRT_ROOT=/path/to/TensorRT
```

#### 其他搜索路径

- `/usr/local/TensorRT`
- `/usr/include`、`/usr/lib/x86_64-linux-gnu`
- CMake 内遗留路径（仅作兜底，勿依赖）

---

## 7. small_gicp

CMake 按以下顺序解析链接目标：

1. `small_gicp::small_gicp`（imported target）
2. `small_gicp`（裸 target）
3. `find_library(small_gicp)`

若均未找到，配置失败并报错：

```
small_gicp library not found for linking.
```

通常需要从源码编译安装 [koide3/small_gicp](https://github.com/koide3/small_gicp)，并确保 CMake 能找到其 config 或库文件（例如安装到 `/usr/local` 或将 `CMAKE_PREFIX_PATH` 指向安装前缀）。

使用位置：

- `src/dynamic_cloud_processor.cpp` — GICP 位姿估计
- `modules/registration/src/color_icp.cpp` / `include/color_icp.h` — robotic-front main 配准与兼容头；版本见 `modules/registration/UPSTREAM.json`

---

## 8. 子模块：diff-gaussian-rasterization

仅 **`run_frontend`** 链接此静态库，默认位置为：

```
submodules/diff-gaussian-rasterization-w-pose-main/build/libCudaRasterizer.a
```

可通过 `SIO_CUDA_RASTERIZER_LIBRARY` 指定其他位置；本机脚本使用 `build/dependencies/build/rasterizer/libCudaRasterizer.a`。

### 8.1 初始化子模块

```bash
cd /path/to/SIO_SLAM
git submodule update --init --recursive
```

子模块还依赖 `third_party/glm`（通过 `.gitmodules` 拉取）。

### 8.2 预先编译光栅化器

```bash
cd submodules/diff-gaussian-rasterization-w-pose-main
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON
make -j$(nproc)
```

要求：

- CMake ≥ 3.20
- 支持 CUDA 的编译器（`nvcc`）
- C++17 / CUDA 17 标准

完成后应存在：

```
submodules/diff-gaussian-rasterization-w-pose-main/build/libCudaRasterizer.a
```

若该文件缺失，链接 `run_frontend` 会失败。

---

## 9. 仓库内嵌文件检查

CMake 配置阶段会检查以下文件；缺失则 **立即 FATAL_ERROR**：

| 文件 | 用途 |
|------|------|
| [`project_config.yaml`](../project_config.yaml) | 顶层运行配置（数据集、模型路径、默认内参） |
| `include/color_icp/yaml.h` | Color ICP 配置解析（来自 [hanzheteng/color_icp](https://github.com/hanzheteng/color_icp)） |
| `src/yaml.cpp` | 同上 |

`project_config.local.yaml` 为可选本机覆盖，不参与存在性检查。

---

## 10. CMake 构建选项

| 选项 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `SIO_ENABLE_PROFILING` | `option` | `OFF` | 开启前端性能剖析埋点 |
| `SIO_BUILD_RUNTIME` | `option` | `ON` | 构建 GPU 主程序与集成测试；关闭后仅生成 CPU 测试 |
| `SIO_ENABLE_LIBTORCH` | `option` | `OFF` | 启用 LibTorch 后端优化，需提供 `Torch_DIR` 或 `CMAKE_PREFIX_PATH` |
| `SIO_CUDA_RASTERIZER_LIBRARY` | `CACHE FILEPATH` | 子模块下的静态库路径 | 预编译光栅化器路径 |
| `BUILD_TESTING` | CTest | colcon 默认 | 生成 `sio_unit_tests` 与 `pipeline_smoke_test` |
| `TENSORRT_ROOT_HINT` | `CACHE PATH` | 空 | TensorRT 根目录 |
| `NVINFER_INCLUDE_HINT` | `CACHE PATH` | 空 | `NvInfer.h` 所在目录 |
| `NVINFER_LIBRARY_HINT` | `CACHE PATH` | 空 | `libnvinfer` 所在目录 |
| `CUDA_TOOLKIT_ROOT_DIR` | `CACHE PATH` | 空 | CUDA 根目录 |

编译宏：

| 宏 | 条件 | 作用 |
|----|------|------|
| `SIO_ENABLE_PROFILING` | `-DSIO_ENABLE_PROFILING=ON` | 启用 profiler |
| `SIO_SLAM_PROJECT_CONFIG_PATH` | 始终定义 | 指向 `project_config.yaml` 绝对路径 |
| `SIO_USE_LIBTORCH` | `SIO_ENABLE_LIBTORCH=ON` | 启用 LibTorch 后端优化 |

---

## 11. 推荐构建流程

### 11.1 首次完整构建

```bash
# 1. 环境
source /opt/ros/<distro>/setup.bash

# 2. 子模块与 3DGS 光栅化器（仅需 run_frontend 时）
git submodule update --init --recursive
cd submodules/diff-gaussian-rasterization-w-pose-main
mkdir -p build && cd build
cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON
make -j$(nproc)
cd /path/to/SIO_SLAM

# 3. （可选）本机 TensorRT 路径
cat > project_config.local.yaml <<'EOF'
tensorrt_root: /path/to/TensorRT-10.x.x.x
EOF

# 4. 主项目构建
colcon build --packages-select sio_slam --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  -DTENSORRT_ROOT_HINT=/path/to/TensorRT-10.x.x.x

# 5. 加载环境
source install/setup.bash
```

### 11.2 开启 Profiling

```bash
colcon build --packages-select sio_slam --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  -DSIO_ENABLE_PROFILING=ON
```

### 11.3 运行测试

```bash
colcon test --packages-select sio_slam --event-handlers console_direct+
colcon test-result --verbose
```

---

## 12. 构建目标与链接关系

```
┌─────────────────────────────────────────────────────────────┐
│  公共轻量依赖（sio_configure_light_target）                    │
│  OpenCV, PCL, Eigen, yaml-cpp, OpenMP(可选)                  │
│  rclcpp, sensor_msgs, geometry_msgs, pcl_conversions         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  运行时扩展（sio_configure_runtime_target）                   │
│  + small_gicp, libnvinfer, libcudart, nvinfer_plugin(可选)   │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
     run_demo / pipeline_smoke_test    run_frontend
                                       + libCudaRasterizer.a
                                       + CUDAToolkit (REQUIRED)
```

| 源文件分组 | 包含模块 | 用于目标 |
|------------|----------|----------|
| `SIO_LIGHT_SRC_FILES` | dataset、feature、image_preprocessor、IMU、keyframe、state、yaml | `sio_unit_tests` |
| `SIO_RUNTIME_SRC_FILES` | 轻量 + color_icp + dynamic_cloud_processor | `run_demo`、`pipeline_smoke_test` |
| `SIO_FRONTEND_SRC_FILES` | 运行时 + frontend、backend、3DGS、torch_pose_optimize | `run_frontend` |

---

## 13. colcon 工作空间目录

| 目录 | 是否提交 Git | 说明 |
|------|--------------|------|
| `build/` | 否（`.gitignore`） | CMake / 编译中间文件 |
| `install/` | 否（`.gitignore`） | 安装产物；`source install/setup.bash` 后运行 |
| `log/` | 建议忽略 | colcon 构建日志 |

---

## 14. 常见配置错误

| 报错 | 原因 | 处理 |
|------|------|------|
| `Missing project_config.yaml` | 顶层配置缺失 | 从仓库恢复或参照示例创建 |
| `NvInfer.h not found` | TensorRT 未安装或路径未告知 CMake | 配置 `project_config.local.yaml` 或 `-DTENSORRT_ROOT_HINT` |
| `libnvinfer not found` | TensorRT 库路径错误 | 设置 `nvinfer_library_dir` 或 `NVINFER_LIBRARY_HINT` |
| `CUDA runtime library (cudart) not found` | 未安装 CUDA 或未加入 `LD_LIBRARY_PATH` | 安装 CUDA Toolkit，设置 `CUDA_TOOLKIT_ROOT_DIR` |
| `small_gicp library not found` | 未编译安装 small_gicp | 源码编译并加入 `CMAKE_PREFIX_PATH` |
| `yaml-cpp not found` | 缺少开发包 | `sudo apt install libyaml-cpp-dev` |
| 链接 `libCudaRasterizer.a` 失败 | 子模块未预编译 | 见第 8 节 |
| `TensorRT plan file not found`（运行时） | `plan_disp` / `plan_raft` 路径无效 | 修改 `project_config.yaml` 中模型路径 |
| OpenCV `cv::cuda` 相关链接错误 | OpenCV 未启用 CUDA | 换用带 CUDA 的 OpenCV 构建 |

---

## 15. 依赖检查清单

构建前可逐项确认：

- [ ] ROS 2 已 source，`colcon` 可用
- [ ] `g++` 支持 C++17
- [ ] `libopencv-dev`（含 CUDA 模块）、`libpcl-dev`、`libeigen3-dev`、`libyaml-cpp-dev` 已安装
- [ ] NVIDIA 驱动 + CUDA Toolkit 正常（`nvidia-smi`、`nvcc`）
- [ ] TensorRT 头文件与库可被 CMake 找到
- [ ] `small_gicp` 已安装且可链接
- [ ] `project_config.yaml` 存在且字段已按本机修改
- [ ] `include/color_icp/yaml.h` 与 `src/yaml.cpp` 存在
- [ ] （运行 `run_frontend`）子模块 `libCudaRasterizer.a` 已编译
- [ ] （运行主程序）`plan_disp`、`plan_raft` 模型文件路径有效

---

## 16. 相关文档

- [README.md](../README.md) — 项目概览与快速构建
- [project_config.yaml](../project_config.yaml) — 运行时顶层配置
- [config/params.yaml](../config/params.yaml) — 算法参数
- [IMAGE_PREPROCESSOR.md](IMAGE_PREPROCESSOR.md) — 图像预处理模块
