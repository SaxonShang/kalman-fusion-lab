# ImagePreprocessor 图像预处理说明

## 概述

`ImagePreprocessor` 为双目 SLAM 提供**光度域**自适应增强：在 YCrCb 的 **Y 通道**上按场景亮度做提亮或压高光；可选对 Cr/Cb 做**自适应饱和度补偿**（默认开启，可关闭）。**Y 增强是视差/深度重建的核心**；饱和度调节为可选，主要改善点云着色与观感，对几何重建非必需。

几何变换（中心裁减、缩放到推理分辨率）由 [`sio_stereo_geometry`](sio_stereo_geometry.h) 与 `DatasetLoader` 负责，本模块**不处理**裁减/缩放/内参。

| 模块 | 职责 |
|------|------|
| `DatasetLoader` + `sio_stereo_geometry` | 1280×720 → 中心裁减 1280×640 |
| `ImagePreprocessor` | Y 通道自适应光照（全分辨率） |
| `DynamicCloudProcessor` | 896×448 推理缩放 + 视差/光流 |

## 管线位置

```
读入双目图 (DatasetLoader，已中心裁减)
        ↓
ProcessStereoAdaptiveLighting   ← 每帧一次，写回 imgl/imgr
        ↓
视差网络 / RAFT / 点云 / 关键帧存图 / 后端 3DGS
        ↓
preprocessStereoImageCpu        ← 仅几何缩放到 896×448（不再改亮度）
```

**接入入口**

- [`src/frontend.cpp`](../src/frontend.cpp)：`run_frontend` 主流程
- [`apps/run_demo.cpp`](../apps/run_demo.cpp)：演示程序

启动时从 `params.yaml` 加载配置：

```cpp
ImagePreprocessor image_preprocessor;
image_preprocessor.LoadConfig(params_cfg_path.string());
// 每帧
auto [pre_left, pre_right] =
  image_preprocessor.ProcessStereoAdaptiveLighting(frame_data->imgl, frame_data->imgr);
```

## API

| 方法 | 说明 |
|------|------|
| `LoadConfig(path)` | 从 YAML 读取 `image_preprocessor.image_preprocess` |
| `ProcessStereoAdaptiveLighting(left, right)` | **主流程**：左右共用 gain/gamma/场景分类 |
| `ProcessAdaptiveLighting(image)` | 单目封装，独立计算亮度与增益 |

`image_preprocess.enable: false` 时两接口均原图直通。

## 处理流程

### 1. 亮度统计（仅用于决策，在 Y 通道）

- 色彩空间：BGR → YCrCb，取 Y
- 排除阴影（Y &lt; `shadow_exclude_below`）与饱和（Y &gt; `highlight_exclude_above`）后取 **midtone 中位数**
- 全图 Y 的 **P99** 作为高光分位 `highlight_p99`

### 2. 场景分类（左右目平均 raw median + max P99）

| 场景 | 条件 | 处理 |
|------|------|------|
| **暗** | raw median Y &lt; `dark_threshold` (70) | Y 去噪 → 线性 gain → auto gamma |
| **正常** | 介于暗/亮阈值之间 | 直通 |
| **亮** | raw median &gt; `bright_threshold` (170) 或 P99 ≥ `highlight_ceiling` (245) | Y 软 rolloff + `bright_gamma` |

### 3. 暗场景

1. **高光感知 gain**（基于左右平均 midtone）  
   `gain = min(target_luminance / midtone, gain_max)`  
   若 P99 过高：`gain ≤ highlight_ceiling / highlight_p99`

2. **去噪**（可选）：仅当 gain &gt; 1，对 Y 做 3×3 高斯（σ=`denoise_sigma`）

3. **线性提亮**：Y × gain，clip 到 255

4. **自动 gamma**：midtone 在 [`gamma_midtone_dark`, `target_luminance`] 线性映射到 [`gamma_min`, 1.0]

5. **可选饱和度补偿**（`saturation_enable`）：Y 增强后 Cr/Cb 相对发灰，按实际 Y gain 线性插值放大色度：
   ```
   sat = 1 + (saturation_gain - 1) × (y_gain - 1) / (gain_max - 1)
   ```
   默认 `saturation_gain: 1.5`（Y gain 达 `gain_max` 时的上限）。仅暗场景且 gain>1 时生效（`saturation_only_dark`）。

### 4. 亮场景

对 Y 应用分段 LUT（左右共用）：

```
Y ≤ highlight_knee (220)  : 不变
Y > 220                   : Y' = 220 + highlight_slope × (Y - 220)
```

若 `bright_gamma > 1`，再对 Y 做 gamma 校正进一步压高光。

### 5. 双目同步

左右目分别统计亮度，但共用：

- 场景分类（平均 raw median、最大 P99）
- `shared_gain`（平均 midtone）
- `gamma_midtone`（平均 midtone）
- 亮场景同一套 LUT

保证立体匹配的光度一致性。

## 配置

配置文件：[`config/params.yaml`](../config/params.yaml) → `image_preprocessor.image_preprocess`

```yaml
image_preprocessor:
  image_preprocess:
    enable: true
    dark_threshold: 70.0
    bright_threshold: 170.0
    luminance_percentile: 50.0
    target_luminance: 70.0
    gain_max: 1.8
    denoise: true
    denoise_ksize: 3
    denoise_sigma: 0.8
    auto_gamma: true
    gamma_min: 0.55
    gamma_midtone_dark: 15.0
    highlight_aware: true
    shadow_exclude_below: 10.0
    highlight_exclude_above: 220.0
    highlight_ceiling: 245.0
    highlight_percentile: 99.0
    highlight_knee: 220
    highlight_slope: 0.45
    bright_gamma: 1.15
    saturation_enable: true
    saturation_gain: 1.5
    saturation_only_dark: true
```

### 参数说明

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `enable` | true | 总开关 |
| `dark_threshold` | 70 | 暗场景 raw median Y 阈值 |
| `bright_threshold` | 170 | 亮场景 raw median Y 阈值 |
| `luminance_percentile` | 50 | 亮度统计分位（中位数） |
| `target_luminance` | 70 | 暗场景提亮目标 midtone |
| `gain_max` | 1.8 | 最大线性增益 |
| `denoise` / `denoise_ksize` / `denoise_sigma` | true / 3 / 0.8 | 暗场景 Y 去噪 |
| `auto_gamma` / `gamma_min` / `gamma_midtone_dark` | true / 0.55 / 15 | 暗场景自动 gamma |
| `highlight_aware` | true | 按 P99 限制 gain |
| `shadow_exclude_below` | 10 | 统计时排除的阴影 Y |
| `highlight_exclude_above` | 220 | 统计时排除的饱和 Y |
| `highlight_ceiling` | 245 | 高光感知 gain 参考上限 |
| `highlight_percentile` | 99 | 高光分位 |
| `highlight_knee` | 220 | 亮场景软压缩起点 |
| `highlight_slope` | 0.45 | 亮场景软压缩斜率 |
| `bright_gamma` | 1.15 | 亮场景额外压高光（≤1 关闭） |
| `saturation_enable` | true | 是否做 Cr/Cb 饱和度补偿（纯重建可关） |
| `saturation_gain` | 1.5 | Y gain=`gain_max` 时的色度增益上限 |
| `saturation_only_dark` | true | 仅暗场景且 gain>1 时做 sat |

关闭预处理：`image_preprocess.enable: false`。纯几何重建、不关心点云颜色时可设 `saturation_enable: false`，只保留 Y 增强。

## 效果可视化

亮场景 rolloff + gamma 曲线与合成对比图：

```bash
conda activate foundation_stereo
python scripts/visualize_bright_rolloff.py -o build/bright_rolloff_demo.png
```

## 设计约束（SLAM）

- **全局变换**：Y 通道全局 gain/gamma/LUT；sat 为 Cr/Cb 全局线性缩放，均不破坏极线几何
- **左右一致**：双目必须共用参数，避免立体不匹配
- **Y 优先**：视差/深度主要依赖 Y 对比度；sat 为可选下游颜色优化
- **不做 CLAHE/局部均衡**：局部增强易引入伪视差

### 饱和度与重建的关系

| 环节 | sat 影响 |
|------|----------|
| 视差 / 深度 | 间接、通常很小（匹配主要靠亮度梯度） |
| 点云 RGB | 直接（着色更自然） |
| ColorICP（`color_icp_lambda≈0.9`） | 几乎可忽略 |
| 3DGS 光度项 | 中等 |

验证 sat 是否影响几何时，应看有效视差比例、GICP 位姿等，**不要用 HSV-S mean**（Y 提亮后该指标会失真）。

## 源码与测试

| 路径 | 说明 |
|------|------|
| [`include/image_preprocessor.h`](../include/image_preprocessor.h) | 类与配置定义 |
| [`src/image_preprocessor.cpp`](../src/image_preprocessor.cpp) | 实现 |
| [`tests/unit/test_image_preprocessor.cpp`](../tests/unit/test_image_preprocessor.cpp) | 单元测试 |

```bash
./sio_unit_tests --gtest_filter='ImagePreprocessorTest.*'
```

## 与几何预处理的分工

图像相关预处理分两阶段，勿混淆：

1. **光度**（本文）：`ImagePreprocessor`，在裁减后全分辨率图像上，每帧一次  
2. **几何**（[`docs` 中 `sio_stereo_geometry`](../src/sio_stereo_geometry.cpp)）：`ComputeCenterCropRect` / `CenterCropAndResize`，在 `DynamicCloudProcessor` 入口缩放到网络输入尺寸  

内参同步见 `LoadInferenceCameraIntrinsics` / `LoadCroppedCameraIntrinsics`（[`sio_config.cpp`](../src/sio_config.cpp)），与 `ImagePreprocessor` 无关。
