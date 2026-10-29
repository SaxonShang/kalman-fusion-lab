此目录归档模型说明、接口与身份校验；ONNX 权重和 .plan 需按复现指南在 models/ 中另行准备。

# 本机模型

- `disp.onnx`：双目视差，`left/right`，FP32 NCHW，batch 动态、RGB 0..255，固定空间大小 448×896；输出 `disp`，浮点像素视差。
- `raft_large_896x448_iter6.onnx`：RAFT 光流，`img1/img2`，FP32 NCHW 1×3×448×896，RGB -1..1；六轮输出依次为 `flow_list`、`1038`、`1330`、`1622`、`1914`、`2206`，使用最后一轮 `2206`。
- `tensorrt/disp_896x448_fp16.plan` 与 `tensorrt/raft_large_896x448_iter6.plan`：本机 RTX 4060、TensorRT 10.14.1 构建目标。源 ONNX SHA256 写入相邻 `.source-sha256` 文件，转换日志在 `tensorrt/logs/`。
- `onnx/disp_cost_volume.onnx`：为本机内存限制生成的派生图，原始 `disp.onnx` 不改写。保留原始特征/相关运算与 FP16/FP32 转换，仅将 312 次 ScatterND 逐平面写入换为 Pad/Concat；原始与派生代价体经过 ONNX Runtime 数值对比后才写入 `.verified.json`。原始与派生 SHA256、验证误差保存在该文件中。

在项目根目录运行 `bash scripts/build_models.sh all` 构建；可用 `disp` 或 `raft` 单独选择。视差模型 batch 固定为 1，I/O 保持 FP32，内部允许 FP16；workspace 默认上限 3 GiB，实际总显存还包括权重与激活。可通过 `SIO_TRT_WORKSPACE_MB` / `SIO_TRT_OPT_LEVEL` 调整构建设置。原始 ONNX 不改写。模型转换后需验证实际输出；转换成功本身不代表模型精度已验证。

默认先使用本机 `build/venv/model-tools/bin/python` 检查/准备派生图，依赖见项目根目录 `requirements-model-tools.txt`；默认优化级别为 0。`.build-key` 记录源 ONNX、构建参数和工具脚本指纹，参数变化不会误用旧引擎。显式设置 `SIO_DISP_ONNX=disp.onnx` 可改为直接转换原图。

`python3 scripts/fusion_lab.py run --sequence up --mode all` 默认先单独导出视差缓存，再运行独立 GICP、ESKF 和 ESIKF。缓存使用生产裁剪及光照处理；此模块实验不应用完整前端的邻域视差后处理。SGBM 基线保留为显式 `--config experiments/fusion/configs/baseline.yaml`。

原工作区的 models/ 被 Git 忽略；本发布目录 results/models/ 随仓库跟踪。
