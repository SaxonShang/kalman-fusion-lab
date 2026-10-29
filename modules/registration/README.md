# GICP 版本与本地接入

本目录采用 [Ben-cpy/robotic-front 的 main](https://github.com/Ben-cpy/robotic-front/tree/744ac2d993ae629b0cc44a3b1d995e04ab4fad48/modules/registration)，固定提交 `744ac2d993ae629b0cc44a3b1d995e04ab4fad48`（2026-10-28 接入）。[UPSTREAM.json](UPSTREAM.json) 保存原始文件 SHA256 和本地适配说明。后续实验的配置快照记录此版本；旧实验保留原样。

- `src/color_icp.cpp`：上游配准入口、small_gicp 缓存与 Stereo 适配。
- `src/stereo_gicp.cpp`：上游投影求解器，包括重关联、LM、双目测量协方差与可选点面候选。
- `tests/test_stereo_gicp.cpp`：上游 13 项控制与已知答案测试。
- 项目旧头文件转发到本模块。旧投影配准循环已移除；`src/stereo_gicp.cpp` 仅保留原 ESIKF 使用的观测工具。

默认类型为欧氏 **GICP**，与上游 main 一致。在 `color_icp.registration_type` 中可显式选择 `gicp`、`icp`、`plane_icp`、`stereo_gicp`；切换会使预处理缓存失效。融合实验默认对照 `small_gicp` 与 `stereo` 两种方法，见 [实验配置](../../experiments/fusion/configs/baseline.yaml)。不在方法之间自动回退。

本地适配包括 yaml-cpp 配置读取（兼容平铺及 `color_icp` 子节）、PCL/FLANN 编译兼容，以及失败结果初始化和诊断字段。求解器的目标函数、关联、更新、Hessian 与停止行为保持上游版本。Stereo 的 `stereo_point_plane`、`stereo_disparity_normals`、`stereo_source_voxel_representative` 默认均关闭；点面组合需要显式配置。早停与协方差地板通过 `ConfigureStereoGicpSolver` 设置，仅写同名 YAML 字段不会自动生效。

## 与卡尔曼滤波的协方差接口

本地适配边界使用各求解器实际的扰动约定，与上游前端统一使用分块旋转的接法有所区别。求解器返回的原始协方差保持不变，完整交叉项保留在实验文件中。

设相对位姿 `T_target_source = [R,t]`，锚点世界旋转为 `A`，ESKF 使用 `[δθ_current_body, δp_world]`：

- small_gicp 的 Hessian 对应右扰动 `T Exp(δ)`；转换 Jacobian 为 `diag(I, A R)`。
- Stereo 对应左扰动 `Exp(δ) T`；转换 Jacobian 为 `[[Rᵀ,0],[-A[t]×,A]]`。
- 相机到机体的共轭变换先用 `Ad(T_imu_cam)` 处理。转换函数在 [utils.h](../../include/utils.h)，单测用非单位位姿和有限差分检查。

`covariance_from_hessian` 区分真实求解协方差和失败时的默认值；滤波不使用默认协方差。Stereo 的原始 `converged`、`status` 和逐轮诊断分别导出。上游默认 Stereo 路径在迭代耗尽或停止于局部极小时也可能置 `converged=true`，评估时应同时查看 `solver_status`，不能据此宣称精度提高。

逐轮归一化 Hessian 谱诊断仅在点面候选启用时计算；默认 Stereo 路径中的 `-1` 表示未计算。欧氏 GICP 未提供该逐轮记录，也未提供独立点距 RMSE，后者导出为 NaN。

关键帧 ESIKF 仍直接读取点云与 IMU 先验，不依赖 GICP 的位姿或协方差。使用方式见 [融合实验工作区](../../experiments/fusion/README.md)。
