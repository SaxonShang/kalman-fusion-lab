# 融合协方差诊断（2026-10-29）

完整的研究目标、数据、算法计算过程、对照解释与待审查问题，见 [实验审阅说明](EXPERIMENT_REVIEW_BRIEF.md)。阅读与评议不需要配置环境或复跑。

本轮保留模型视差、固定帧对、GICP main `744ac2d` 和观测参数，修正两处可独立复现的实现问题。数值正确性与配准精度分别检查，R/P 敏感性实验不改变默认配置。

## 已修正的实现问题

### 初始 P 的轴索引

状态使用 `[θ(0:3), p(3:6), v(6:9), bg(9:12), ba(12:15), g(15:18), expo(18), reserved(19:21)]`。

旧构造函数仅将 `P(6,6)` 设为较小速度方差，并从 `(10,10)` 开始设置偏置部分，导致 `vy/vz`、`bgx` 沿用大方差。例如 `bgx` 初始方差是另外两轴的 1000 倍。现在按物理三维块设置 v、bg、ba、g，并显式设置 expo；保留已有方差数值，不据真值调参。

在 `up` 结束段，旧 P 的位置标准差约 `[169.60,1023.03,2.18] m`、旋转标准差约 `[105.09,6.63,7.16]°`；修正后分别约 `[168.69,168.69,0.78] m` 与 `[6.63,6.62,6.63]°`。轴异常已消除，但长期纯惯性绝对 P 仍很大，不能认为已完成初始不确定性标定。

### ESKF 统一阻尼的量纲混合

旧方法对 `S=P_pose+R` 使用 `λ=1e-6×max(1,trace(S)/6)`，然后求解 `S+λI`。S 同时包含 rad² 和 m²；位置方差较大时，该 λ 会作为额外的角度噪声压低旋转增益。

现在先用 `D=diag(sqrt(diag(S)))` 转换到无量纲坐标，分解 `S_n=D⁻¹ S D⁻¹`，通过线性求解计算 K，不显式求逆。正常正定情况不加阻尼；接近浮点分辨率的退化情况仅在无量纲坐标加入相对 jitter，再映回各维。Joseph 协方差仍使用同一次 K 和真实观测 R。

同一 `up/static_end/small_gicp` 冻结 P 和 R 下，旧 K 的旋转对角线约 `[0.1791,0.0694,0.0081]`；修正后约 `[0.99999,0.99999,0.99984]`。独立 NumPy 缩放求解证实新结果；这不是依赖 GT 的权重调整。

新 ESKF 文件包含 `innovation_solver=diagonal_whitening`、`innovation_scale`、`innovation_covariance_normalized`。其中 `damping` 表示无量纲相对 jitter；旧文件里的同名字段是原始统一阻尼，比较时应看求解器标记。NIS 使用实际求解的 S。

## 三组对照与验证

使用 [fusion_audit.py](../scripts/fusion_audit.py) 自动执行：

1. **solver_only**：保持旧观测、先验均值和完整 P，仅换成修正后的 ESKF 求解。
2. **repropagated**：复用同一模型视差，重新传播修正初始 P 后的先验，再执行独立 GICP、ESKF、ESIKF。
3. **sensitivity**：保持新观测，事先固定 `P×0.0001`，比较 `0.1R / R / 10R / 1000R / 100000R` 及固定 R，作为诊断，不自动选择“最优”默认值。

六条序列新增 960 条记录：240 solver_only、360 repropagated、360 sensitivity，全部通过输入/后验数值检查。原始观测与仅求解器对照的先验均值、完整 P 逐元素一致。重新传播的先验均值、GICP 位姿和 R 保持一致，60 个 GICP 位姿的最大变化为 0；改变的是 P 和滤波计算。各组新 K 与独立 NumPy 解的最大相对误差低于 `2.1e-11`。

修复前两项回归测试失败，修复后全量 **93 项测试通过**，还覆盖了接近浮点分辨率的相关创新矩阵。完整 GPU 前端、demo、集成测试与视差导出目标均重新编译通过；未运行完整 SLAM 后端。

日志：`build/covariance-regressions-before-fix.log`、`build/covariance-final-tests.log`、`build/covariance-runtime-build.log`、`build/fusion-covariance-audit.log`。修复前的源文件与实验程序保留在 `build/fusion-before-covariance-fixes/`。

## 精度结果与剩余限制

以下是 30 组固定帧对的 ESKF `R_x1` 条件 RMSE，不是全序列 ATE：

| 方法 | 原 ESKF | 仅修正求解、同一 P | 再修正初始 P |
|---|---:|---:|---:|
| 欧氏 GICP → ESKF，平移 mm | 18.812 | 18.887 | 18.832 |
| 欧氏 GICP → ESKF，旋转 ° | 0.36887 | 0.47008 | 0.46620 |
| Stereo → ESKF，平移 mm | 22.016 | 22.021 | 22.021 |
| Stereo → ESKF，旋转 ° | 0.09395 | 0.14733 | 0.14732 |

IMU 先验为 25.754 mm / 0.05961°。修正数值求解后旋转误差更接近 GICP 观测误差，部分指标变差；旧统一阻尼曾无意中削弱视觉观测。当前 P 很大、Hessian R 很小，增益接近相信视觉观测，不能把“数值检查通过”当作精度达标。

接下来的实验应重点区分两个问题：初始 P/IMU 噪声与视觉 R 尚未统计标定；当前 P 是长时间绝对惯性传播结果，而报告使用同一锚点条件相对真值，锚点相关性尚未进入观测模型。小 NIS 不能单独证明一致性，也不宜只靠统一放大 R 来掩盖这些区别。

ESIKF 仍直接读取相同点云和 IMU 先验，本轮仅受初始 P 索引修复影响，点级观测与迭代公式未改；不继承 GICP 或 ESKF 后验。

## 使用入口

[诊断报告索引](../results/diagnostics/index.html) 中可查看 P/R 逐轴标准差、K 对照、NIS、灵敏度结果和每条序列的原始报告。当前完整报告为 [20261029_105246](../results/diagnostics/20261029_105246/report.html)。IDE 的本机选择已更新到重新传播 P 后的 `up` 观测。

以下命令用于原工作区的 `artifacts/`。发布快照跨机器运行时请使用 `python3 scripts/reproduce_published.py audit`；历史 manifest 保留原绝对路径，不直接改写。

原工作区重新生成报告（不重跑实验）：

```bash
python3 scripts/fusion_audit.py --manifest artifacts/fusion_diagnostics/20261029_105246/manifest.json
```

从已有模型基线重新跑对照，所有结果进入新目录，原始记录不覆盖：

```bash
python3 scripts/fusion_audit.py --reference artifacts/fusion/20261029_103*_all --run-controls
```

2026-10-29 起按用户要求准备独立的新 GitHub 实验仓库；原工作区的远程配置保持不变。新仓库的历史记录保存在 `results/`，跨机器重建报告及回放请使用 [复现指南](REPRODUCING.md)；原工作区的上述命令继续适用。创建新仓库和首次上传由用户在终端执行，见 [发布步骤](PUBLISH_NEW_REPOSITORY.md)。
