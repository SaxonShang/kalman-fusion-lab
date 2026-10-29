# 以 IMU 运动先验稳定的 3DGS 位姿优化

> 五篇 SCI 之 **Paper 4** · [索引](SCI_五篇论文索引.md)  
> 工程：`src/backend.cpp`，`src/pose_3dgs_optimizer.cpp`，`src/torch_pose_optimize.cpp`，`src/imu_integator.cpp`  
> 工程导读：[SIO_SLAM_后端说明.md](SIO_SLAM_后端说明.md)

## 建议题目

IMU-Prior Stabilized Pose Optimization for 3D Gaussian Splatting in a Sliding-Window Backend

（备选）Stabilizing 3DGS Photometric Pose Refinement with IMU Motion Priors and Preintegration Constraints

---

## 1. Introduction

### 1.1 研究背景

视觉-惯性 SLAM 的前端通常已能给出可用的关键帧位姿与协方差，但在需要更高局部精度、或希望用稠密外观进一步纠偏时，后端仍有细化空间。近年来，3D Gaussian Splatting（3DGS）以一组各向异性高斯显式表示场景，支持实时可导光栅化，从而把“渲染图像与真实图像的差异”直接变成对相机位姿的梯度信号。这比稀疏特征重投影更稠密，也比隐式神经场更易嵌入实时系统。

问题在于：光度损失对位姿往往是高度非线性、多峰的。关键帧图像若存在弱纹理墙面、运动模糊、自动曝光跳变或局部动态残留，仅靠最小化 $\|\hat{I}-I\|^2$ 很容易把位姿拧向错误局部极小。惯性测量单元（IMU）在几十到上百赫兹上连续观测角速度与比力，短时预积分对相对运动的约束通常比单次外观匹配更稳。因此，把 IMU 当作 **3DGS 位姿优化的运动先验与测量约束**，比“只用 3DGS 做建图、位姿仍纯视觉”更符合后端细化的实际需求。

### 1.2 现有方法问题

现有工作大致有三类不足。

第一，不少 3DGS-SLAM 把高斯地图主要用于重建与新视角合成，位姿仍来自独立的视觉里程计或粗跟踪；即使加入光度位姿优化，也很少系统讨论如何用惯性约束防止细化发散。

第二，纯外观细化默认“渲染误差下降 ⇒ 位姿变好”。在纹理丰富、光照稳定时成立；一旦外观退化，梯度方向可能与真实运动无关，表现为位姿抖动、偶发跳变或优化后轨迹反而变差。

第三，经典 VIO 对预积分因子、偏置与边缘化已有成熟写法，但与可导光栅化结合时，需要额外说清三件事：先验如何进入迭代初值、IMU 残差如何写成与光度同一套增量变量、相机系渲染梯度如何变回 IMU 状态空间。缺少这些说明时，工程上容易出现“IMU 只用来给个初值、优化过程又被光度拉跑”的半吊子融合。

### 1.3 本文思路（核心定位）

本文明确区分两种信息源的职责：

1. **IMU 预积分 → 运动先验**：在相邻关键帧之间积分得到相对运动，并传播为当前帧的参考位姿 $\mathbf{x}^{\mathrm{imu}}$。它告诉优化器“短时间内机体大概在哪、朝向哪”，既作初值，也作软锚点。  
2. **3DGS 可导渲染 → 外观观测**：在当前（或试探）相机位姿下渲染，用光度（及可选深度、SSIM）提供细化信号，修正先验中尚未被惯性完全约束的误差（例如缓慢漂移的平移尺度细节、需外观对齐的小姿态偏差）。  
3. **优化目标中的 IMU 测量项 → 稳定器**：不只在开始时用一次先验，而是在每一步损失里加入相对 $\mathbf{x}^{\mathrm{imu}}$ 的旋转/平移一致性，以及限制本步增量过大的正则，使光度更新不能无界拉扯位姿。

一句话概括：**3DGS 负责“看清楚”，IMU 负责“别跑偏”。** 本文不是用 3DGS 取代惯性导航，而是用惯性约束给 3DGS 光度位姿细化加保险。

### 1.4 主要贡献

1. 提出面向后端关键帧的 **IMU 先验驱动的 3DGS 位姿细化**框架：预积分提供运动先验，光栅化提供外观梯度，二者在同一增量变量下联合更新。  
2. 给出 IMU 一致性残差及其对机体系位姿的解析梯度，说明其如何与光度梯度一起进入联合损失，从而在迭代中持续发挥稳定作用。  
3. 给出相机系渲染梯度到 IMU 系的伴随变换，保证优化坐标与预积分状态定义一致，避免“梯度在相机系、状态在机体系”的错配。  
4. 设计以有/无 IMU 项为主的消融与退化场景实验，验证 IMU 测量项对位姿稳定、回退次数与轨迹指标的影响。

---

## 2. 方法说明

### 2.1 系统在整体 SLAM 中的位置

前端（动态滤波、Stereo-GICP、ESKF/ESIKF）输出关键帧的图像、深度、IMU 状态与关键帧间 IMU 数据。后端在独立线程的滑动窗口中，对窗口内关键帧做局部细化：不追求一次性全局平差，而是在短窗口内用稠密外观把位姿“磨细”，同时用 IMU 防止磨细过程失控。细化后的位姿可写回关键帧，并用于高斯地图的增密、剪枝与局部对齐。

### 2.2 模块角色分工

| 模块 | 输入 | 输出 | 在融合中的含义 |
|------|------|------|----------------|
| IMU 预积分 | 上一关键帧状态、帧间 IMU 序列 | $\Delta R,\Delta p,\Delta v$ 及先验 $\mathbf{x}^{\mathrm{imu}}$ | 短时运动的物理可信参考 |
| 3DGS 渲染器 | 相机位姿、世界系高斯地图、观测图 | 渲染图、光度/深度损失、相机系位姿梯度 | 稠密外观是否对齐的度量 |
| 梯度坐标变换 | 相机系 $\mathbf{g}_c$、外参 | 机体系 $\mathbf{g}_{\mathrm{photo}}$ | 让外观梯度作用在与 IMU 相同的状态上 |
| IMU 一致性 | 当前估计 vs $\mathbf{x}^{\mathrm{imu}}$ | $\mathbf{g}_{\mathrm{imu}}$ 或直接正则 | 每步把估计拉回惯性参考 |
| 联合优化器（LibTorch Adam） | $\mathbf{g}_{\mathrm{photo}},\mathbf{g}_{\mathrm{imu}}$、标量误差、权重 | 位姿增量、偏置增量 | 一次更新同时权衡“对齐图像”与“贴近 IMU” |
| 高斯地图管理 | 关键帧观测、位姿、梯度统计 | 更新后的高斯参数 | 为下一次渲染提供场景；主文贡献仍在位姿稳定 |

### 2.3 单关键帧细化流水线（说明）

下面按时间顺序说明一次 `WindowOptimize` 风格迭代在做什么，以及每一步若缺少会怎样。

**步骤 1：构造运动先验。**  
用上一关键帧已优化（或前端给出）的状态，乘上本段预积分增量，得到 $\mathbf{x}^{\mathrm{imu}}$。若没有这一步，3DGS 只能从前端粗位姿或任意初值开始，弱纹理时搜索范围过大。

**步骤 2：准备相机位姿并渲染。**  
将当前机体状态经外参变为左目世界位姿，调用可导 3DGS 前向，得到 $\hat{I}$（及 $\hat{Z}$）。渲染质量差时，后续光度梯度噪声大，更依赖 IMU 项托底。

**步骤 3：外观梯度回传并变到机体系。**  
反传得到相机李代数梯度后，必须用外参变到机体，才能与预积分状态相加。若跳过变换，等于在错误切空间里改 IMU 状态，表现为旋转/平移更新耦合错乱。

**步骤 4：计算 IMU 一致性梯度。**  
比较当前 $(R,\mathbf{p})$ 与 $(R^{\mathrm{imu}},\mathbf{p}^{\mathrm{imu}})$，得到 $\mathbf{g}_{\mathrm{imu}}$。这是“测量项进入优化”的关键：即使本帧图像完全不可靠，该项仍给出指向先验的下降方向。

**步骤 5：组装联合损失并 Adam 一步（或若干微步）。**  
将光度线性项、IMU 线性项、增量范数先验、可选偏置项合成标量 $\mathcal{L}$，对 $\delta\boldsymbol{\xi}$ 与偏置增量求导并更新。权重决定更信图像还是更信惯性。

**步骤 6：安全回退。**  
若出现 NaN、损失爆炸或位姿跳变超阈，恢复到 $\mathbf{x}^{\mathrm{imu}}$ 并施加凸组合式一致性正则。工程上这保证“宁可信 IMU，也不提交坏位姿”。

### 2.4 为何 IMU 能稳定 3DGS（机理说明）

可以从四个层次理解稳定性来源。

1. **初值层。** 预积分在短时内积分误差有界，给出的 $R^{\mathrm{imu}},\mathbf{p}^{\mathrm{imu}}$ 通常已落入正确光度盆地附近，减少多峰误吸。  
2. **软锚层。** 一致性损失在状态空间放置以先验为中心的碗状势能；光度梯度要挪动位姿，必须“爬过”这只碗。纹理越差，越应加大碗的曲率（增大 $w_r,w_p$）。  
3. **步长层。** 对 $\|\delta\boldsymbol{\theta}\|^2,\|\delta\boldsymbol{\rho}\|^2$ 加权，等价于限制单次更新的信任域，避免一次 Adam 步长过大直接跳出盆地。  
4. **偏置层（可选增强）。** 若预积分对偏置的雅可比可用，则位姿增量与 $\delta\mathbf{b}$ 联立，避免“用错误偏置生成的先验”被当作绝对真理；偏置也可被小步修正，使先验本身随优化变准。

相反，若只有 3DGS 光度：系统在外观良好时可能更“贴图”，但在退化帧上缺乏物理可行方向，轨迹易出现毛刺。若只有 IMU 先验、不做外观细化：短时稳，但无法利用地图纹理修正缓慢漂移。二者联合才是本文设定的工作点。

### 2.5 与前端滤波的关系（避免职责混淆）

前端 ESKF/ESIKF 已在帧率上融合 IMU 与点云几何；后端 3DGS 优化是在关键帧尺度上的**二次细化**。后端 IMU 项使用的是关键帧间预积分，与前端同源，但优化变量侧重位姿（及偏置），不以重新估计整段速度协方差为主。前端保证“跟得住”，后端保证“关键帧更准、地图更好用”。系统集成篇讨论二者异步调度；本篇只展开后端内部的先验–外观–测量项机制。

---

## 3. 理论推导

约定：每个显示公式后给出符号含义。优化状态定义在 **IMU 机体相对世界**；3DGS 前向在 **左目相机系** 完成。下列推导与当前工程路径一致，便于对照代码。

### 3.1 状态、外参与相机位姿

后端节点保存的机体系名义状态取前端兼容子集：

$$
\mathbf{x}
=
\bigl(R_{wb},\,\mathbf{p}_{wb},\,\mathbf{v}_{wb},\,\mathbf{b}_g,\,\mathbf{b}_a\bigr).
$$

其中：

- $R_{wb}\in\mathrm{SO}(3)$：机体相对世界的旋转；  
- $\mathbf{p}_{wb}\in\mathbb{R}^3$：机体在世界系的位置（m）；  
- $\mathbf{v}_{wb}$：世界系速度（m/s），主要用于预积分预测，本后端一步中通常不作为 Adam 主变量；  
- $\mathbf{b}_g,\mathbf{b}_a$：陀螺 / 加计偏置，可与位姿增量联合微调。

相机与 IMU 之间外参在标定后固定。记 $T_{b\leftarrow c}=(R_{bc},\mathbf{t}_{bc})$ 为**机体←相机**（点从相机系变到机体系），代码中为 `R_bc_`、`t_bc_`。则左目在世界系下的位姿为

$$
R_{wc}=R_{wb}R_{bc},
\qquad
\mathbf{p}_{wc}=\mathbf{p}_{wb}+R_{wb}\mathbf{t}_{bc}.
$$

其中 $R_{wc},\mathbf{p}_{wc}$ 直接送入 3DGS。说明：高斯球的均值在世界系存储，渲染时再变到当前相机；因此改 $R_{wb},\mathbf{p}_{wb}$ 会改变所有高斯的投影，从而改变光度，这正是位姿可观性的来源。

### 3.2 IMU 预积分运动先验

设上一关键帧状态为 $\mathbf{x}_{k-1}$，两关键帧之间的 IMU 数据经预积分压缩为相对增量 $(\Delta R,\Delta\mathbf{p},\Delta\mathbf{v})$（表达在上一关键帧机体系的常用离散形式）。则当前帧的**运动先验**为

$$
\begin{aligned}
R_{wb}^{\mathrm{imu}}
&=
R_{wb,k-1}\,\Delta R,
\\
\mathbf{p}_{wb}^{\mathrm{imu}}
&=
\mathbf{p}_{wb,k-1}+R_{wb,k-1}\Delta\mathbf{p},
\\
\mathbf{v}_{wb}^{\mathrm{imu}}
&=
\mathbf{v}_{wb,k-1}+R_{wb,k-1}\Delta\mathbf{v}.
\end{aligned}
$$

其中：

- $\Delta R\in\mathrm{SO}(3)$：预积分旋转增量；  
- $\Delta\mathbf{p},\Delta\mathbf{v}$：预积分位置 / 速度增量；  
- $\mathbf{x}^{\mathrm{imu}}$：代码中的 `imu_init`，即 3DGS 优化应对齐的惯性参考。

**说明。** 预积分把高速率 IMU 变成关键帧之间的一条相对约束，避免优化时反复重积分原始采样。先验在本文中承担三重角色：（i）迭代初值，缩小搜索域；（ii）一致性损失的参考点，定义“离 IMU 有多远”；（iii）数值失败时的回退点，保证输出物理上不太离谱。若预积分本身因偏置或时间戳错误而偏差很大，后端稳定性能下降——这要求前端时间对准与偏置估计基本可靠，而不是后端单独可解决的问题。

### 3.3 3DGS 外观观测（光度 / 深度）

在相机位姿 $(R_{wc},\mathbf{p}_{wc})$ 下，可导光栅化得到渲染强度 $\hat{I}$（可选深度 $\hat{Z}$）。观测左目图像为 $I$，视差得到的深度为 $Z$，像素置信度为 $c$。定义

$$
\mathcal{L}_{\mathrm{photo}}
=
\frac{1}{|\Omega|}
\sum_{\mathbf{u}\in\Omega}
\bigl\|\hat{I}(\mathbf{u})-I(\mathbf{u})\bigr\|^2,
$$

$$
\mathcal{L}_{\mathrm{depth}}
=
\sum_{\mathbf{u}}
c(\mathbf{u})
\bigl(\hat{Z}(\mathbf{u})-Z(\mathbf{u})\bigr)^2.
$$

其中 $\Omega$ 为参与统计的有效像素集合（可剔除无高斯覆盖或深度无效处）。

**说明。** $\mathcal{L}_{\mathrm{photo}}$ 衡量“当前位姿下地图看起来像不像这张图”。它提供细化能力，但也把图像噪声、光照变化、地图未更新区域的误差写进梯度。深度项在双目可靠时可增加沿视线的约束，对 $t_z$ 类误差更直接；置信度 $c$ 用于降低远距、遮挡边缘的权重。工程上还可计算 SSIM 等结构误差，作为对光度的补充标量，并与同一 $\mathbf{g}_{\mathrm{photo}}$ 耦合（见 §3.6）。

渲染器对相机位姿的一阶敏感性写为

$$
\mathbf{g}_c
=
\frac{\partial\mathcal{L}_{\mathrm{photo}}}{\partial\boldsymbol{\xi}_c}
=
\begin{bmatrix}\mathbf{g}_{\theta,c}\\ \mathbf{g}_{p,c}\end{bmatrix}
\in\mathbb{R}^6.
$$

其中 $\boldsymbol{\xi}_c$ 为相机位姿李代数扰动；$\mathbf{g}_{\theta,c},\mathbf{g}_{p,c}$ 分别为旋转与平移部分。该梯度在相机切空间，尚不能直接加到机体状态上。

### 3.4 渲染梯度：相机系 → 机体系

因优化变量是 $R_{wb},\mathbf{p}_{wb}$，需把 $\mathbf{g}_c$ 变到机体切空间：

$$
\mathbf{g}_{p,b}=R_{bc}\,\mathbf{g}_{p,c},
\qquad
\mathbf{g}_{\theta,b}=R_{bc}\,\mathbf{g}_{\theta,c}+\mathbf{t}_{bc}\times\mathbf{g}_{p,b}.
$$

其中：

- $\mathbf{g}_{p,b}$：机体系位置梯度；  
- $\mathbf{g}_{\theta,b}$：机体系旋转梯度；  
- $\mathbf{t}_{bc}\times\mathbf{g}_{p,b}$：杠杆臂引起的姿态–位置耦合——相机不在 IMU 原点时，平移梯度会诱导额外的旋转梯度。

记 $\mathbf{g}_{\mathrm{photo}}=[\mathbf{g}_{\theta,b}^\top,\mathbf{g}_{p,b}^\top]^\top$。

**说明。** 这一变换与 SE(3) 外参的伴随/链式法则一致，是“IMU 先验与 3DGS 外观在同一状态空间说话”的前提。漏掉叉积项时，常见症状是旋转更新偏弱或符号怪异，细化后姿态抖、位置尚可。深度损失若只以标量 $E_{\mathrm{depth}}$ 进入联合目标、不显式反传深度位姿梯度，则主要起加权正则作用；若渲染器同时给出深度对位姿的梯度，可与光度一样变到机体后叠加进 $\mathbf{g}_{\mathrm{photo}}$。

### 3.5 IMU 测量项：一致性残差与梯度（稳定核心）

令当前优化中的机体位姿为 $(R,\mathbf{p})$，先验为 $(R^{\mathrm{imu}},\mathbf{p}^{\mathrm{imu}})$。定义旋转与平移一致性损失

$$
\mathcal{L}_{\mathrm{imu,rot}}
=
w_r
\bigl\|
\mathrm{Log}\bigl(R^\top R^{\mathrm{imu}}\bigr)
\bigr\|^2,
\qquad
\mathcal{L}_{\mathrm{imu,pos}}
=
w_p
\bigl\|
\mathbf{p}-\mathbf{p}^{\mathrm{imu}}
\bigr\|^2.
$$

其中：

- $w_r,w_p\ge 0$：一致性权重（`imu_rot_consistency_weight` / `imu_pos_consistency_weight`）；  
- $\mathrm{Log}:\mathrm{SO}(3)\to\mathbb{R}^3$：把相对旋转映为轴角向量。

对位姿的解析梯度（实现 `ComputeImuConsistencyGradient`）为

$$
\mathbf{g}_{\mathrm{imu}}
=
\begin{bmatrix}
-2 w_r\,\phi\,\hat{\mathbf{a}} \\[4pt]
2 w_p\,(\mathbf{p}-\mathbf{p}^{\mathrm{imu}})
\end{bmatrix},
$$

其中 $\phi\hat{\mathbf{a}}=\mathrm{Log}(R^\top R^{\mathrm{imu}})$。

**说明。**  
- 平移项是标准二次势，梯度指向先验位置，模长与偏离成正比。  
- 旋转项对 $\|\mathrm{Log}(\cdot)\|^2$ 求导，梯度沿相对旋转的轴角方向，把当前姿态扭向 $R^{\mathrm{imu}}$。  
- 在联合损失里使用 $\delta\boldsymbol{\xi}^\top\mathbf{g}_{\mathrm{imu}}$（见下节）时，Adam 会同时收到“减小光度”与“靠近 IMU”的方向；两者同向则加速收敛，反向则按权重折中——这正是稳定机制的工作方式。

当优化失败需要硬恢复时，可采用凸组合式正则（`ApplyImuConsistencyRegularization`）：

$$
\mathbf{p}\leftarrow(1-w_p)\mathbf{p}+w_p\mathbf{p}^{\mathrm{imu}},
\qquad
R\leftarrow R\,\mathrm{Exp}\bigl(w_r\,\mathrm{Log}(R^\top R^{\mathrm{imu}})\bigr).
$$

这不是每步主路径，而是安全网：保证输出不会比先验更不可用。

### 3.6 联合目标与一步线性化更新

设本步优化变量为位姿增量 $\delta\boldsymbol{\xi}=(\delta\boldsymbol{\theta},\delta\boldsymbol{\rho})\in\mathbb{R}^6$ 与偏置增量 $\delta\mathbf{b}_g,\delta\mathbf{b}_a$。在当前渲染线性化点，用一阶展开构造标量代理损失（与 `BuildJointLoss` 对应）：

$$
\begin{aligned}
\mathcal{L}
&=
\lambda_{\mathrm{ph}}
\bigl(
E_{\mathrm{photo}}
+
\delta\boldsymbol{\xi}^\top\mathbf{g}_{\mathrm{photo}}
\bigr)
+
\lambda_{\mathrm{ssim}}
E_{\mathrm{ssim}}
\,
\delta\boldsymbol{\xi}^\top\mathbf{g}_{\mathrm{photo}}
+
\lambda_d E_{\mathrm{depth}}
\\
&\quad
+
\lambda_{\mathrm{imu}}
\,
\delta\boldsymbol{\xi}^\top\mathbf{g}_{\mathrm{imu}}
+
\lambda_{\mathrm{pre}}
\bigl(
w_r\|\delta\boldsymbol{\theta}\|^2
+
w_p\|\delta\boldsymbol{\rho}\|^2
\bigr)
+
\lambda_b\mathcal{L}_{\mathrm{bias}}.
\end{aligned}
$$

其中：

- $E_{\mathrm{photo}},E_{\mathrm{ssim}},E_{\mathrm{depth}}$：本步前向得到的光度 / SSIM / 深度标量；  
- $\lambda_{\mathrm{ph}},\lambda_{\mathrm{ssim}},\lambda_d,\lambda_{\mathrm{imu}},\lambda_{\mathrm{pre}},\lambda_b$：各项权重；  
- $\delta\boldsymbol{\xi}^\top\mathbf{g}_{\mathrm{photo}}$：把“沿光度下降方向走”写成对增量可微的项；  
- $\delta\boldsymbol{\xi}^\top\mathbf{g}_{\mathrm{imu}}$：**IMU 测量梯度进入同一次反向**，是稳定项的实现形式；  
- $\lambda_{\mathrm{pre}}(w_r\|\delta\boldsymbol{\theta}\|^2+\cdots)$：信任域式步长先验，抑制单步过大。

**各项直观解释。**  
- 仅 $\lambda_{\mathrm{ph}}$ 很大：尽量贴图，易过拟合噪声。  
- 仅 $\lambda_{\mathrm{imu}}$ 很大：贴死先验，3DGS 几乎不纠正。  
- $\lambda_{\mathrm{pre}}$ 适中：允许小步外观修正，禁止飞步。  
- $\lambda_{\mathrm{ssim}}$：在强度绝对值不可靠但结构仍在时，增强对边缘/纹理结构的敏感（通过同一 $\mathbf{g}_{\mathrm{photo}}$ 调制）。

偏置耦合（预积分雅可比可用时）：

$$
\mathbf{r}_{bg}=\delta\boldsymbol{\theta}+J_{R,b_g}\delta\mathbf{b}_g,
\qquad
\mathbf{r}_{ba}=\delta\boldsymbol{\rho}+(J_{p,b_a}+\alpha J_{v,b_a})\delta\mathbf{b}_a,
$$

$$
\mathcal{L}_{\mathrm{bias}}
=
\|\mathbf{r}_{bg}\|^2
+
\alpha\|\mathbf{r}_{ba}\|^2
\quad(\alpha\approx 0.3).
$$

其中 $J_{R,b_g},J_{p,b_a},J_{v,b_a}$ 为预积分相对偏置的一阶雅可比。含义是：若偏置改变应连带修正等效位姿增量，则二者不应独立乱走。雅可比未接入时，该项退化为对增量本身的额外惩罚，仍有一定步长约束作用，但失去“修正先验生成过程”的能力。

对 $\mathcal{L}$ 用 Adam 求得增量后注入名义状态：

$$
R\leftarrow R\,\mathrm{Exp}(\delta\boldsymbol{\theta}),
\qquad
\mathbf{p}\leftarrow\mathbf{p}+\delta\boldsymbol{\rho},
\qquad
\mathbf{b}\leftarrow\mathbf{b}+\delta\mathbf{b}.
$$

可在同一关键帧上重复“渲染 → 梯度 → 联合一步”若干次，形成小规模迭代；窗口内多关键帧则可依次或联合处理，但单帧机制已包含本文主张的 IMU 稳定思想。

### 3.7 信息流与设计取舍

```text
上一关键帧状态 + 帧间 IMU
        │
        ▼
 预积分 → x_imu（运动先验）──────────────┐
        │                                │
        ▼                                ▼
 外参 → 相机位姿 → 3DGS 渲染/反传    一致性残差 L_imu
        │                                │
        ▼                                ▼
 g_c →（外参伴随）→ g_photo    ◄── 联合 L ──►  g_imu
        │                                │
        └────────── Adam：δξ, δb ────────┘
                       │
                       ▼
              稳定位姿 → 写回关键帧 / 更新地图
```

**设计取舍说明。**  
- 采用“线性化代理损失 + Adam”，而不是每步完整 Gauss–Newton 法方程，是为了与可导渲染生态一致、实现简单、易调学习率；代价是统计最优性不如经典 IEKF/因子图一次成型。  
- IMU 项用“相对先验的二次一致”而非完整 15 维预积分因子，是刻意把后端聚焦在**位姿稳定**；速度、重力等仍以前端为准。  
- 高斯地图可同步更新，但若地图错误与位姿错误耦合过强，应降低地图学习率或隔步更新，避免“坏位姿写坏地图、坏地图再反咬位姿”的正反馈。

### 3.8 符号总表

| 符号 | 含义 |
|------|------|
| $\mathbf{x}^{\mathrm{imu}},R^{\mathrm{imu}},\mathbf{p}^{\mathrm{imu}}$ | 预积分传播的运动先验 |
| $\Delta R,\Delta p,\Delta v$ | 关键帧间 IMU 预积分增量 |
| $R_{bc},t_{bc}$ | 相机–IMU 外参（机体←相机） |
| $\hat{I},I,\mathcal{L}_{\mathrm{photo}}$ | 渲染图、观测图、光度损失 |
| $\mathbf{g}_c,\mathbf{g}_{\mathrm{photo}}$ | 相机系 / 机体系光度梯度 |
| $\mathcal{L}_{\mathrm{imu,rot/pos}},\mathbf{g}_{\mathrm{imu}}$ | IMU 一致性损失与梯度 |
| $\delta\boldsymbol{\xi}=(\delta\boldsymbol{\theta},\delta\boldsymbol{\rho})$ | 本步旋转/平移增量 |
| $\lambda_{\mathrm{imu}},\lambda_{\mathrm{pre}},w_r,w_p$ | IMU 项与步长先验相关权重 |

### 3.9 实现索引

| 内容 | 位置 |
|------|------|
| 预积分预测先验 | `PredictStateFromImuPreintegration`（`backend.cpp`） |
| IMU 一致性梯度 / 硬正则 | `ComputeImuConsistencyGradient` / `ApplyImuConsistencyRegularization` |
| 3DGS 渲染与光度梯度 | `Pose3DGSOptimizer::RenderAndPoseGradients` |
| 相机→机体梯度 | `WindowOptimize` 中基于 $R_{bc},t_{bc}$ 的变换 |
| 联合损失与 Adam | `TorchPoseOptimizer::BuildJointLoss` / `BackwardAndStep` |
| 预积分计算 | `ImuIntegrator::ImuPreintegrationPose` |
| 坐标系与梯度流文字说明 | [SIO_SLAM_后端说明.md](SIO_SLAM_后端说明.md) §2.4 |

---

## 4. 实验

实验应能回答一个核心问题：**加入 IMU 运动先验与优化中的 IMU 测量项之后，3DGS 位姿细化是否更稳、更准？** 轨迹精度是结果，回退次数、抖动与消融则直接对应机制。

1. **平台与实现细节**：滑窗长度、每关键帧迭代次数、$\lambda_{\mathrm{imu}},w_r,w_p,\lambda_{\mathrm{pre}}$、Adam 学习率、是否启用深度/SSIM/偏置项；报告 CPU/GPU 与是否 LibTorch。  
2. **数据与指标**：EuRoC、TUM VI、KITTI；ATE/RPE；另计优化 NaN/回退次数、单关键帧后端耗时、位姿高频抖动（短基线 RPE）。  
3. **主消融（对应本文主张）**  
   - A：仅 3DGS 光度（无 IMU 一致性、可用随机或前端初值）  
   - B：IMU 初值 + 光度（有先验初值，但优化中无 $\mathbf{g}_{\mathrm{imu}}$）  
   - C：IMU 初值 + 光度 + 一致性项（本文完整）  
   - D：C + 深度 / SSIM / 偏置  
   预期：B 相对 A 改善初值相关失败；C 相对 B 降低退化序列上的发散与抖动。  
4. **外部对比**：visual-only 3DGS 后端；ORB-SLAM3 / VINS-Fusion 等 VIO（说明后端细化相对前端的增益，而非宣称取代前端）。  
5. **退化场景**：弱纹理走廊、快速转动、曝光变化；对比有无 IMU 项的失败率。  
6. **权重敏感性**：增大 $w_r,w_p$ 直至轨迹贴齐预积分（外观收益消失）；减小直至接近纯视觉毛刺——用于给出推荐工作区间说明。  
7. **副指标**：PSNR/SSIM/地图规模——用于说明稳定位姿对建图的连带好处，避免只报渲染不报定位。  
8. **资源**：显存、单帧后端时延，讨论实时异步可行性。

---

## 5. 结论

1. 将 IMU 预积分作为 3DGS 位姿优化的运动先验，并在目标函数中持续加入 IMU 一致性测量项，能够在保留外观细化能力的同时显著提高优化稳定性。  
2. 稳定性来自初值、软锚、步长先验（及可选偏置耦合）多层机制；仅“用 IMU 给个初值”而不加测量项，在退化外观下仍不足。  
3. 相机系渲染梯度必须正确变到机体系，才能与预积分先验在同一状态空间联合，这是实现正确性的前提而非实现细节。  
4. 权重选择体现信任分配：过大则放弃外观修正，过小则回到纯 3DGS 的不稳；需按场景纹理与 IMU 质量调节。  
5. **局限**：预积分与时间戳错误会直接污染先验；后端一致性模型刻意简化，未展开完整多状态预积分因子图；地图与位姿强耦合时需额外调度。  
6. **后续**：贯通预积分偏置雅可比、多关键帧联合 IMU–光度因子、以及与前端协方差的一致性传递（见系统集成篇）。
