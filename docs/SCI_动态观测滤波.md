# 基于位姿不确定度的几何–神经光流动态观测滤波

> 五篇 SCI 之 **Paper 1** · [索引](SCI_五篇论文索引.md) · 工程：`src/dynamic_cloud_processor.cpp`，`include/internal/geo_flow_error.h`

## 建议题目

Stereo-Inertial Dynamic Observation Filtering with Optical-Flow Uncertainty for Motion-Aware Static Reconstruction

## 1. Introduction

### 1.1 研究背景

在双目-惯性 SLAM 中，动态目标首先污染的往往不是后端地图，而是**前端观测本身**。行人、车辆、摆动物体等独立运动会同时破坏三处假设：（i）双目视差对应静物；（ii）帧间光流可由相机刚体解释；（iii）点云配准的 target 点云为静态场景。一旦动态点进入 GICP / ESIKF，错误对应会把位姿拧偏，再经滤波协方差收缩被“自信地”传播下去。鲁棒核与后端边缘化只能减轻已发生的污染；若能在观测层提前识别动态区域并只保留静态点，则更可能从源头切断误差链。

双目系统额外提供稠密深度，IMU 提供短时相对运动，二者结合起来可以回答：“若场景全是静态的，像素应该怎么走？”把这个预测与神经网络估计的实际光流对比，就得到不依赖语义类别的运动一致性检验——这正是本文动态观测滤波的物理出发点。

### 1.2 现有方法问题

现有动态处理大致有三类不足。

第一，纯语义分割依赖训练集类别，对未知运动物体、局部形变或分割边界噪声敏感，且通常不利用 IMU 几何。

第二，大量工程实现使用固定像素阈值 $\tau$（或随深度线性放大的启发式 $\tau(Z)$）。旋转误差引起的流偏差与深度无关，平移误差则按 $1/Z$ 放大；固定阈值无法同时适配远近与旋转/平移主导情形，近处易误杀、远处易漏检。

第三，不少工作止于报告掩码 IoU，很少说明净化后的静态点集如何改变后续配准内点率、滤波创新一致性与轨迹 ATE。对 SLAM 而言，掩码精度只是中间量，**定位链路收益**才是最终判据。

### 1.3 本文思路

针对上述问题，本文提出基于双目几何与 IMU 位姿不确定度的动态观测滤波。几何流与门控椭圆分两路构造，不以 $\mathrm{Cov}(T_{c\leftarrow k})$ 充当 $P_{\mathrm{cam}}$。流程概览如下：

1. 用双目视差恢复参考关键帧稠密点云；  
2. 用 IMU 预测名义位姿复合出参考→当前的相对变换 $T_{c\leftarrow k}$（只用于投影）；同时用关键帧处置零、再与主滤波同步预测和 GICP 更新的并行协方差取出相机系 $6\times 6$ 位姿协方差 $P_{\mathrm{cam}}$（含交叉项，只用于门控）；  
3. 将参考点投影到当前帧，经 Z-buffer 生成几何光流 $\mathbf{f}_{\mathrm{geo}}$；  
4. 用 RAFT 等网络估计观测光流 $\mathbf{f}_{\mathrm{raft}}$，残差 $\mathbf{r}_f=\mathbf{f}_{\mathrm{raft}}-\mathbf{f}_{\mathrm{geo}}$；  
5. 经光流雅可比把 $P_{\mathrm{cam}}$ 映射为像素协方差 $\Sigma_f=\mathbf{J}_f P_{\mathrm{cam}}\mathbf{J}_f^\top$，加脊后做马氏门控 $s=\mathbf{r}_f^\top\tilde\Sigma_f^{-1}\mathbf{r}_f\ge 1$；  
6. 形态学与边界连通域清理后，输出参考帧静态索引 $\mathcal{S}$，供 Stereo-GICP / ESKF / ESIKF 使用。

方法刻意**不修改当前帧点云**，只净化作为配准 target 的参考关键帧点，以减小对前端数据结构的侵入。详细公式见第 3 节；从关键帧相对位姿协方差到门控的五步链见 §3.5。

### 1.4 主要贡献

1. 提出由双目几何与 IMU 运动先验共同驱动的 Z-buffer 几何光流预测，避免远点覆盖近点。  
2. 提出将关键帧相对位姿协方差（关键帧处置零后与主滤波同步预测、GICP 更新，含交叉项）映射到可倾斜的光流椭圆，并以马氏门控替代固定像素阈值的动态判别；明确旋转 / 平移不确定度映到像素半轴的量级（$\sim f\sigma_\omega$ 与 $\sim f\sigma_t/Z$）及随之松紧的门控策略。  
3. 给出双流同时过小才跳过检验、边界连通域清理等工程规则，降低静止漏检与出入视野伪影。  
4. 以静态点保留率、配准/滤波链路指标与轨迹误差，验证动态滤波对前端鲁棒性的提升。

## 2. 方法说明

### 2.1 系统总体流程与在前端中的位置

输入为同步双目图像与 IMU。本模块插在“双目重建之后、点云配准 / ESIKF 之前”：上一关键帧点云先经动态滤波得到静态子集，再作为 GICP 的 target 或 ESIKF 的锚点几何。当前帧仍完整参与重建，以免误删当前观测中尚可用的结构。整体数据流为：

```text
双目 → 视差/点云(参考KF) → 几何光流(f_geo)
IMU  → T_c←k（名义相对位姿，只投影） ↗        ↓
     → P_rel（关键帧置零后的并行协方差）→ P_cam → Σ_f → 马氏门控
图像对 → RAFT(f_raft) → 残差 r_f ↗                         → 静态索引 S → 配准/滤波
```

### 2.2 双目重建与参考几何建立

左右目估计视差后，按针孔模型反投影得到参考帧左目系点云，并在三角化时把参考像素写成 `StereoPoint` 的字段 `u`、`v`（`has_pixel` 为 1）。几何光流、RAFT 与参考图像都读这个字段，不再把 XYZ 投回参考内参恢复 $(u,v)$。点云同时承担两类角色：一是后续静态场景结构；二是几何光流的深度锚点——没有可靠深度就无法由刚体位姿预测像素位移。无有效视差（$d\le 0$，或被邻域一致性滤波清零）是左右目不重叠和立体失败区域：这些像素不建点，不进入静态云，也不标为动态。过近过远的点同样不建点，以免用噪声深度制造假几何流。

### 2.3 IMU 先验：名义相对位姿与关键帧相对协方差

IMU 误差状态传播同时给出两类量，用途分开：

1. **名义相对位姿** $T_{c\leftarrow k}$：由世界系相机位姿复合得到，只用于把参考点投到当前帧、生成 $\mathbf{f}_{\mathrm{geo}}$。  
2. **关键帧相对位姿协方差** $P_{\mathrm{cam}}\in\mathbb{R}^{6\times 6}$：另有一份与主滤波同维的并行协方差。参考关键帧成立时该矩阵置零，之后主滤波的每一次 IMU 预测（$\Phi P\Phi^{\top}+Q$）和 GICP / ESIKF 位姿更新都作用到这份矩阵上。门控取它的位姿 $6\times 6$（含 $P_{\theta\theta}$、$P_{pp}$ 与交叉 $P_{\theta p}$），经外参旋转与杠杆臂转到左目相机系。它是 $\mathrm{Cov}(T_{c\leftarrow k})$，不是当前绝对协方差；主滤波自己的绝对协方差保持不变。不从绝对协方差里减 $P_k$，也不使用两帧交叉 $P_{k,c}$。

平移对角设下界 $\sigma_{t,\min}^2$，覆盖未建模的短时平移不确定度，避免并行协方差被 GICP 压到很小时 $\Sigma_f$ 过小、几乎所有残差都被判动态（过杀）；交叉项保留，不把椭圆强制轴对齐。并行协方差从关键帧处的零开始，随之后的预测增长，并在 GICP 更新时被同一增益压回去，因此不是跳过视觉更新的开环预积分。从这一相对协方差走到马氏门控的五步见 §3.5。

### 2.4 几何光流与神经光流

几何光流回答“静态刚体世界下像素应如何移动”；神经光流回答“图像里实际测到怎样移动”。二者在静态区域应大致一致；独立运动物体上神经光流往往显著偏离几何预测。当前图像的 Z-buffer 在每个整数像素 $(u',v')$ 上只保留当前深度 $Z'$ 最近的点做检验，视点变化后被遮挡的点不进入动态或静态候选。

### 2.5 光流不确定度建模与动态判别

两路分开：$T_{ck}$ 只定几何流（椭圆中心），$P_{cam}$ 只定椭圆形状。由 $\Sigma_f = J_f P_{cam} J_f^{T}$ 得到像素协方差，加脊后得 $\tilde{\Sigma}_f$。马氏分数 $s = r_f^{T}\tilde{\Sigma}_f^{-1} r_f$：$s\ge 1$ 落在 $1\sigma$ 椭圆外，判为动态。从关键帧相对协方差到 $s$ 的五步（投影点协方差是 $\mathbf{J}_f$ 内部的中间映射）见 §3.5。

交叉项不改中心，只让椭圆相对像素轴倾斜。同号斜残差沿长轴，$s$ 更小（位姿误差更能解释）；反号沿短轴，$s$ 更大（更像独立运动）。对角 $P$ 在光心附近轴对齐，不区分这两种方向。位姿协方差映到像素后：旋转项 $\sigma_f\sim f\sigma_\omega$（与深度无关），平移项 $\sigma_f\sim f\sigma_t/Z$（近处放大）——量级与门控松紧见 §3.4、§3.6。仅当几何流与神经流**同时**很小时才跳过检验——避免相机近似静止时因 $f_{geo}\approx 0$ 而放过仍在动的物体。交叉项如何进入 $\Sigma_{uv}$ 见 §3.5。

### 2.6 后处理与前端接口

动态掩码经腐蚀膨胀去噪；触及顶/左/右边界的连通域做条带清理，抑制物体进出视野造成的假动态条带（底边常接地面，策略上保留）。参考点投影落入动态像素则移出 $\mathcal{S}$。超出当前图像、位于相机后方，或在当前视图中被更近表面遮挡的点标为不可见，单独输出为不可见点云，既不是静态也不是动态；每个当前像素只检验 $Z'$ 最近的可见点。没有视差图像素索引的点同样不可见（左右目不重叠），并入该不可见点云，既不进入 $\mathcal{S}$ 也不标为动态。输出 $\mathcal{S}$ 作为姊妹篇 Stereo-GICP / ESKF·ESIKF 的静态 target 契约。

## 3. 理论推导

每个显示公式后给出符号含义；光流雅可比与位姿误差→光流量级见 §3.4，从关键帧相对位姿协方差到门控的五步链见 §3.5，马氏门控与由协方差驱动的策略见 §3.6。姊妹篇：[投影 Stereo-GICP](SCI_投影StereoGICP.md)、[ESKF 与 ESIKF](SCI_ESKF与ESIKF.md)。解析灵敏度对照：[flow_pose_error_sensitivity.md](flow_pose_error_sensitivity.md)。

### 3.1 问题与假设

#### 问题表述

设参考关键帧左目图像为 $I_k$，当前帧左目图像为 $I_c$。参考帧经双目重建得到左目系点云 $\{\mathbf{P}_i\}_{i=1}^{N}$。动态滤波在**不修改当前帧点云**的前提下，从参考点云中剔除独立运动点，输出静态索引集合 $\mathcal{S}$，供后续 GICP / ESIKF 的 target 使用。

其中：

- $I_k,I_c$：参考 / 当前左目强度图；
- $\mathbf{P}_i\in\mathbb{R}^3$：参考左目系三维点；
- $\mathcal{S}\subseteq\{1,\ldots,N\}$：静态点索引。

#### 建模假设

1. **静态场景刚性**：静态点的像素位移仅由相机相对刚体运动引起。
2. **IMU 先验可用**：短时名义相对位姿 $T_{c\leftarrow k}$ 由 IMU 预测复合得到；光流椭圆取自当前预测协方差 $P^-$ 转到相机系后的 $6\times 6$ 位姿块 $P_{\mathrm{cam}}$（含轴间相关与 $P_{\theta p}$）。$P_{\mathrm{cam}}$ 是相机系位姿微扰的协方差代理，不是 $\mathrm{Cov}(T_{c\leftarrow k})$。
3. **神经光流为观测**：RAFT 等网络给出稠密光流 $\mathbf{f}_{\mathrm{raft}}$，含估计噪声，但不引入独立物体运动模型。
4. **位姿误差一阶小**：用光流对相机系位姿微扰的雅可比 $\mathbf{J}_f$ 做线性映射 $P_{\mathrm{cam}}\mapsto\Sigma_f=\mathbf{J}_f P_{\mathrm{cam}}\mathbf{J}_f^\top$。

---

### 3.2 相对位姿与几何光流

#### 相对位姿构造

世界系名义位姿经外参变到左目相机系后，

$$
T_{c\leftarrow k}
=
\bigl(T_{w\leftarrow c}^{\mathrm{cam}}\bigr)^{-1}
T_{w\leftarrow k}^{\mathrm{cam}}
=
\begin{bmatrix}
R & t \\
\mathbf{0}^\top & 1
\end{bmatrix}.
$$

其中：

- $T_{c\leftarrow k}\in\mathrm{SE}(3)$：参考左目系 → 当前左目系；
- $T_{w\leftarrow k}^{\mathrm{cam}},T_{w\leftarrow c}^{\mathrm{cam}}$：参考 / 当前左目相对世界的位姿；
- $R\in\mathrm{SO}(3)$：相对旋转；
- $t\in\mathbb{R}^3$：相对平移（m）；
- $\mathbf{0}$：三维零向量。

对参考点 $\mathbf{P}=(X,Y,Z)^\top$，

$$
\mathbf{P}' = R\mathbf{P}+t = (X',Y',Z')^\top.
$$

其中 $\mathbf{P}'$ 为同一点在当前左目系的坐标。

#### 针孔投影

$$
\pi(\mathbf{P})
=
\begin{pmatrix}
f_x\dfrac{X}{Z}+c_x \\[4pt]
f_y\dfrac{Y}{Z}+c_y
\end{pmatrix},
\qquad
\mathbf{u}'=\pi(\mathbf{P}').
$$

其中：

- $\pi$：针孔投影；当前像素取 $\pi(\mathbf{P}')$，参考像素不由 $\pi(\mathbf{P})$ 恢复；
- $f_x,f_y$：焦距（px）；
- $c_x,c_y$：主点（px）；
- $\mathbf{u}=(u,v)^\top$：参考像素，取自视差图索引，不由 $\mathbf{P}$ 再投影得到；
- $\mathbf{u}'=(u',v')^\top$：当前像素。

#### 几何光流定义

$$
\mathbf{f}_{\mathrm{geo}}(\mathbf{u})
=
\mathbf{u}'-\mathbf{u}
=
\begin{pmatrix}
f_x X'/Z'+c_x-u \\[2pt]
f_y Y'/Z'+c_y-v
\end{pmatrix}.
$$

其中 $\mathbf{f}_{\mathrm{geo}}$ 为 IMU 先验诱导的几何光流（px）；$(u,v)$ 是视差图索引，不是把 $\mathbf{P}$ 再投影到参考图像的结果。

#### Z-buffer

超出当前图像或位于相机后方的点标为不可见，不进入动态或静态候选，且不占用当前像素 Z-buffer。落在当前图像内的点按整数像素 $(u',v')$ 竞争，只保留当前深度 $Z'$ 最小者作为候选：马氏检验可标为动态，否则保留为静态。同一像素上 $Z'$ 更大的点视为被当前视图遮挡，同样标为不可见，不进入候选。没有有效视差索引的点标为不可见（左右目不重叠或立体失败），不默认保留为静态。

---

### 3.3 神经光流与残差

裁剪缩放后的 $(I_k,I_c)$ 输入 RAFT，得

$$
\mathbf{f}_{\mathrm{raft}}(\mathbf{u})\in\mathbb{R}^2.
$$

光流残差：

$$
\mathbf{r}_f
=
\mathbf{f}_{\mathrm{raft}}-\mathbf{f}_{\mathrm{geo}}
=
\begin{pmatrix} r_u \\ r_v \end{pmatrix}.
$$

其中：

- $\mathbf{f}_{\mathrm{raft}}$：观测光流（px）；
- $\mathbf{r}_f$：残差；$r_u,r_v$ 为其水平 / 垂直分量。

静态理想模型：$\mathbf{r}_f\approx\mathbf{0}$（在位姿与深度噪声范围内）。独立运动使 $\|\mathbf{r}_f\|$ 显著超出该范围。

---

### 3.4 位姿扰动到光流的一阶映射（逐步）

#### 归一化坐标

$$
x_n=\frac{u-c_x}{f_x},\qquad
y_n=\frac{v-c_y}{f_y},\qquad
\rho=\frac{1}{Z}.
$$

其中 $x_n,y_n$ 为归一化像平面坐标；$\rho$ 为逆深度（1/m）。

#### 小扰动下的光流敏感度

相机系位姿受到小扰动 $(\delta\boldsymbol{\omega},\delta\mathbf{t})$ 时，像点位移的一阶项由经典视觉里程计公式给出。该微扰作用在把参考点变到当前相机系的刚体变换上；其协方差取自 $T_{c\leftarrow k}$ 的相对位姿协方差映到相机系的 $P_{\mathrm{cam}}$（§3.5）。将 $\mathbf{J}_f\in\mathbb{R}^{2\times 6}$ 按列写出：

**旋转列（与深度无关）：**

$$
\begin{aligned}
\mathbf{J}_{\omega_x}
&=
\begin{pmatrix}
-f_x x_n y_n \\
-f_y(1+y_n^2)
\end{pmatrix},
&
\mathbf{J}_{\omega_y}
&=
\begin{pmatrix}
f_x(1+x_n^2) \\
f_y x_n y_n
\end{pmatrix},
\\
\mathbf{J}_{\omega_z}
&=
\begin{pmatrix}
-f_x y_n \\
f_y x_n
\end{pmatrix}.
\end{aligned}
$$

**平移列（与 $1/Z$ 成正比）：**

$$
\begin{aligned}
\mathbf{J}_{t_x}
&=
\begin{pmatrix}
f_x\rho \\ 0
\end{pmatrix},
&
\mathbf{J}_{t_y}
&=
\begin{pmatrix}
0 \\ f_y\rho
\end{pmatrix},
&
\mathbf{J}_{t_z}
&=
\begin{pmatrix}
-f_x x_n\rho \\
-f_y y_n\rho
\end{pmatrix}.
\end{aligned}
$$

其中：

- $\mathbf{J}_{\omega_\ast},\mathbf{J}_{t_\ast}$：光流对对应自由度的雅可比列（px/rad 或 px/m）；
- 推导要点：旋转改变视线方向，故敏感度不含 $\rho$；平移改变点在相机系中的位置，经投影链式法则出现 $\rho$。

组装为完整 $2\times 6$ 光流雅可比：

$$
\mathbf{J}_f
=
\bigl[
\mathbf{J}_{\omega_x}\ \mathbf{J}_{\omega_y}\ \mathbf{J}_{\omega_z}\
\mathbf{J}_{t_x}\ \mathbf{J}_{t_y}\ \mathbf{J}_{t_z}
\bigr]
=
\begin{bmatrix}
-f_x x_n y_n & f_x(1+x_n^2) & -f_x y_n & f_x\rho & 0 & -f_x x_n\rho \\
-f_y(1+y_n^2) & f_y x_n y_n & f_y x_n & 0 & f_y\rho & -f_y y_n\rho
\end{bmatrix}.
$$

其中行对应 $(\mathrm{d}u,\mathrm{d}v)$，列对应 $(\omega_x,\omega_y,\omega_z,t_x,t_y,t_z)$。未写出的交叉项（如 $\partial v/\partial t_x$）在该针孔模型下为 $0$。

#### 由投影链式法则简述来源

刚体小扰动下相机系点变化 $\mathrm{d}\mathbf{P}=\delta\boldsymbol{\omega}\times\mathbf{P}+\delta\mathbf{t}$。针孔投影 $u=f_x X/Z+c_x$、$v=f_y Y/Z+c_y$ 的微分为

$$
\mathrm{d}u
=
f_x\frac{Z\,\mathrm{d}X-X\,\mathrm{d}Z}{Z^2},
\qquad
\mathrm{d}v
=
f_y\frac{Z\,\mathrm{d}Y-Y\,\mathrm{d}Z}{Z^2}.
$$

将 $\mathrm{d}X,\mathrm{d}Y,\mathrm{d}Z$ 用 $\delta\boldsymbol{\omega},\delta\mathbf{t}$ 展开并改用 $(x_n,y_n,\rho)$，即得上表各列。旋转项不含 $\rho$（只改变视线方向）；平移项均含 $\rho$（深度越近越敏感）。该乘积 $\mathbf{J}_f=\mathbf{J}_{\pi}\mathbf{J}_{\mathbf{P}}$ 即 §3.5 五步链的第 3 步，点位置协方差不另存矩阵。

#### 实现对照（本篇雅可比）

| 公式量 | 代码 |
|--------|------|
| $\mathbf{J}_f$ 各列 | `sio_geo_flow::ComputeFlowJac`（`include/internal/geo_flow_error.h`） |
| $P^-\to P_{\mathrm{cam}}$（当前预测位姿 $6\times 6$，相机系） | `ImuStatePoseCovInCamera`（`include/utils.h`） |
| 平移对角下界 $\sigma_{t,\min}^2$ | `MakePoseCovCam` / `kMinSigmaTransM` |
| $\Sigma_f=\mathbf{J}_f P_{\mathrm{cam}}\mathbf{J}_f^\top$ | `AccumulateSigmaFFromPoseCov` |
| $s=\mathbf{r}_f^\top\tilde\Sigma_f^{-1}\mathbf{r}_f$ | `MahalanobisSq` |
| 动态掩码主流程 | `src/dynamic_cloud_processor.cpp` |

#### 为何不能用固定像素阈值（推导动机）

设仅有偏航误差 $\sigma_{\omega_z}$，光心附近 $x_n\approx y_n\approx 0$ 时 $\|\mathbf{J}_{\omega_z}\sigma_{\omega_z}\|$ 仍可产生水平流误差，且**与深度无关**。  
设仅有前向平移误差 $\sigma_{t_z}$，则误差幅值 $\propto \rho=1/Z$，**近处放大**。  
因此固定 $\tau$（px）无法同时适配远近与旋转/平移主导情形，必须用协方差椭圆。下节给出各轴引起的光流量级与旋转–平移等价，作为门控半轴的物理刻度。

#### 位姿误差 / 协方差引起的光流变化（量级）

一阶下，相机系位姿微扰 $\boldsymbol{\xi}_{\mathrm{cam}}=(\delta\boldsymbol{\omega},\delta\mathbf{t})$ 引起的几何光流变化即

$$
\delta\mathbf{f}
=
\begin{pmatrix}\delta u\\ \delta v\end{pmatrix}
=
\mathbf{J}_f\boldsymbol{\xi}_{\mathrm{cam}}.
$$

其中 $\delta\mathbf{f}$ 的单位为 px；其协方差正是 §3.5 的 $\Sigma_f=\mathbf{J}_f P_{\mathrm{cam}}\mathbf{J}_f^\top$。故「位姿不确定度有多大」与「光流残差容差应放多宽」由同一雅可比连通。下列量级取近光心（$x_n,y_n\approx 0$）、$f\approx f_x\approx f_y$、特征深度 $Z$（可用深度中位数），与 [flow_pose_error_sensitivity.md](flow_pose_error_sensitivity.md) 的一阶结论一致；端点误差 $\mathrm{EPE}=\|\delta\mathbf{f}\|$。

| 误差类型 | 特征 EPE（量级） | 单位换算 |
|----------|------------------|----------|
| 绕相机 $X/Y$ 旋转 $\theta$（rad） | $\sim f\,\theta$ | $\sim f\cdot(\pi/180)$ **px/°** |
| 绕光轴 $Z$ 旋转 | 中心 $\sim 0$；半视场角 $\alpha$ 处 $\sim f\alpha\,\theta$ | 通常弱于绕 $X/Y$ |
| 横向平移 $\delta t_\perp$ | $\sim f\,\delta t_\perp/Z$ | $\sim f/(100Z)$ **px/cm** |
| 前向平移 $\delta t_z$ | 中心 $\sim 0$；边缘 $\sim f\,\|x_n\|\,\delta t_z/Z$ | 弱于同量级横向平移 |

近光心灵敏度排序（同「看起来不大」的物理单位下）：

$$
\omega_y,\ \omega_x
\;\gg\;
\delta t_x,\ \delta t_y
\;\gg\;
\omega_z,\ \delta t_z.
$$

**旋转–平移等价。** 使绕 $Y$（或 $X$）旋转 $\theta$（rad）与横向平移 $\delta t_\perp$ 产生同量级光流误差：

$$
f\,\theta
\approx
\frac{f\,\delta t_\perp}{Z}
\quad\Rightarrow\quad
\delta t_\perp\approx Z\,\theta
\quad\Rightarrow\quad
1^\circ
\;\Leftrightarrow\;
\delta t_\perp\approx Z\cdot\frac{\pi}{180}\approx 0.0175\,Z.
$$

| 特征深度 $Z$ | $1^\circ$ 等价横向平移 |
|--------------|----------------------|
| 1 m | ≈ 1.7 cm |
| 3 m | ≈ 5.3 cm |
| 5 m | ≈ 8.7 cm |
| 10 m | ≈ 17.5 cm |

**随深度的定性行为。** 绕 $X/Y$ 的旋转误差：EPE **几乎不随 $Z$ 变**；横向平移：EPE $\propto 1/Z$，**远处钝感**；前向平移 / 绕 $Z$：中心弱，整体偏弱。同一套位姿误差：近处平移贡献相对更明显；远处更接近「旋转误差决定光流误差」。

**不确定度映射（门控半轴刻度）。** 记 $\sigma_\omega$、$\sigma_t$ 为 $P_{\mathrm{cam}}$ 对角上的旋转 / 平移标准差，映到像素光流标准差 $\sigma_f$ 时，近光心有

$$
\sigma_f^{(\omega)}\sim f\,\sigma_\omega,
\qquad
\sigma_f^{(t)}\sim\frac{f\,\sigma_t}{Z}.
$$

例：取本篇算例内参 $f=370\,\mathrm{px}$，$\sigma_{\omega_y}=0.01\,\mathrm{rad}$（≈ $0.57^\circ$）$\Rightarrow\sigma_f^{(\omega)}\approx 3.7\,\mathrm{px}$；$\sigma_{t_y}=0.02\,\mathrm{m}$、$Z=2\,\mathrm{m}$$\Rightarrow\sigma_f^{(t)}\approx 3.7\,\mathrm{px}$（与 §3.5 数值算例一致）。深度误差主要污染平移诱导流（相对深度误差 $\varepsilon=\delta Z/Z$ 与平移流同阶耦合），不另写入 $P_{\mathrm{cam}}$，由视差有效性与脊 $\tau_0$ 吸收。

对动态滤波的直接含义：固定 $\tau$ 无法同时匹配「远处仍有 $f\theta$ 的旋转流误差」与「近处 $f\sigma_t/Z$ 被放大的平移流误差」；门控半轴必须随 $P_{\mathrm{cam}}$ 与 $Z$ 变——策略细则见 §3.6。

---

### 3.5 协方差链：从关键帧相对位姿协方差到门控

门控不确定度按一条链传递：关键帧相对位姿协方差 → 相机系位姿协方差 $P_{\mathrm{cam}}$ → 几何投影点协方差 → 光流协方差 $\Sigma_f$ → 马氏门控。五步写在下面。几何流只依赖 $T_{c\leftarrow k}$，所以进入 $\mathbf{J}_f$ 的是这份相对协方差，不是滤波的绝对当前协方差。$A$ 的分块、交叉项来源、平移地板的近场含义与数值算例仍在本节后文；完整判定见 §3.6。

#### 五步链

**（1）关键帧相对协方差（并行副本）。** 主滤波的绝对协方差（`imu_states_.cov` 与 `imu_pred_state_curr_.cov`）照常传播，不在关键帧处置零。光流门控另持一份同维并行协方差 `flow_gate_cov_`（$21\times 21$）。参考关键帧成立的瞬间，这份矩阵整块置零：该时刻的关键帧状态对门控而言是已知的，此后累积的是相对该关键帧的不确定度，即 $\mathrm{Cov}(T_{c\leftarrow k})$ 在误差状态中的对应量。不从绝对协方差里减去 $P_k$，也不使用两帧交叉 $P_{k,c}$。

误差状态仍是 $21$ 维：前 $6$ 维为位姿 $(\delta\boldsymbol{\theta},\delta\mathbf{p})$（旋转误差在 IMU 机体系，位置误差在世界系），其后依次为速度、陀螺偏置、加计偏置、重力与曝光余量。置零之后，主滤波在这一关键帧与当前帧之间每一次改协方差的步骤，都作用到这份并行矩阵上，线性化与主滤波同一次计算。IMU 预测与 `ImuPreintegrationCov` 写在同一次循环里，$\Phi$ 与 $Q_d$ 跟主滤波这一拍相同，$P$ 用并行副本自己的先验。这不是从关键帧另起一份只含 IMU、跳过中间 GICP 的预积分：

$$
P
\leftarrow
\Phi P\Phi^{\top}+Q_d,
\qquad
\Phi=F_d=I+F\,\mathrm{d}t,
\qquad
Q_d=G Q_c G^{\top}\,\mathrm{d}t.
$$

其中 $F$ 为 $21\times 21$ 连续误差状态转移，$G$ 把陀螺、加计及其偏置游走噪声映入状态，$Q_c$ 为该项连续噪声协方差；与主滤波的 $P\leftarrow\Phi P\Phi^{\top}+Q$ 是同一次传播，不另写一套噪声模型。有位姿观测时，用主滤波这一步已经算出的 $K$、$H$、$R$，对并行 $P$ 做同一个 Joseph 更新（非关键帧 `ESKF` 的 GICP 更新，以及关键帧 `ESIKF` 的位姿更新）：

$$
P
\leftarrow
(I-KH)\,P\,(I-KH)^{\top}
+
K R K^{\top}.
$$

没有位姿更新的步骤，并行协方差只按预测增长。门控读取的是当前帧预测之后、本帧 GICP / ESIKF 之前的这份矩阵。新关键帧写入之后，这份矩阵再次置零，下一段相对协方差从新的参考帧重新累积。速度、偏置与位姿的交叉不另做舒尔补，它们已经通过 $\Phi$ 和 $K$ 进入左上 $6\times 6$：

$$
P_{\xi}^{\mathrm{rel}}
=
\begin{bmatrix}
P_{\theta\theta}^{\mathrm{rel}} & P_{\theta p}^{\mathrm{rel}} \\
P_{p\theta}^{\mathrm{rel}} & P_{pp}^{\mathrm{rel}}
\end{bmatrix}.
$$

$T_{c\leftarrow k}$ 的均值仍由世界系名义位姿复合，供第 3 步的投影使用。

**（2）相机系位姿协方差 $P_{\mathrm{cam}}$。** 把 $\boldsymbol{\xi}=(\delta\boldsymbol{\theta},\delta\mathbf{p}_w)$ 变到左目相机系微扰 $\boldsymbol{\xi}_{\mathrm{cam}}=(\delta\boldsymbol{\omega},\delta\mathbf{t})$。线上路径对并行协方差的位姿块做这一次帧变换（`FlowGatePoseCovInCamera`；其中的 $A$ 与 `ImuStatePoseCovInCamera` 相同，旋转取当前预测的名义姿态）。绝对 `imu_pred_state_curr_.cov` 不进入这一步：

$$
P_{\mathrm{cam}}
=
A P_{\xi}^{\mathrm{rel}} A^{\top}.
$$

$A$ 含外参旋转 $R_{ic}^{\top}$ 与杠杆臂 $-R_{ic}^{\top}[\mathbf{t}_{ic}]_{\times}$，分块见下节。得到 $P_{\mathrm{cam}}$ 之后，`MakePoseCovCam` 只抬平移对角；这一下界是工程地板，不是 $\Phi P\Phi^{\top}$ 的输出，交叉项保持原值：

$$
(P_{\mathrm{cam}})_{t_i t_i}
\leftarrow
\max\bigl((P_{\mathrm{cam}})_{t_i t_i},\,\sigma_{t,\min}^2\bigr),
\qquad
\sigma_{t,\min}=0.02\,\mathrm{m}.
$$

**（3）几何投影点协方差。** 名义投影 $\mathbf{P}'=R\mathbf{P}+t$、$\mathbf{u}'=\pi(\mathbf{P}')$ 只决定 $\mathbf{f}_{\mathrm{geo}}$ 的均值。投影点相对相机系位姿的协方差不单独存成矩阵，它是 $\mathbf{J}_f$ 内部的中间映射：微扰先推动三维点，针孔微分再把它映到像素。与 §3.4 和 `ComputeFlowJac` 相同，线性化点是参考点 $\mathbf{P}=(X,Y,Z)^{\top}$ 的 $(x_n,y_n,\rho)$，不在投影后的 $\mathbf{P}'$ 上重算 $\mathbf{J}_{\pi}$。

$$
\mathrm{d}\mathbf{P}
=
\delta\boldsymbol{\omega}\times\mathbf{P}+\delta\mathbf{t}
=
\mathbf{J}_{\mathbf{P}}\boldsymbol{\xi}_{\mathrm{cam}},
\qquad
\mathbf{J}_{\mathbf{P}}
=
\bigl[-[\mathbf{P}]_{\times}\ \ I_3\bigr]\in\mathbb{R}^{3\times 6},
$$

$$
\mathbf{J}_{\pi}
=
\begin{bmatrix}
f_x\rho & 0 & -f_x x_n\rho \\
0 & f_y\rho & -f_y y_n\rho
\end{bmatrix},
\qquad
\mathbf{J}_f
=
\mathbf{J}_{\pi}\mathbf{J}_{\mathbf{P}}.
$$

点位置协方差 $\Sigma_{\mathbf{P}}=\mathbf{J}_{\mathbf{P}} P_{\mathrm{cam}}\mathbf{J}_{\mathbf{P}}^{\top}$ 与投影像素协方差 $\mathbf{J}_{\pi}\Sigma_{\mathbf{P}}\mathbf{J}_{\pi}^{\top}$ 都不保存；`ComputeFlowJac` 直接写出乘积 $\mathbf{J}_f$（即 §3.4 各列，旋转列中的 $x_n y_n$ 来自 $\mathbf{J}_{\pi}(-[\mathbf{P}]_{\times})$）。深度噪声不写入 $\Sigma_{\mathbf{P}}$，仍由视差有效性与脊 $\tau_0$ 吸收。

**（4）光流协方差。** 参考像素 $\mathbf{u}$ 取自视差图索引，不作为随机变量，故几何光流与投影像素共用位姿传播项

$$
\Sigma_f
=
\mathbf{J}_f P_{\mathrm{cam}}\mathbf{J}_f^{\top}
=
\mathbf{J}_{\pi}\Sigma_{\mathbf{P}}\mathbf{J}_{\pi}^{\top}.
$$

与这一传播项分开的两处工程量：平移对角地板已在第 2 步写入 $P_{\mathrm{cam}}$，经 $\mathbf{J}_f$ 进入 $\Sigma_f$；各向同性脊加在 $\Sigma_f$ 之后（`AddIsotropicRidge`），只抬像素对角，不动 $\Sigma_{uv}$：

$$
\tilde\Sigma_f
=
\Sigma_f+\tau_0^2 I_2,
\qquad
\tau_0\approx 1.5\,\mathrm{px}.
$$

**（5）门控。** 神经光流与几何光流之差对加脊后的 $\tilde\Sigma_f$ 做马氏距离平方，门限为 $1$（`MahalanobisSq`）：

$$
\mathbf{r}_f
=
\mathbf{f}_{\mathrm{raft}}-\mathbf{f}_{\mathrm{geo}},
\qquad
s
=
\mathbf{r}_f^{\top}\tilde\Sigma_f^{-1}\mathbf{r}_f,
\qquad
s\ge 1 \Rightarrow \mathbf{u}\in\mathcal{D}.
$$

$s\ge 1$ 表示残差落在由 $\tilde\Sigma_f$ 定义的 $1\sigma$ 椭圆外，判为动态。双流跳过与形态学见 §3.6。

#### 并行协方差进入 $P_{\mathrm{cam}}$（即 $\mathrm{Cov}(T_{c\leftarrow k})$）

上列第 2 步的 $P_{\xi}^{\mathrm{rel}}$ 与 $A$ 展开如下。误差状态为混合坐标：旋转误差 $\delta\boldsymbol{\theta}$ 在 IMU 机体系，位置误差 $\delta\mathbf{p}$ 在世界系。取并行协方差的位姿 $6\times 6$ 块

$$
P_{\xi}^{\mathrm{rel}}
=
\begin{bmatrix}
P_{\theta\theta}^{\mathrm{rel}} & P_{\theta p}^{\mathrm{rel}} \\
P_{p\theta}^{\mathrm{rel}} & P_{pp}^{\mathrm{rel}}
\end{bmatrix},
\qquad
\boldsymbol{\xi}
=
\begin{bmatrix}\delta\boldsymbol{\theta}\\\delta\mathbf{p}_w\end{bmatrix},
$$

再变到左目相机系微扰 $\boldsymbol{\xi}_{\mathrm{cam}}=(\delta\boldsymbol{\omega},\delta\mathbf{t})$。相机原点 $p_{\mathrm{wc}}=p_{\mathrm{wb}}+R_{wb}\mathbf{t}_{ic}$，故旋转误差经杠杆臂进入相机平移：

$$
\delta\boldsymbol{\omega}=R_{ic}^{\top}\delta\boldsymbol{\theta},
\qquad
\delta\mathbf{t}
=
R_{w\leftarrow c}^{\top}\delta\mathbf{p}_w
-
R_{ic}^{\top}[\mathbf{t}_{ic}]_{\times}\delta\boldsymbol{\theta},
\qquad
A
=
\begin{bmatrix}
R_{ic}^{\top} & \mathbf{0} \\
-R_{ic}^{\top}[\mathbf{t}_{ic}]_{\times} & R_{w\leftarrow c}^{\top}
\end{bmatrix},
$$

$$
P_{\mathrm{cam}}
=
A P_{\xi}^{\mathrm{rel}} A^{\top}.
$$

$A$ 的左下块即使 $P_{\theta p}^{\mathrm{rel}}=\mathbf{0}$ 也会在 $P_{\mathrm{cam}}$ 中产生 $P_{\omega t}$（例如机体系前向基线 + yaw $\to$ 相机侧向平移）。$P_{\theta p}^{\mathrm{rel}}$ 与轴间非对角经 $A$ 一并转入，不强制对角化。

线上路径**不手写** $\rho$。`FlowGatePoseCovInCamera` 取并行协方差左上 $6\times 6$ 再乘 $A$。交叉项是这份相对误差的相关信念，不是法兰把俯仰和侧移「绑在一起」。来源只有两层：

1. **并行 $P$ 里已经有。** 从关键帧处的零矩阵出发，IMU 传播 $P\leftarrow \Phi P\Phi^{\top}+Q$：姿态误差会通过比力/重力走进速度再走进位置（俯仰错 $\leftrightarrow$ 前后/高低位置错），于是 $P_{\theta p}\neq 0$；轴间也会被 $\Phi$ 填满。GICP / ESIKF 的 Joseph 更新把同一个 $R$ 经主滤波的 $K$ 写回并行 $P$，杠杆块里的交叉留在后验里，并被下一帧的预测带到 $P_{\xi}^{\mathrm{rel}}$。偏置、速度与位姿的交叉在传播里再漏回 $6\times 6$。
2. **$A$ 又造一批。** 同一刚体误差换到相机切空间：$\delta\mathbf{t}$ 含 $-R_{ic}^{\top}[\mathbf{t}_{ic}]_{\times}\delta\boldsymbol{\theta}$。即使 $P_{\xi}$ 对角，只要 $\|\mathbf{t}_{ic}\|>0$，相机系就会出现 $\mathrm{Cov}(\omega_z,t_y)$ 这类项。$R_{ic}^{\top}P_{\theta\theta}R_{ic}$ 还会把机体轴相关拧到相机轴上。

对应关系：$\mathrm{Corr}(\omega_y,t_y)=0.9$ 表示「当前这次误差更可能同号出现俯仰+侧移，而不是反号」，不是电机耦合。步 2 把这一项写成 $0.9$ 只为探针椭圆斜不斜；步 3 与步 5 用这份相对 $P_{\mathrm{cam}}$，相关可正可负、可大可小，由关键帧之后的传播和 GICP 更新决定。$\mathbf{J}_f$ 再把 $P_{\mathrm{cam}}$ 映到 $\Sigma_{uv}$：即使 $P$ 对角，偏轴点的 $\omega_x,t_z$ 也能让椭圆斜——那是像素几何，不是 $P$ 的交叉。

平移对角再设下界（不改交叉项）：

$$
(P_{\mathrm{cam}})_{t_i t_i}
\leftarrow
\max\bigl((P_{\mathrm{cam}})_{t_i t_i},\,\sigma_{t,\min}^2\bigr),
\qquad
\sigma_{t,\min}=0.02\,\mathrm{m}.
$$

其中：

- $R_{ic},\mathbf{t}_{ic}$：IMU←相机外参（$p_{\mathrm{body}}=R_{ic}p_{\mathrm{cam}}+\mathbf{t}_{ic}$）；
- $R_{w\leftarrow c}=R_{wb}R_{ic}$：当前左目相对世界的旋转；
- $P_{\xi}^{\mathrm{rel}}$：并行协方差的位姿 $6\times 6$。这份矩阵在参考关键帧时刻为零，其后与主滤波同步做 IMU 预测和 GICP / ESIKF 更新；进入 $\mathbf{J}_f$ 的就是它，不是 `imu_pred_state_curr_.cov`；
- $P_{\theta p}^{\mathrm{rel}}$：旋转–平移交叉协方差；与杠杆臂一起使光流椭圆可以相对像素轴倾斜；
- $\sigma_{t,\min}$：平移标准差下界，**只在近处绑住**。点的三维度误差 $\delta\mathbf{P}=\delta\boldsymbol{\omega}\times\mathbf{P}+\delta\mathbf{t}$，远处 $\|\delta\boldsymbol{\omega}\times\mathbf{P}\|\propto Z$，旋转项已经很大，$1$–$2\,\mathrm{cm}$ 的 $\delta\mathbf{t}$ 可忽略；映到像素后平移贡献 $\propto 1/Z$，远处本来就交给旋转 $+\tau_0$。并行协方差收到毫米后，真正会被地板撑开的是近点（$Z\sim 1$–$2\,\mathrm{m}$），防止近处静态过杀。$2\,\mathrm{cm}$ 是近场工程默认；$1\,\mathrm{cm}$ 同属近场选择，步 1 在近距离静态上看 $s$ vs $\chi^{2}_{2}$，远处不要用这个数解释漏检/过杀。只截断平移对角，不抬旋转。

$\mathbf{J}_f$ 把相机系位姿微扰映射到光流；$P_{\mathrm{cam}}$ 是并行协方差经 $A$ 之后的完整 $6\times 6$，也就是 $\mathrm{Cov}(T_{c\leftarrow k})$ 在相机切空间中的表示。关键帧之前的绝对不确定度不进入：并行副本在关键帧处置零，而不是把 $P_k$ 从绝对 $P_c$ 里减掉，两帧交叉 $P_{k,c}$ 也不用。关键帧之后它随预测增长，并被中间帧的 GICP 更新压回，因此不是跳过视觉更新的开环预积分。全球漂留在主滤波的绝对协方差里，不把椭圆撑宽。

#### 一阶映射（保留交叉项）

第 4 步的位姿传播项按分量写开。$\Sigma_{\mathbf{P}}$ 不单独存储，直接从 $\mathbf{J}_f$ 与 $P_{\mathrm{cam}}$ 相乘：

$$
\Sigma_f
=
\mathbf{J}_f P_{\mathrm{cam}} \mathbf{J}_f^{\top}.
$$

若 $P_{\mathrm{cam}}$ 退化为对角 $\mathrm{diag}(\sigma_i^2)$，则上式还原为独立轴外积 $\sum_i\sigma_i^2\mathbf{J}_{f,i}\mathbf{J}_{f,i}^{\top}$。一般情形下 $P_{\theta p}$、轴间相关以及杠杆臂使 $(\Sigma_f)_{uv}\neq 0$ 即使单个 $\mathbf{J}_{f,i}$ 只贡献水平或竖直流，椭圆不必与像素轴对齐。

按分量（与代码 `AccumulateSigmaFFromPoseCov` 一致）：令 $j_u^{T}$、$j_v^{T}$ 为 $J_f$ 的两行，

$$
\Sigma_{uu}=j_u^{T}P_{cam}j_u,
\qquad
\Sigma_{uv}=j_u^{T}P_{cam}j_v,
\qquad
\Sigma_{vv}=j_v^{T}P_{cam}j_v.
$$

$P_{cam}$ 对角时，$\Sigma_{uv}$ 只来自「同一轴同时推 $u$ 和 $v$」（偏轴点的 $\omega_x$、$t_z$ 等，属像素几何）。$P_{cam}$ 有交叉时，多出双线性项：轴 $i$ 的流贡献与轴 $j$ 的流贡献按 $\mathrm{Cov}(\xi_i,\xi_j)$ 相乘，椭圆相对像素轴倾斜。

#### 交叉项如何改门控

名义 $T_{ck}$ 决定 $f_{geo}$（椭圆中心）；$P_{cam}$ 决定长短轴和朝向。交叉是滤波对**误差同号出现**的信念，不是电机把俯仰和侧移绑在一起。

光心附近只点亮 $\omega_y$ 与 $t_y$，且 $\mathrm{Corr}(\omega_y,t_y)=\rho_{\omega t}$（步 2 取 $0.9$；此处 $\rho_{\omega t}$ 不是逆深度）：

$$
\frac{\partial u}{\partial\omega_y}\approx f_x,
\qquad
\frac{\partial v}{\partial t_y}\approx\frac{f_y}{Z}.
$$

于是 $\Sigma_{uv}\neq 0$。对角对照（丢掉 $P_{\theta p}$）下 $\Sigma_{uv}=0$，同样欧氏长度的残差 $s$ 相同。马氏分数显式含交叉（§3.6）：

$$
s
=
\frac{
r_u^{2}\tilde{\Sigma}_{vv}
-2 r_u r_v\tilde{\Sigma}_{uv}
+r_v^{2}\tilde{\Sigma}_{uu}
}{
\tilde{\Sigma}_{uu}\tilde{\Sigma}_{vv}-\tilde{\Sigma}_{uv}^{2}
}.
$$

$\tilde{\Sigma}_{uv}$ 的符号决定哪条 $45^{\circ}$ 是长轴。$\rho_{\omega t}>0$ 时：

| 残差 | 落在哪 | $s$ | 门控含义 |
|------|--------|-----|----------|
| $r_{+}=(2.8,\,2.8)$ | 长轴 | 更小 | 「俯仰+侧移」是滤波认为会一起出现的误差，放行 |
| $r_{-}=(2.8,\,-2.8)$ | 短轴 | 更大 | 反号组合几乎不会由位姿误差产生，判动态 |

$|\rho_{\omega t}|$ 越大，长轴越长、短轴越短。丢掉 $P_{\theta p}$ 后 $s(r_{+})=s(r_{-})$，斜残差上 F1 下降。独立运动物体的 $r_f$ 方向一般不落在 $P_{cam}$ 预言的相关方向上，短轴一侧更容易 $s\ge 1$；相机位姿误差造成的残差应沿长轴，幅值不小也可能 $s<1$。这是交叉项相对「只取对角标准差、再画轴对齐椭圆」多出来的判别力。

对照三档：

| 门控 | 形状 | 缺什么 |
|------|------|--------|
| 固定阈值 $\tau$（px） | 圆 | 远近、旋转/平移用同一半径；旋转与 $Z$ 无关，平移 $\propto 1/Z$ |
| 对角 $P_{cam}$ | 轴对齐椭圆 | 光心附近不区分同号/反号斜残差 |
| 完整 $P_{cam}$（交叉+杠杆臂） | 可倾斜椭圆 | 主路径。平移地板只撑近处对角，不改交叉；若先写 $1\,\mathrm{mm}$ 交叉再被地板抬到 $2\,\mathrm{cm}$，有效 $|\rho_{\omega t}|$ 会被冲掉 |

#### 各向同性脊

第 4 步里加在位姿传播项之外的脊为

$$
\tilde\Sigma_f
=
\Sigma_f+\tau_0^2 I_2,
\qquad
\tau_0\approx 1.5\,\mathrm{px}.
$$

其中：

- $\tilde{\Sigma}_f$：用于门控的最终协方差；
- $\tau_0$：神经光流 / 量化等各向同性噪声标准差；
- $I_2$：$2\times 2$ 单位阵；脊保证 $\Sigma_f=0$ 时仍可求逆。

#### 数值算例：从 $P_{cam}$ 到椭圆再到 $s$

下列步 1–7 是本组数的演算顺序，不是上文五步链的编号。取光心一点，与步 2、单测 `PoseCrossTermTiltsFlowEllipseOffAxes` 同一组数。相机与 IMU 对齐、无杠杆臂，故 $A=I$、$P_{cam}=P_{\xi}$。内参 $f_x=f_y=370$，深度 $Z=2\,\mathrm{m}$（逆深度 $1/Z=0.5$），$x_n=y_n=0$。脊 $\tau_0=1.5\,\mathrm{px}$。

**步 1：写出 $P_{cam}$。** 只点亮俯仰 $\omega_y$ 与侧移 $t_y$，相关系数 $\rho_{\omega t}=0.9$：

$$
\sigma_{\omega_y}=0.01\,\mathrm{rad},\qquad
\sigma_{t_y}=0.02\,\mathrm{m},\qquad
\mathrm{Cov}(\omega_y,t_y)=0.9\times 0.01\times 0.02=1.8\times 10^{-4}.
$$

列序 $(\omega_x,\omega_y,\omega_z,t_x,t_y,t_z)$，其余元为 0：

$$
P_{cam}
=
\begin{bmatrix}
0 & 0 & 0 & 0 & 0 & 0 \\
0 & 10^{-4} & 0 & 0 & 1.8\times 10^{-4} & 0 \\
0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1.8\times 10^{-4} & 0 & 0 & 4\times 10^{-4} & 0 \\
0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}.
$$

**步 2：平移地板。** $\sigma_{t,\min}^2=4\times 10^{-4}$，对角 $t_y$ 已在下界上，交叉 $1.8\times 10^{-4}$ 不动。

**步 3：该像素的 $J_f$。** $x_n=y_n=0$ 时 §3.4 只剩四项非零：

$$
J_f
=
\begin{bmatrix}
0 & 370 & 0 & 185 & 0 & 0 \\
-370 & 0 & 0 & 0 & 185 & 0
\end{bmatrix}.
$$

含义：$\partial u/\partial\omega_y=f_x=370$，$\partial u/\partial t_x=\partial v/\partial t_y=f_x/Z=185$。两轴对像素的 $1\sigma$ 贡献碰巧相等：$0.01\times 370=3.7\,\mathrm{px}$，$0.02\times 185=3.7\,\mathrm{px}$。

**步 4：$\Sigma_f=J_f P_{cam} J_f^{T}$。** 只有 $\omega_y$ 推 $u$、$t_y$ 推 $v$，交叉把二者焊在一起：

$$
\begin{aligned}
\Sigma_{uu}
&=
370^{2}\times 10^{-4}
=
13.69,\\
\Sigma_{vv}
&=
185^{2}\times 4\times 10^{-4}
=
13.69,\\
\Sigma_{uv}
&=
370\times 185\times 1.8\times 10^{-4}
=
12.321.
\end{aligned}
$$

对角 $P$ 时 $\Sigma_{uv}=0$，椭圆轴对齐；这里 $\Sigma_{uv}/\sqrt{\Sigma_{uu}\Sigma_{vv}}=0.9$，与 $\rho_{\omega t}$ 相同。

**步 5：加脊。** $\tau_0^{2}=2.25$，

$$
\tilde{\Sigma}_f
=
\begin{bmatrix}
15.94 & 12.321 \\
12.321 & 15.94
\end{bmatrix}.
$$

**步 6：椭圆轴。** $2\times 2$ 对称阵的主轴转角（从 $u$ 轴转到长轴，逆时针为正）

$$
\theta
=
\frac{1}{2}
\operatorname{atan2}\bigl(2\tilde{\Sigma}_{uv},\;\tilde{\Sigma}_{uu}-\tilde{\Sigma}_{vv}\bigr).
$$

本例 $\tilde{\Sigma}_{uu}=\tilde{\Sigma}_{vv}=15.94$，分母为 $0$；$2\tilde{\Sigma}_{uv}>0$，故 $\operatorname{atan2}(+24.64,\,0)=+90^{\circ}$，$\theta=45^{\circ}$。特征值在等对角时退化为 $\lambda_{\pm}=\tilde{\Sigma}_{uu}\pm\tilde{\Sigma}_{uv}=15.94\pm 12.321$；长轴特征向量 $(1,1)$，短轴 $(1,-1)$。一般情形 $\tilde{\Sigma}_{uu}\neq\tilde{\Sigma}_{vv}$ 时 $\theta$ 不是 $45^{\circ}$，半轴仍是 $\sqrt{\lambda_{\pm}}$，

$$
\lambda_{\pm}
=
\frac{\tilde{\Sigma}_{uu}+\tilde{\Sigma}_{vv}}{2}
\pm
\sqrt{
\Bigl(\frac{\tilde{\Sigma}_{uu}-\tilde{\Sigma}_{vv}}{2}\Bigr)^{2}
+\tilde{\Sigma}_{uv}^{2}
}.
$$

本例数值：

| 轴 | 特征值 | $1\sigma$ 半轴 $\sqrt{\lambda}$ | 方向 |
|----|--------|--------------------------------|------|
| 长轴 | $28.26$ | $5.32\,\mathrm{px}$ | $(1,1)$，同号俯仰+侧移 |
| 短轴 | $3.62$ | $1.90\,\mathrm{px}$ | $(1,-1)$，反号 |

固定阈值是半径 $1.5\,\mathrm{px}$ 的圆；对角 $P$ 加脊后是半径 $\sqrt{15.94}\approx 4.0\,\mathrm{px}$ 的圆（本例两轴贡献相等）。完整 $P$ 把容差沿相关方向放到 $5.32\,\mathrm{px}$，垂直方向收到 $1.90\,\mathrm{px}$。

**步 7：两个等长残差的 $s$。** $\|r_{+}\|=\|r_{-}\|=2.8\sqrt{2}\approx 3.96\,\mathrm{px}$。$\Delta=15.94^{2}-12.321^{2}\approx 102.28$，

$$
s(r_{+})
=
\frac{2\times 15.94\times 7.84-2\times 12.321\times 7.84}{\Delta}
\approx 0.55
<1
\quad\text{（长轴，放行）},
$$

$$
s(r_{-})
=
\frac{2\times 15.94\times 7.84+2\times 12.321\times 7.84}{\Delta}
\approx 4.33
\ge 1
\quad\text{（短轴，判动态）}.
$$

丢掉交叉后 $\tilde{\Sigma}_{uv}=0$，两者都是 $15.68/15.94\approx 0.98$，分不开。固定 $\tau=1.5\,\mathrm{px}$ 则 $3.96>1.5$，两者都杀。

有杠杆臂时，即使 $P_{\xi}$ 对角，$A$ 也会先在 $P_{cam}$ 里造出 $\mathrm{Cov}(\omega_z,t_y)$ 之类项，再重复步 3–7；主路径不手写 $\rho_{\omega t}$，交叉来自滤波 $P^{-}$ 与 $A$。

---

### 3.6 马氏判别与完整判定逻辑

五步链的第 5 步展开如下。

#### 马氏距离平方

$$
s(\mathbf{u})
=
\mathbf{r}_f^\top\tilde\Sigma_f^{-1}\mathbf{r}_f.
$$

推导：若 $\mathbf{r}_f\sim\mathcal{N}(\mathbf{0},\tilde\Sigma_f)$，则 $s$ 服从自由度为 2 的 $\chi^2$ 分布；取门限 $1$ 对应「落在由 $\tilde\Sigma_f$ 定义的 $1\sigma$ 椭圆外」的工程门控（与实现 `MahalanobisSq` 一致）。

二维显式求逆（$\Delta=\tilde\Sigma_{uu}\tilde\Sigma_{vv}-\tilde\Sigma_{uv}^2$）：

$$
\tilde\Sigma_f^{-1}
=
\frac{1}{\Delta}
\begin{bmatrix}
\tilde\Sigma_{vv} & -\tilde\Sigma_{uv} \\
-\tilde\Sigma_{uv} & \tilde\Sigma_{uu}
\end{bmatrix},
$$

$$
s
=
\frac{
r_u^2\tilde\Sigma_{vv}
-2 r_u r_v\tilde\Sigma_{uv}
+r_v^2\tilde\Sigma_{uu}
}{\Delta}.
$$

其中 $\tilde{\Sigma}_{uu},\tilde{\Sigma}_{uv},\tilde{\Sigma}_{vv}$ 为 $\tilde{\Sigma}_f$ 的元素；$\Delta$ 为行列式。交叉项只通过 $\tilde{\Sigma}_{uv}$ 进入 $s$：同号斜残差减 $s$，反号加 $s$，椭圆因而倾斜（§3.5）。

#### 动态集合

$$
\mathbf{u}\in\mathcal{D}
\quad\Leftrightarrow\quad
s(\mathbf{u})\ge 1.
$$

其中 $\mathcal{D}$ 为动态像素集。门限固定为 $1$（$1\sigma$ 椭圆外）；**自适应不改 $\chi^2$ 分位数，而改椭圆本身**——即把 $P_{\mathrm{cam}}$ 经 $\mathbf{J}_f$ 映成像素容差（§3.4 量级 + §3.5 传播）。

#### 由位姿协方差驱动的门控策略

门控要同时完成两件事：（i）位姿误差引起的残差应尽量 $s<1$（静态过杀低）；（ii）独立运动引起的残差应尽量 $s\ge 1$（动态漏检低）。二者的物理刻度正是 §3.4 的 $\sigma_f^{(\omega)}\sim f\sigma_\omega$ 与 $\sigma_f^{(t)}\sim f\sigma_t/Z$。

**对角主导时的像素半轴（光心附近，暂略交叉与脊）。** 对水平分量，

$$
\sigma_u
\approx
\sqrt{
f_x^2\sigma_{\omega_y}^2
+(f_x\rho)^2\sigma_{t_x}^2
+\cdots
},
$$

竖直分量同理（$\omega_x$、$t_y$ 主导）。加脊后有效半轴 $\tilde\sigma_u=\sqrt{\sigma_u^2+\tau_0^2}$。交叉项再把轴对齐椭圆拧斜（§3.5），不改变「远近 / 旋转平移」的主导排序。

**随深度收紧 / 放宽（同一帧内）。**

| 区域 | $P_{\mathrm{cam}}$ 映到 $\Sigma_f$ 的主导项 | 门控含义 |
|------|---------------------------------------------|----------|
| 近处（$Z\sim 1$–$2\,\mathrm{m}$） | 平移 $\propto 1/Z$ 放大；旋转仍 $\sim f\sigma_\omega$ | 椭圆宜宽；若 $P^-$ 已收到毫米级，必须靠 $\sigma_{t,\min}$ 撑开平移对角，否则静态过杀 |
| 远处 | 平移钝化；旋转半轴 $\approx f\sigma_\omega$ **几乎不随 $Z$ 变** | 勿因「几何流幅值小」把容差收到固定小 $\tau$；远处漏检往往来自旋转容差不足被误判为「流已对齐」 |
| 像面边缘 | $\omega_z$、$t_z$ 与 $x_n,y_n$ 耦合增强 | 半轴略增；边界连通域清理另抑出入视野伪影 |

**随滤波状态收紧 / 放宽（帧间）。**

| $P_{\mathrm{cam}}$ 状态 | 椭圆 | 策略解读 |
|-------------------------|------|----------|
| 开环漂大 / 视觉长期未更新 | 整体偏宽 | 偏**漏检**：宁可放过可疑点，避免用过窄门控在漂位姿下误删静态结构 |
| 收敛良好（$\sigma_\omega$、$\sigma_t$ 小） | 自动收窄 | 偏**敏感**：慢速 / 远距弱动态更容易 $s\ge 1$；此时脊 $\tau_0$ 防止 $\Sigma_f\to 0$ 时数值过杀 |
| 仅平移对角被 $\sigma_{t,\min}$ 抬起 | 近处变宽、交叉 $\|\rho_{\omega t}\|$ 可能被冲淡 | 地板只服务近场；远处解释漏检/过杀时不要拿 $2\,\mathrm{cm}$ 当地主因（§3.5） |

**方向性门控（交叉项）：静态松、动态紧。**  
位姿误差残差应落在 $\mathbf{J}_f$ 与 $P_{\mathrm{cam}}$ 预言的相关锥内（长轴，$s$ 更小）；独立运动的 $\mathbf{r}_f$ 方向一般不落在该锥内（短轴，$s$ 更大）。因此完整 $P_{\mathrm{cam}}$ 相对对角 $P$：对「像相机微扰」的静态斜残差更松，对「反号 / 非刚体」组合更紧——这是交叉项多出来的判别力，不是改门限。

**与固定阈值、双流跳过的分工。**

1. **固定 $\tau$（px）**：近处等价过紧（$f\sigma_t/Z$ 常 $> \tau$）、远处对旋转又常过松或过紧失调——§3.4 的 $1^\circ\leftrightarrow 0.0175\,Z$ 说明「一度 vs 一厘米」不能共用一个像素半径。  
2. **马氏 $s\ge 1$**：门限不变，半轴随 $f$、$Z$、$P_{\mathrm{cam}}$ 变；实现上不另设随深度的启发式 $\tau(Z)$。  
3. **脊 $\tau_0\approx 1.5\,\mathrm{px}$**：吸收神经光流 / 量化底噪，**不能**替代 $\sigma_{t,\min}$ 的近场平移地板。  
4. **双流跳过**（$\|\mathbf{f}_{\mathrm{geo}}\|^2$ 与 $\|\mathbf{f}_{\mathrm{raft}}\|^2$ 同时 $<\tau_{\mathrm{skip}}$）：处理「相机近似静止」——此时 $\mathbf{f}_{\mathrm{geo}}\approx\mathbf{0}$，但椭圆仍可能很窄；若只看几何流跳过会漏检仍在动的物体。跳过的是检验，不是把 $P_{\mathrm{cam}}$ 置零。

**实操核对（与步 1、步 2 一致）。**  
全静态、门控用 $P_{\mathrm{arm}}$ 时：$s$ 应接近 $\chi^2_2$（均值约 $2$），$s<1$ 比例与 $1\sigma$ 门限一致，说明半轴没有系统性偏窄/偏宽。注入单轴 $P$ 时：旋转主导残差幅值应与 $Z$ 关系弱，平移主导应近大远小；$\mathrm{Corr}(\omega_y,t_y)>0$ 时 $s(\mathbf{r}_+)<s(\mathbf{r}_-)$。动态序列上再看掩码 F1——椭圆校准与分类是两层问题，不要用 IoU 反推半轴公式。

#### 附加条件（推导理由）

1. **视差有效**：$d>d_{\min}$，且深度在跟踪量程内，否则无法可靠定义 $\mathbf{f}_{\mathrm{geo}}$ 与 $\rho$。
2. **双流跳过规则**：仅当
   $$
   \|\mathbf{f}_{\mathrm{geo}}\|^2<\tau_{\mathrm{skip}}
   \ \wedge\
   \|\mathbf{f}_{\mathrm{raft}}\|^2<\tau_{\mathrm{skip}}
   $$
   （如 $\tau_{\mathrm{skip}}=0.25\,\mathrm{px}^2$）时跳过检验。  
   理由：若只在 $\|\mathbf{f}_{\mathrm{geo}}\|$ 很小时跳过，则相机近似静止时运动物体的大 $\mathbf{f}_{\mathrm{raft}}$ 会被漏检。

#### 后处理与静态索引

对动态掩码做腐蚀–膨胀；对触及顶/左/右边界的连通域清理边界条带（保留底边），抑制出入视野伪影。若参考点投影落入 $\mathcal{D}$，则从 $\mathcal{S}$ 剔除。投影后超出当前图像、位于相机后方，或被当前视图更近表面遮挡的点标为不可见，单独输出，不写入 $\mathcal{S}$，也不标为动态；每个当前像素只检验 $Z'$ 最近的点。没有有效视差索引的点标为不可见（左右目不重叠），同样单独输出，既不是静态也不是动态。

---

### 3.7 算法流程（与公式对应）

协方差的五步链见 §3.5。与公式对应的计算顺序为：

1. 由 IMU 预测复合 $T_{c\leftarrow k}$；由关键帧并行协方差经外参与杠杆臂得 $P_{\mathrm{cam}}$（平移对角加下界，保留交叉项）——椭圆宽度随该相对不确定度变，不另设 $\tau(Z)$；
2. 对每个有效参考点计算 $\mathbf{P}',\mathbf{f}_{\mathrm{geo}}$；当前图像按整数像素保留最小 $Z'$，仅该点进入检验；
3. 读 $\mathbf{f}_{\mathrm{raft}}$，算 $\mathbf{r}_f$；
4. 由 $(x_n,y_n,\rho)$ 组 $\mathbf{J}_f$，算 $\Sigma_f=\mathbf{J}_f P_{\mathrm{cam}}\mathbf{J}_f^\top$，加脊得 $\tilde\Sigma_f$（近处平移项 $\propto 1/Z$，远处旋转项 $\sim f\sigma_\omega$ 主导）；
5. 算 $s$，执行 §3.6 判定（含协方差驱动的松紧与双流跳过）；
6. 形态学与边界清理 → 输出 $\mathcal{S}$。

---

### 3.8 符号总表（本篇）

| 符号 | 含义 |
|------|------|
| $T_{ck}$（即 $T_{c\leftarrow k}$）、$R,t$ | 名义相对位姿（只生成 $f_{geo}$，椭圆中心） |
| $P^{\mathrm{rel}},P_{\xi}^{\mathrm{rel}},P_{\theta p}$ | 光流门控的并行协方差（关键帧时刻为零，其后与主滤波同步预测与位姿更新）、其位姿 $6\times 6$、旋转–平移交叉 |
| $P_{cam}$（即 $P_{\mathrm{cam}}$）、$A$ | 相机系位姿协方差（椭圆形状）；混合误差到相机微扰 |
| $R_{ic},t_{ic},R_{wc}$ | IMU←相机外参、世界←当前相机旋转 |
| $P,P',u,u'$ | 三维点与投影 |
| $f_{geo},f_{raft},r_f$ | 几何流、神经流、残差 |
| $\delta\mathbf{f}=(\delta u,\delta v)$、$\mathrm{EPE}$ | 位姿微扰引起的光流一阶变化；端点误差 $\|\delta\mathbf{f}\|$ |
| $\theta$、$\delta t_\perp$、$\alpha$ | 旋转误差（rad）、横向平移误差（m）、半视场角（量级） |
| $\sigma_\omega,\sigma_t,\sigma_f$ | 旋转 / 平移 / 像素光流标准差量级；$\sigma_f^{(\omega)}\sim f\sigma_\omega$，$\sigma_f^{(t)}\sim f\sigma_t/Z$ |
| $x_n,y_n,\rho$ | 归一化坐标、逆深度 $1/Z$ |
| $\rho_{\omega t}$ | 位姿交叉相关系数，如 $\mathrm{Corr}(\omega_y,t_y)$；不是逆深度 |
| $J_f$；$j_u,j_v$ | 光流雅可比 $\mathbf{J}_{\pi}\mathbf{J}_{\mathbf{P}}$；其两行 |
| $\mathbf{J}_{\mathbf{P}},\mathbf{J}_{\pi},\Sigma_{\mathbf{P}}$ | 参考点对 $\boldsymbol{\xi}_{\mathrm{cam}}$ 的 $3\times 6$、针孔 $2\times 3$、点位置协方差；不单独存储，见 §3.5 第 3 步 |
| $\Sigma_f,\tilde{\Sigma}_f,\Sigma_{uv},\tau_0$ | 光流协方差、加脊、交叉元、脊参数 |
| $r_{+},r_{-}$ | 沿长轴 / 短轴的探针残差 |
| $s,\mathcal{D},\mathcal{S}$ | 马氏分数、动态像素、静态索引 |

---

### 3.9 与后两篇的接口

- 输出 $\mathcal{S}$ 作为 [Stereo-GICP 论文](SCI_投影StereoGICP.md) / [ESKF·ESIKF](SCI_ESKF与ESIKF.md) 的关键帧静态 target。
- 本篇不推导配准雅可比与卡尔曼更新。

## 4. 实验

实验只判断动态滤除：残差、门控、静态过杀、动态召回 / 掩码 IoU。**机械臂眼在手上**用来控 $T$、$P$ 和 $\Delta t$；公开集在有标签时补残差、过杀与 IoU。

实验只有 **§4.7 的五步** 这一条主顺序，列在 §4.1，每步一行（场景、位姿来源、测什么、通过判据）。链路是双目重建 → 由位姿生成几何光流 → 与 RAFT 比残差 → 用位姿协方差做动态门控。光流量级见 §3.4–§3.6 与 [flow_pose_error_sensitivity.md](flow_pose_error_sensitivity.md)。

### 4.1 一览

只读下表即可按步执行。细则见 §4.7.2，指标见 §4.7.3。失败区单独列表，不与成功区平均。

| 实验 | 做哪些 | 考察什么 | 期望结论 |
|------|--------|----------|----------|
| **步 1** | 相机运动，场景静止。名义位姿 $T^{\mathrm{gt}}$（$T_{c\leftarrow k}^{\mathrm{gt}}$），不注入（$\boldsymbol{\varphi}=\mathbf{0}$，$\mathbf{e}=\mathbf{0}$）。同一遍上，门控用真值位姿协方差 $P_{\mathrm{arm}}$。 | $\mathbf{f}_{\mathrm{geo}}$ 对 RAFT 的 $\|\mathbf{r}_f\|$：立体深度与 RAFT 的地板。静态马氏分数 $s$ 对 $\chi^{2}_{2}$。 | 静态中位 $\|\mathbf{r}_f\|$ 接近 $\tau_0$。$s$ 接近 $\chi^{2}_{2}$（均值约 $2$），$s<1$ 比例与 $1\sigma$ 门限一致；手眼底噪不是动态。此后每一次注入或换成 $T^{\mathrm{imu}}$，都写成相对本组的增量。不过则步 2–5 不宣称可用。 |
| **步 2** | 同序列。图像与 RAFT 不动。先在 $T^{\mathrm{gt}}$ 上一次只注入相机系一轴 $\varphi_x,\varphi_y,\varphi_z,e_x,e_y,e_z$，$\boldsymbol{\varphi}=\mathrm{Log}(R_{\mathrm{gt}}^{\top} R)$，不用欧拉角；六轴后再一组多轴偏置（$\varphi_y$ 与 $e_y$ 同时非零）。扫幅必须盖住步 3 的各轴 IMU 误差。另起：位姿均值仍是真值，注入已知协方差。 | 光流变化 $\delta\mathbf{f}=\mathbf{J}_f\boldsymbol{\xi}$ 相对步 1。主点处 $f\|\varphi_x\|$ 竖直、$f\|\varphi_y\|$ 水平；$\varphi_z$ 在主点 $\sim 0$、边缘 $\sim r_{\mathrm{px}}\|\varphi_z\|$；$f\|e_{x,y}\|/Z$ 为横向流；$e_z$ 主要打在离轴。协方差注入：旋转流是否与深度无关，平移流是否按 $1/Z$，交叉项为正时椭圆是否倾斜。 | 各轴与上列相对误差 $\lesssim 10\%$。横轴是分量，纵轴是相对步 1 的增量。步 3 某轴超出扫幅则先补扫本步，再读步 3，然后才进步 4。协方差注入不代替逐轴光流曲线。 |
| **步 3** | 相机运动，场景静止，与步 1 同一帧对。名义位姿换成 $T^{\mathrm{imu}}$（$T_{c\leftarrow k}^{\mathrm{imu}}$），门控用当前位姿协方差 $P_c^{-}$。整幅被标动态时先缩短 IMU 积分窗；可选在同一 bag 上只平移 IMU 时间戳做失败诊断，然后回到本步。静态像素过不了则不进步 4。不放宽门控。 | 各轴 $(\boldsymbol{\varphi},\mathbf{e})$ 在步 2 曲线上的读数；静态 $\|\mathbf{r}_f\|$ 相对步 1 的膨胀；静态 $s$ 是否整幅出椭圆。时间戳诊断：残差场是否像 $\mathbf{J}_f(\boldsymbol{\omega},\mathbf{v})\Delta t$。 | 静态过杀留在协方差椭圆内，不是整幅 $s\ge 1$。整幅动态则缩短积分窗；主路径不把 $\Delta t$ 写进 $P$。加宽 $P$ 能救静态像素、会漏慢物体。缩短后仍整幅动态则停，不做步 4–5。 |
| **步 4** | 相机运动，场景有动态物体。名义位姿 $T^{\mathrm{gt}}$，不注入。弱动态与强动态分表。 | 背景 $\|\mathbf{r}_f\|$；物体召回与掩码 IoU。 | 背景残差保持在步 1 的地板。本组召回 / 掩码 IoU 是剔除上界。 |
| **步 5** | 与步 4 同一动态序列。名义位姿 $T^{\mathrm{imu}}$，门控只用当前位姿协方差 $P_c^{-}$（与步 3 同一个 $P_{\mathrm{cam}}$，含已缩短的积分窗）。步 3 未过则不做。 | 背景过杀对步 3；召回、漏检与掩码 IoU 相对步 4 的下降。 | 背景误检与步 3 同一量级。召回下降即 IMU 位姿误差的代价。通过与否只看过杀、漏检与 IoU。 |

相机静止且场景有动态为可选上界（§4.7.5），缺这组不阻塞上表。动态序列不再分轴。

掩码对照、工程核对与公开集仍挂在五步上，只报残差、过杀、漏检与 IoU：

| 实验 | 做哪些 | 考察什么 | 期望结论 |
|------|--------|----------|----------|
| **掩码** | 步 3 与步 5 各加一列对照：固定像素阈值、对角 $P$、完整 $P_{\mathrm{cam}}$。召回与 IoU 在步 4–5，静态过杀在步 1 与步 3。不代替逐轴注入。 | 谁能分开物体和相机运动 | 完整 $P$（交叉+杠杆）的 F1 / IoU 最高；对角接近但斜残差更差；固定阈值远近不能两头兼顾。 |
| **工程** | 双流跳过、边界清理；$\sigma_{t,\min}=1$ vs $2\,\mathrm{cm}$（只比近处）。放在步 1 / 步 3 的静态数据上。 | 伪影、近场过杀 | 跳过/清理降低静止误检；地板只绑近处。 |
| **公开集** | EuRoC 弱动态、TUM VI、KITTI、VKITTI 2 | 静态残差与过杀；有语义/光流 GT 时补 IoU、漏检 | VKITTI 报光流/语义 IoU。公开集通常不能只平移 IMU 戳；仍报残差与静态过杀，不把轨迹误差当通过判据。 |

### 4.2 实验符号与参数

| 符号 / 参数 | 含义 | 典型取值或来源 |
|-------------|------|----------------|
| $T_{w\leftarrow f}$ | 法兰相对世界（臂正运动学） | 控制器反馈，与快门对齐 |
| $T_{\mathrm{flange}\leftarrow\mathrm{cam}}$ | 手眼外参：法兰←左目 | 手眼标定 |
| $T_{\mathrm{imu}\leftarrow\mathrm{cam}}$、$(R_{ic},\mathbf{t}_{ic})$ | IMU←左目，含杠杆臂 | `params.yaml` 的 $T_{\mathrm{imu\_cam}}$ |
| $T_{w\leftarrow c}^{\mathrm{arm}}$ | 臂给出的相机真值 | $T_{w\leftarrow f}T_{\mathrm{flange}\leftarrow\mathrm{cam}}$ |
| $T_{c\leftarrow k}$ | 参考左目→当前左目（几何流均值） | 步 1、步 2、步 4 用臂真值；步 3 与步 5 用 IMU 复合 |
| $P_{\mathrm{arm}}$ | 臂 + 手眼的位姿协方差（GT 底噪） | 重复定位与手眼残差，常 $\ll P_c^{-}$ |
| $P_k$ | 参考关键帧在滤波中的位姿协方差 | 不进门控 |
| $P_c^{-}$ / $P_{\xi}^{-}$ | 当前帧 IMU 预测位姿 $6\times 6$ | `imu_pred_state_curr_.cov` 左上块 |
| $P_{k,c}$ | 参考与当前的交叉协方差 | 不进门控 |
| $P_{\mathrm{cam}}$ | 门控用的相机系 $6\times 6$ | $A P_{\xi}^{-}A^{\top}$，平移对角再截断 |
| $A$ | 混合误差→相机微扰 | 含 $R_{ic}$ 与 $[\mathbf{t}_{ic}]_{\times}$，见 §3.5 |
| $\mathrm{Ad}_{T_{c\leftarrow k}}$ | 把 $P_k$ 变到与 $P_c$ 同一相对切空间 | 主路径不用于门控；相对协方差的定义见 §4.4 |
| $P_{c\leftarrow k}$ | 相对位姿协方差（含参考不确定度） | $\approx P_c^{\mathrm{cam}}+\mathrm{Ad}P_k^{\mathrm{cam}}\mathrm{Ad}^{\top}$（不相关时） |
| $\sigma_{t,\min}$ | 平移标准差下界，防 $P$ 收缩后过杀 | 实现默认 $0.02\,\mathrm{m}$ |
| $\tau_0$ | 光流各向同性脊 | $1.5\,\mathrm{px}$ |
| $\tau_{\mathrm{skip}}$ | 双流都小则跳过 | $0.25\,\mathrm{px}^{2}$ |
| $s$ | 马氏分数 $\mathbf{r}_f^{\top}\tilde\Sigma_f^{-1}\mathbf{r}_f$ | $s\ge 1$ 判动态（$1\sigma$ 椭圆外） |
| $\chi^{2}_{2}$ | 二维高斯下 $s$ 的参考分布 | 校准时期望均值 $\approx 2$ |
| $P_{\theta p}$ | 旋转–平移交叉 | 分轴注入时可选一组，验证椭圆倾斜 |
| $\Delta t$ | IMU 时间戳相对快门的固定偏置 | 步 3 整幅被标动态时，可选注入 $0,\pm 5,\pm 10,\pm 20,\pm 40\,\mathrm{ms}$，然后回到步 3 |

步 2 建议事先写死的协方差幅度，便于复现：旋转 $0.1^\circ$ 与 $1^\circ$；平移 $1\,\mathrm{mm}$ 与 $1\,\mathrm{cm}$（或与 $\sigma_{t,\min}$ 同量级的 $2\,\mathrm{cm}$）。一次只点亮一个轴（或一对相关轴），其余为 0。这套幅度是往 $P$ 里写的，与步 2 的 $(\boldsymbol{\varphi},\mathbf{e})$ 网格分开。

### 4.3 机械臂平台（主控实验）

双目 + IMU 固连于法兰（eye-in-hand）。手眼标定得 $T_{\mathrm{flange}\leftarrow\mathrm{cam}}$，与 $T_{\mathrm{imu}\leftarrow\mathrm{cam}}$ 一并写入。法兰位姿 $T_{w\leftarrow f}$ 经

$$
T_{w\leftarrow c}^{\mathrm{arm}}
=
T_{w\leftarrow f}\,T_{\mathrm{flange}\leftarrow\mathrm{cam}}
$$

作为相机系位姿真值。时间戳与快门 / IMU 对齐；重复定位精度与手眼残差记为 $P_{\mathrm{arm}}$（通常远小于开环 IMU 的 $P^-$）。臂真值给的是**名义** $T$，不是 SLAM 的 $P^-$。

场景按「相机怎么动 × 场景里目标怎么动」分三类。强动态与弱动态**分开报告**，不把两类平均后掩盖失败模式。

1. **相机静止或近似静止，场景内目标全部静止**  
   - 相机：法兰锁定、悬停或极短停顿；$\|t\|$、$\|\log R\|$ 接近 0，$\mathbf{f}_{\mathrm{geo}}\approx\mathbf{0}$。  
   - 目标：桌面标定板、墙面、地面等全部静止，无行人、滑台、车辆。  
   - 检验：双流均接近零时跳过规则是否误杀；残差应落在门控椭圆内（手眼底噪不是动态）。  
   - 数据：臂锁定或极慢扫静态桌面；EuRoC / TUM VI 悬停片段。  
   - 用途：双流跳过会不会误杀；不报 IoU。椭圆校准是步 1，相机要在动。

2. **相机运动，场景内目标全部静止**  
   - 相机：相对场景做刚体运动。臂上拉开三档速度（慢移 $\sim 5\,\mathrm{cm/s}$、中速 $\sim 0.3\,\mathrm{m/s}$、快走 / 快转 $\sim 1\,\mathrm{m/s}$ 或 $\sim 30^\circ/\mathrm{s}$），并覆盖纯移、纯转与复合；公开集对应走廊、室内走动、少交通路段。  
   - 目标：仍全部静止，无独立运动。像素位移应全部由相机刚体解释。  
   - 检验：$\mathbf{f}_{\mathrm{geo}}$ 与 $\mathbf{f}_{\mathrm{raft}}$ 应一致；马氏椭圆是否把相机运动误判为动态（过杀）。远近点、旋转主导 / 平移主导分开看。步 3 的慢 / 中 / 快也在这类。  
   - 数据：臂扫静态桌面 + 纹理；EuRoC MH/V 运动序列、TUM VI 室内、KITTI 无交通或少动态路段。  
   - 用途：步 1 的椭圆校准、步 2 的逐轴光流与协方差探针、步 3 的 IMU 位姿过杀。过杀率；全静态数据不报物体 IoU。

3. **场景内有独立运动目标**（主实验为相机运动；相机静止为可选，见 §4.7.5）  
   - 相机：主序列为运动（至少中速平移或短弧）。相机静止、目标在动时，真值位姿下 $\mathbf{f}_{\mathrm{geo}}\approx\mathbf{0}$，残差上界更干净，但缺这组不阻塞 §4.7 五步。  
   - 目标：滑台 / 第二臂 / 规定轨迹行人 / 车辆，运动与相机刚体无关。按强度分两档，**分表**：  
     - **弱动态**：远处、慢速、占像面小（远行人、慢滑台、远车）；残差刚出椭圆，易漏检。  
     - **强动态**：近距、快速、大面积遮挡或同向伴行；残差显著超出椭圆，易牵连静态邻域、过杀边界。  
   - 检验：掩码与物体投影比 Precision / Recall / F1 与 IoU。步 4 用真值位姿，步 5 用 IMU 位姿；通过与否看过杀、漏检与 IoU。  
   - 数据：臂 + 已知物体；KITTI raw 交通、VKITTI 2（光流/深度/语义真值）、含行人室内片段。

对照：固定像素阈值、对角 $P_{\mathrm{cam}}$、现方法（完整 $P_{\mathrm{cam}}$ + 交叉 + 杠杆臂）。比的是残差、过杀、漏检与 IoU。

### 4.4 三层 $P$：定义留在这里

门控椭圆应对**相对运动** $T_{c\leftarrow k}$ 的不确定度。主路径的 $P_{\mathrm{cam}}$ 取 §3.5 的并行协方差：关键帧时刻置零，再经与主滤波相同的 IMU 预测和 GICP 更新，进入 $\mathbf{J}_f$ 的是 $\mathrm{Cov}(T_{c\leftarrow k})$，不是绝对当前 $6\times 6$，也不把 $P_k$ 减进去。机械臂上把三层协方差分开，不要混进同一个 $P$。本节只定义三层。步 1 用真值位姿协方差做门控核对，步 3 与步 5 的门控用这一相对协方差。

| 来源 | 含义 | 用在哪一步 |
|------|------|----------------|
| $P_{\mathrm{arm}}$ | 法兰 / 手眼重复精度（真值位姿协方差） | 步 1：臂位姿当 $T_{c\leftarrow k}$ 的均值，门控用 $P_{\mathrm{arm}}$，看静态 $s$ 是否接近 $\chi^{2}_{2}$ |
| $P_k$ | 参考关键帧在滤波里已经收缩过的协方差 | 不进门控 |
| $P_c^{-}$ | 从锚点 IMU 传播到当前的预测协方差 | 步 3 与步 5 的门控只用它 |

若参考与当前近似不相关，相对位姿在相机系的一阶合成（保守、偏宽）为

$$
P_{c\leftarrow k}
\;\approx\;
P_{c}^{\mathrm{cam}}
+
\mathrm{Ad}_{T_{c\leftarrow k}}\,
P_{k}^{\mathrm{cam}}\,
\mathrm{Ad}_{T_{c\leftarrow k}}^{\top}.
$$

主路径不使用上式：门控用 §3.5 的相对并行协方差，而不是 $P_c^{\mathrm{cam}}+\mathrm{Ad}P_k^{\mathrm{cam}}\mathrm{Ad}^{\top}$。

手眼误差进步 1 的底噪，不要当成动态。步 2 注入的已知 $P$ 不是上面三层里的某一层实测值，也不是 $(\boldsymbol{\varphi},\mathbf{e})$。时间戳平移只作为步 3 整幅被标动态时的失败诊断，做完回到步 3，不要和步 2 的 $P$ 注入混成一类。协方差幅度在 §4.2 事先写明（平移 $1\,\mathrm{mm}$ / $1\,\mathrm{cm}$，旋转 $0.1^\circ/1^\circ$）。

### 4.5 步 2 的两件事：位姿偏置与 $P$ 探针

六轴位姿误差注入是 §4.7 **步 2 的前半**：同一静止场景上，对 $T^{\mathrm{gt}}$ 一次只加 $\varphi_x,\varphi_y,\varphi_z,e_x,e_y,e_z$ 之一，$\boldsymbol{\varphi}=\mathrm{Log}(R_{\mathrm{gt}}^{\top} R)$，不用欧拉角；然后再一组多轴偏置。动态序列（步 4–5）不再分轴。光流增量是否等于 $f\|\varphi_x\|$ 等，以前半为准。

协方差探针是同一步的后半，与前半分开。位姿均值保持真值；注入已知单轴 $P$，再加一个交叉项。核对旋转流与深度无关、平移流按 $1/Z$、交叉项为正时椭圆倾斜。这不是 $(\boldsymbol{\varphi},\mathbf{e})$ 注入，不代替逐轴光流曲线。幅度、4+1 轴表和倾斜椭圆的两条探针路写在 §4.7 步 2。

固定阈值的问题正是「旋转与深度无关、平移 $\propto 1/Z$」。机械臂能单独点亮一根轴。六轴两两的 $P$ 组合会爆炸；交叉项用步 2 的一组 $\varphi_y,e_y$ 偏置，加上一对 $P_{\theta p}$。步 4–5 用完整 $P_{\mathrm{cam}}$ 报召回与 IoU，不做分轴扫描。

### 4.6 时间不同步只作为步 3 的失败诊断

不同步是**均值偏置**，不是步 2 那种往 $P$ 里写相关。它不是另一步：步 3 把整幅标成动态时，先缩短积分窗；可选在同一 bag 上只平移 IMU 时间戳，看残差场是否像 $\mathbf{J}_f(\boldsymbol{\omega},\mathbf{v})\Delta t$，然后回到步 3。公式、$\Delta t$ 网格、速度表，以及「加宽 $P$ 能救静态像素、会漏慢物体；主路径不把 $\Delta t$ 写进 $P$」，都写在 §4.7 步 3。诊断完仍要静态像素留在椭圆内，才进入动态场景。

### 4.7 五步实验：残差地板、分轴位姿误差与 IMU 门控

链路是：双目重建 → 由位姿生成几何光流 $\mathbf{f}_{\mathrm{geo}}$ → 与 RAFT 的 $\mathbf{f}_{\mathrm{raft}}$ 比残差 $\mathbf{r}_f$ → 用 $P_{\mathrm{cam}}$ 经 $\mathbf{J}_f$ 做马氏门控。本节按这个顺序测五步。每步只改一个因素（是否注入位姿误差、名义位姿用 $T^{\mathrm{gt}}$ 还是 $T^{\mathrm{imu}}$、场景是否含独立运动）。步 1 在同一遍上用真值位姿协方差核对 $s$；步 2 在逐轴注入之后另注入已知协方差看椭圆；步 3 若整幅被标动态，先缩短积分窗，可选做时间戳诊断再回到本步。光流增量相对步 1；步 5 的背景过杀对步 3，召回下降对步 4。通过与否看残差、过杀、漏检与 IoU。不同条件不平均进一张表。

相机静止且场景有动态物体是可选上界，缺这组不阻塞下面五步（§4.7.5）。实现钩子：`src/dynamic_cloud_processor.cpp`，`sio_geo_flow::{ComputeFlowJac,MakePoseCovCam,AccumulateSigmaFFromPoseCov,MahalanobisSq}`，`ImuStatePoseCovInCamera`。一阶光流式见 §3.4 与 [flow_pose_error_sensitivity.md](flow_pose_error_sensitivity.md)。

#### 4.7.1 符号（五步勿混）

旋转误差用相机系旋转向量，不用欧拉角，也不收成单一转角。设几何流所用旋转为 $R$、真值为 $R_{\mathrm{gt}}$（世界系左目，或同一约定下的相对旋转），则

$$
\boldsymbol{\varphi}
=
\mathrm{Log}(R_{\mathrm{gt}}^{\top} R)
=
\begin{pmatrix}\varphi_x\\ \varphi_y\\ \varphi_z\end{pmatrix}.
$$

$\boldsymbol{\varphi}$ 在真值相机系。平移误差取同一相机系的 $\mathbf{e}=(e_x,e_y,e_z)^{\top}$（小量下 $\mathbf{e}\approx R_{\mathrm{gt}}^{\top}(\mathbf{t}-\mathbf{t}_{\mathrm{gt}})$）。列序 $(\varphi_x,\varphi_y,\varphi_z,e_x,e_y,e_z)$ 与 $\mathbf{J}_f$ 的 $(\omega_x,\omega_y,\omega_z,t_x,t_y,t_z)$ 对齐，故 $\delta\mathbf{f}\approx\mathbf{J}_f\boldsymbol{\xi}$，$\boldsymbol{\xi}=(\boldsymbol{\varphi},\mathbf{e})$。表和曲线报这六个分量；$\|\boldsymbol{\varphi}\|$ 只能作辅列。

| 符号 | 含义 | 来源 | 用于 |
|------|------|------|------|
| $T_{w\leftarrow c}^{\mathrm{gt}}$ | 相机世界系位姿真值 | **臂**：法兰×手眼；**公开集**：EuRoC / TUM VI / KITTI 轨迹 GT（与左目外参对齐） | 步 1、2、4 的名义位姿；步 3、5 用来读位姿误差 |
| $T_{c\leftarrow k}^{\mathrm{gt}}$ | 参考→当前相对位姿真值 | $(T_{w\leftarrow c}^{\mathrm{gt}})^{-1}T_{w\leftarrow k}^{\mathrm{gt}}$ | 步 1、2、4 生成 $\mathbf{f}_{\mathrm{geo}}^{\mathrm{gt}}$；步 3 用来读 IMU 相对真值的误差 |
| $T_{c\leftarrow k}^{\mathrm{imu}}$ | IMU 估计的相对位姿 | 与线上一致的 IMU 预测复合，只用于投影 | 步 3 换名义位姿之后，以及步 5 |
| $\boldsymbol{\varphi},\,\mathbf{e}$ | 相机系旋转向量与平移误差 | 注入时 $R$ 为扰动后旋转；步 3 与步 5 取 IMU 的旋转 / 平移 | 步 2 逐轴注入的横轴；步 3、5 在步 2 曲线上的读数。不是步 2 的协方差注入 |
| $P_{\mathrm{cam}}$ | 门控用相机系 $6\times 6$ | 步 1 用 $P_{\mathrm{arm}}$；步 2 注入已知单轴 $P$；步 4 沿用步 1 的真值门控；步 3 与步 5 用线上 $P_c^{-}$（含 $\sigma_{t,\min}$） | $s=\mathbf{r}_f^{\top}\tilde\Sigma_f^{-1}\mathbf{r}_f$ |
| $\mathcal{L}_{\mathrm{dyn}}/\mathcal{L}_{\mathrm{sta}}$ | 动态 / 静态标签 | **臂**：已知物体投影，全静态则全是 $\mathcal{L}_{\mathrm{sta}}$；**VKITTI 2**：语义或运动分割；KITTI / 室内行人：粗标注或框投影 | 步 4、5 的召回与 IoU；步 1、3 的过杀 |
| oracle | 用 $T^{\mathrm{gt}}$ 的残差再套步 1 门控得到的伪标签 | 无语义 GT 时的代理 | 步 5 相对步 4 的一致率；不报绝对 IoU |

**短时窗口。** 默认参考关键帧到当前帧的 IMU 积分间隔 $\Delta t_{\mathrm{kf}}\in\{50,100,200\}\,\mathrm{ms}$。步 3 若整幅被标动态，先缩短这个窗；可选先查时间戳，查完回到步 3。不许先放宽门控。运动速度与 §4.3 对齐：$0.3\,\mathrm{m/s}$、$1\,\mathrm{m/s}$，以及 $3\,\mathrm{m/s}$、$60^\circ/\mathrm{s}$（平台达不到则臂用可达上界，公开集用高速片段，表注「未达」）。强动态与弱动态分表。失败格子不与成功区平均。

#### 4.7.2 五步流程

光流与残差的增量都相对**同像素、同帧对**的步 1。深度三桶：近 $<2\,\mathrm{m}$、中 $2$–$5\,\mathrm{m}$、远 $>5\,\mathrm{m}$。主表顺序就是步 1→2→3→4→5。

**步 1 — 相机运动，场景静止，真值位姿，不注入误差**

- **设定**：场景无独立运动。$T_{c\leftarrow k}\leftarrow T_{c\leftarrow k}^{\mathrm{gt}}$，$\boldsymbol{\varphi}=\mathbf{0}$，$\mathbf{e}=\mathbf{0}$。深度用双目视差（与线上一致）。$\mathbf{f}_{\mathrm{geo}}$ 与 RAFT 的残差就是立体深度和 RAFT 的地板。
- **数据**：臂扫静态桌面（慢 / 中 / 快；纯移、纯转、复合）。公开集用带轨迹 GT 的静止场景片段（EuRoC、TUM VI、少动态 KITTI）。
- **测**：$\mathbf{f}_{\mathrm{geo}}^{\mathrm{gt}}$ 对 $\mathbf{f}_{\mathrm{raft}}$；静态 $\|\mathbf{r}_f\|$ 的直方图和分位数；按深度桶的 EPE。同一遍上，门控用真值位姿协方差 $P_{\mathrm{arm}}$（手眼 / 重复定位底噪），看静态 $s$ 的直方图。
- **通过**：静态中位 $\|\mathbf{r}_f\|$ 接近神经流 / 量化底噪（量级 $\sim\tau_0$）。$s$ 接近 $\chi^{2}_{2}$（均值约 $2$），$s<1$ 的比例与 $1\sigma$ 门限一致；手眼底噪不是动态。地板不过，或 $s$ 对不上 $\chi^{2}_{2}$，则步 2–5 不宣称方法可用——先查视差、RAFT、同步，而不是把门控画宽，后面各步也不把椭圆当成已经校准。此后每一次注入或换成 $T^{\mathrm{imu}}$，都写成相对本组的增量。

**步 2 — 同序列，在真值上逐轴注入位姿误差**

- **设定**：图像与 RAFT 不动，只改算 $\mathbf{f}_{\mathrm{geo}}$ 的位姿。一次只非零一轴，其余为 0，顺序为 $\varphi_x,\varphi_y,\varphi_z,e_x,e_y,e_z$。扫幅必须盖住步 3 实测的各轴 IMU 误差。允许用短试点估计量级来铺网格，试点不代替步 3 的正式表。出步 3 之后若某一轴超出扫幅网格，先补扫步 2，再读步 3，然后才进入步 4。起始网格只是占位，步 3 出数后按各轴实测外延：$|\varphi_i|\in\{0.0017,0.0087,0.0175\}\,\mathrm{rad}$（约 $0.1^\circ,0.5^\circ,1^\circ$，是该分量的模，不是欧拉角），$|e_i|\in\{1,2,5\}\,\mathrm{cm}$。
- **测**：$\delta\mathbf{f}_{\mathrm{geo}}=\mathbf{f}_{\mathrm{geo}}(\boldsymbol{\xi})-\mathbf{f}_{\mathrm{geo}}^{\mathrm{gt}}$，以及残差增量 $\mathbf{r}_f(\boldsymbol{\xi})-\mathbf{r}_f^{(1)}$（RAFT 不变时应约等于 $-\delta\mathbf{f}_{\mathrm{geo}}$）。主点邻域与像面边缘分开，平移轴再按深度桶分开。测到的光流变化对照：

| 只注入 | 主点附近 | 离轴 / 边缘 |
|--------|----------|-------------|
| $\varphi_x$ | 竖直，幅值 $\sim f\|\varphi_x\|$ | 同量级，另有 §3.4 的 $x_n y_n$ 修正 |
| $\varphi_y$ | 水平，幅值 $\sim f\|\varphi_y\|$ | 同上 |
| $\varphi_z$ | $\sim 0$ | $\sim r_{\mathrm{px}}\|\varphi_z\|$；$r_{\mathrm{px}}$ 为到主点的像素距离 |
| $e_x$ | 水平 $\sim f\|e_x\|/Z$ | 同式，随深度桶 |
| $e_y$ | 竖直 $\sim f\|e_y\|/Z$ | 同式 |
| $e_z$ | $\sim 0$ | 主要打在离轴像素，$\sim f\|x_n\|\,\|e_z\|/Z$ |

$r_{\mathrm{px}}\sim f\alpha$（$\alpha$ 为该像素的半视场角），与 §3.4 及 [flow_pose_error_sensitivity.md](flow_pose_error_sensitivity.md) 的边缘滚转项是同一式。$\varphi_x,\varphi_y$ 几乎不随 $Z$；$e_x,e_y$ 按 $1/Z$。符号与 $\mathbf{J}_f$ 各列一致。

- **多轴偏置（只一组）**：六轴单轴之后，再注入一组 $\varphi_y$ 与 $e_y$ 同时非零、其余为 0，幅度取步 3 这两轴的同一量级；反号 $e_y$ 只作短轴对照，不算第二套扫描。$\delta\mathbf{f}_{\mathrm{geo}}$ 仍应等于 $\mathbf{J}_f\boldsymbol{\xi}$。同号组合沿长轴，$s$ 更小；反号沿短轴，$s$ 更大（§3.5 的 $\mathbf{r}_+,\mathbf{r}_-$）。不做六轴全因子。扫幅同样盖住步 3 的 IMU 误差。
- **通过**：各轴测到的光流变化与上表相对误差 $\lesssim 10\%$。横轴是分量，不是 $\|\boldsymbol{\varphi}\|$。纵轴是相对步 1 的增量，不把地板再加进去。

- **协方差探针（位姿均值仍为真值，与上面的 $(\boldsymbol{\varphi},\mathbf{e})$ 分开）**：位姿均值保持臂真值 $T_{c\leftarrow k}^{\mathrm{gt}}$，不把探针写成名义运动。软件注入已知单轴 $P_{\mathrm{cam}}$，再加一个交叉项。这不是 $(\boldsymbol{\varphi},\mathbf{e})$ 位姿注入，不代替上面的逐轴光流曲线。不必做六轴全因子的协方差，也不对每条动态序列分轴。

幅度用 §4.2：旋转 $0.1^\circ$ 与 $1^\circ$；平移 $1\,\mathrm{mm}$ 与 $1\,\mathrm{cm}$（或与 $\sigma_{t,\min}$ 同量级的 $2\,\mathrm{cm}$）。一次只注入下表一列（其余轴方差为 0）。下表不是上面的六轴位姿偏置：

| 注入轴 | 光心附近预期 | 用来证明 |
|--------|----------------|----------|
| $\omega_y$（俯仰） | 残差主要在 $u$，幅值与 $Z$ 关系弱 | 旋转项不含 $\rho$ |
| $\omega_z$（偏航） | 光心几乎无流；像面边缘有 $u$；有 $\mathbf{t}_{ic}$ 时与 $t_y$ 相关 | 杠杆臂，不是立体基线 |
| $t_x$ | 残差主要在 $u$，近处更大 | 平移 $\propto 1/Z$ |
| $t_z$ | 近处放大更狠 | 前向平移 |
| 一个交叉项：$\mathrm{Corr}(\omega_y,t_y)=0.9$ | $\Sigma_{uv}\neq 0$，同号 $45^\circ$ 的 $s$ 更小 | 交叉项为正时椭圆倾斜 |

看近/远点的 $s$ 比、主轴沿 $u$ 还是 $v$、覆盖率。通过：旋转流与深度无关，平移流按 $1/Z$；交叉项为正时椭圆倾斜，$\rho>0$ 则 $s(\mathbf{r}_+)<s(\mathbf{r}_-)$。

名义运动只决定几何流的均值（椭圆中心）；$P_{\mathrm{cam}}$ 决定形状和倾斜。相关项写进 $P$，不要让机械臂同时做俯仰+侧移来制造相关。场景不动。臂做一小段已知、可重复的名义运动（慢速 $t_x$ 或短弧均可），用臂真值当 $T_{c\leftarrow k}$ 算几何流。同一段数据上软件覆盖不同的 $P_{\mathrm{cam}}$，四根单轴和下面的交叉项共用。

交叉项其余元素为 0，只点亮

$$
P(\omega_y,\omega_y)=\sigma_\omega^2,\quad
P(t_y,t_y)=\sigma_t^2,\quad
P(\omega_y,t_y)=P(t_y,\omega_y)=\rho\,\sigma_\omega\sigma_t,
$$

$\rho=0.9$。光心附近 $\partial u/\partial\omega_y\approx f_x$、$\partial v/\partial t_y\approx f_y\rho_{\mathrm{inv}Z}$，故 $\mathrm{Cov}(u,v)\neq 0$，椭圆相对像素轴倾斜。$\sigma_t$ **必须不低于** $\sigma_{t,\min}=0.02\,\mathrm{m}$（建议直接用 $2\,\mathrm{cm}$），$\rho$ 按截断后的 $\sigma_t$ 算；若按 $1\,\mathrm{mm}$ 写交叉、再被地板抬到 $2\,\mathrm{cm}$，有效 $|\rho|$ 会被冲到 $\approx 0.045$，看起来几乎不斜。

全静态残差近似各向同性，只看 $s$ 直方图分不出转过的椭圆。要主动探针，取等欧氏长度（例如 $4\,\mathrm{px}$）：

| 探针残差 | 预期（$\rho>0$） |
|----------|------------------|
| $\mathbf{r}_+=(2.8,2.8)$ | 沿长轴，$s$ 更小 |
| $\mathbf{r}_-=(2.8,-2.8)$ | 沿短轴，$s$ 更大 |
| 对照：丢掉 $P_{\theta p}$ 的对角 $P$ | $s(\mathbf{r}_+)=s(\mathbf{r}_-)$ |

两条路即可，不必再让臂走出 $45^\circ$：

1. **合成探针（建议先做）**：冻结一帧，在光心附近算出 $\tilde\Sigma_f$，直接代入 $\mathbf{r}_\pm$ 比 $s$。与单测同一逻辑。
2. **位姿偏置（要真实像素时）**：图像与臂真值不动；只在算几何流时加已知 $\delta\boldsymbol{\xi}$，残差 $\approx -\mathbf{J}_f\delta\boldsymbol{\xi}$。光心附近取 $f_x\omega_y \approx f_y\rho_{\mathrm{inv}Z}\,t_y$ 得 $\mathbf{r}_+$，反号 $t_y$ 得 $\mathbf{r}_-$。这段 $\delta\boldsymbol{\xi}$ 只为把残差送到长轴或短轴，不代替本步的六轴光流曲线，也不写进名义 $T$ 的对外叙述。

单轴四列同理：注入的是该轴方差，不是「只许臂走那一轴」。臂的名义运动可以共用；若还要看「残差主要在 $u$」，用合成 $\mathbf{r}$ 或对该轴加偏置，不要把 $P$ 的注入和法兰指令绑死。

**步 3 — 相机运动，场景静止，IMU 位姿**

- **设定**：与步 1 同一帧对。名义位姿换成 $T_{c\leftarrow k}^{\mathrm{imu}}$。门控用当前位姿协方差，即线上 $P_{\mathrm{cam}}=P_c^{-}$（经 $A$ 到相机系，含交叉项与 $\sigma_{t,\min}$），不把 $P_{\mathrm{arm}}$ 写进门控。
- **测**：相对 $T^{\mathrm{gt}}$ 的 $\varphi_x,\varphi_y,\varphi_z,e_x,e_y,e_z$（定义同 §4.7.1，旋转误差是旋转向量，不是欧拉角），以及各分量在步 2 曲线上对到哪一条、扫幅网格的哪一格。静态 $\|\mathbf{r}_f\|$ 相对步 1 的膨胀；$\|\mathbf{f}_{\mathrm{geo}}^{\mathrm{imu}}-\mathbf{f}_{\mathrm{geo}}^{\mathrm{gt}}\|$ 对 $\|\mathbf{J}_f\boldsymbol{\xi}\|$。全静态 $s$ 直方图。形态学与边界清理之后的静态过杀率（FPR）。
- **通过**：静态残差留在协方差椭圆内，而不是整幅 $s\ge 1$。清理后连通域过杀率期望 $\lesssim 5\%$；散落的 $s\ge 1$ 尾部是 $1\sigma$ 门限本来就会有的，不靠放大椭圆去掉。禁止为压下过杀而抬固定阈值、抬 $\tau_0$、抬 $\sigma_{t,\min}$ 或人为放大 $P_{\mathrm{cam}}$。静态像素留不住，则不做步 4–5。
- **失败诊断（不是另一步）**：若换上 $T^{\mathrm{imu}}$ 之后整幅被标成动态，先缩短积分窗。可选在同一 bag 上只平移时间戳，查完回到本步。偏置未收敛、窗太长也在这里查。这些都不靠把椭圆画大来过关。缩短窗后仍整幅动态，则停在步 3，不做步 4–5。

不同步是均值偏置，不是步 2 那种往 $P$ 里写相关。图像在快门时刻 $t$，IMU 若按 $t+\Delta t$ 积分，几何流用的 $T_{c\leftarrow k}$ 多（或少）走了一小段刚体，残差场近似

$$
\mathbf{r}_f
\;\approx\;
-\mathbf{J}_f
\begin{bmatrix}\boldsymbol{\omega}\Delta t\\ \mathbf{v}\Delta t\end{bmatrix},
$$

全图同一套 $(\boldsymbol{\omega},\mathbf{v})$，近处平移项按 $1/Z$ 放大。独立运动物体是局部多出来的残差，不要和这张全局场当成一类。

录一次已对齐的 bag，回放时只平移 IMU 时间戳（或只平移图像戳），$\Delta t\in\{0,\pm 5,\pm 10,\pm 20,\pm 40\}\,\mathrm{ms}$。不要拔线、不要重录。臂真值仍对齐快门，用来核对残差是否像 $\mathbf{J}_f(\boldsymbol{\omega},\mathbf{v})\Delta t$。诊断之后回到步 3，仍以静态像素是否留在椭圆内为准。

速度必须拉开，否则小数时延看不出来（$\|\mathbf{v}\|\Delta t$ 相对 $\sigma_{t,\min}$）：

| 名义运动 | $20\,\mathrm{ms}$ 等效位姿 | 预期 |
|----------|------------------------------|------|
| 慢移 $\sim 5\,\mathrm{cm/s}$ | $\sim 1\,\mathrm{mm}$ | 多被地板/脊吃掉，$s$ 几乎不动 |
| 中速 $\sim 0.3\,\mathrm{m/s}$ | $\sim 6\,\mathrm{mm}$ | 近处静态过杀开始升 |
| 快走 / 快转 $\sim 1\,\mathrm{m/s}$ 或 $\sim 30^\circ/\mathrm{s}$ | $\sim 2\,\mathrm{cm}$ 或 $\sim 0.6^\circ$ | 过杀明显；旋转项远近都有 |

对照在全静态 bag 上做三列：不补偿（主路径）；把 $\mathrm{diag}((\mathbf{v}\Delta t)^2,(\boldsymbol{\omega}\Delta t)^2)$ 加进 $P_{\mathrm{cam}}$（只加宽椭圆）；时间戳改回 0（上界）。主路径不把 $\Delta t$ 写进 $P$。

看：静态 $s$ 均值与过杀率；残差场与 $\mathbf{J}_f(\boldsymbol{\omega},\mathbf{v})\Delta t$ 的相关系数（应高）。$|\Delta t|$ 与 $\|\mathbf{v}\|$ 越大，近处过杀越狠。加宽 $P$ 能救静态像素，同时会漏掉慢物体。残差场对上式之后，缩短 $\Delta t_{\mathrm{kf}}$ 并回到换 $T^{\mathrm{imu}}$ 的这一段。公开集只报「官方同步 vs 人为 $\pm 20\,\mathrm{ms}$」两种设定。静态像素留不住，不进入动态场景。

**步 4 — 相机运动，场景有动态物体，真值位姿**

- **设定**：$T_{c\leftarrow k}\leftarrow T_{c\leftarrow k}^{\mathrm{gt}}$，不注入。物体运动与相机刚体无关。弱动态（远、慢、占像面小）与强动态（近、快、大面积）分表。
- **测**：背景（$\mathcal{L}_{\mathrm{sta}}$）上的 $\|\mathbf{r}_f\|$ 是否仍在步 1 的地板；物体上的 Precision / Recall / F1 与掩码 IoU。无稠密标签时，用 $T^{\mathrm{gt}}$ 残差上的 oracle，只报与 oracle 的一致率。
- **通过**：背景残差保持在步 1（中位数与深度桶不因物体出现而整体抬高）。本组召回与掩码 IoU 是剔除的上界；步 5 只与这组比。背景若已偏离步 1，先查视差、RAFT 与标签，不把缺口算到 IMU 上。

**步 5 — 相机运动，场景有动态物体，IMU 位姿**

- **设定**：与步 4 同一序列。名义位姿换成 $T_{c\leftarrow k}^{\mathrm{imu}}$。门控只用当前位姿协方差 $P_c^{-}$，与步 3 同一个 $P_{\mathrm{cam}}$（含步 3 若缩短过的积分窗）。不把 $P_{\mathrm{arm}}$ 写进 $P_{\mathrm{cam}}$。臂给独立运动物体做投影标签，并用来读位姿误差。步 3 的椭圆检查未过，则不做本步。
- **测**：背景过杀率对步 3；物体召回、漏检、F1、掩码 IoU 相对步 4 的下降。下降量就是 IMU 位姿误差的代价，用步 3 各轴 $(\boldsymbol{\varphi},\mathbf{e})$ 在步 2 曲线上的读数解释（旋转误差是旋转向量）。
- **通过**：背景误检与步 3 同一量级。召回相对步 4 的下降单独列出。通过与否只看过杀（FP）、漏检（FN）与掩码 IoU。

#### 4.7.3 指标与判据

| 理论量（§3） | 哪一步测 | 期望 |
|--------------|----------|------|
| 立体 + RAFT 地板 | 步 1 的 $\|\mathbf{r}_f\|$ | 静态中位 $\sim\tau_0$；其后都报相对本步的增量 |
| $\delta\mathbf{f}=\mathbf{J}_f\boldsymbol{\xi}$ | 步 2 的 $\delta\mathbf{f}_{\mathrm{geo}}$ | 分轴符合 §4.7.2 的表，相对误差 $\lesssim 10\%$ |
| $\sigma_f^{(\omega)}\sim f\sigma_\omega$，$\sigma_f^{(t)}\sim f\sigma_t/Z$ | 步 2 逐轴注入的深度桶；步 2 的单轴 $P$；步 3 换 IMU 后的半轴 | 旋转流不随 $Z$；平移流近大远小 |
| $s$ 对 $\chi^{2}_{2}$ | 步 1，门控用真值位姿协方差 $P_{\mathrm{arm}}$ | 均值约 $2$；$s<1$ 比例与 $1\sigma$ 门限一致；手眼底噪不是动态 |
| 换 $T^{\mathrm{imu}}$ 后的椭圆 | 步 3 | 静态残差留在椭圆内；整幅 $s\ge 1$ 则缩短积分窗，可选做时间戳诊断后再回到本步 |
| 交叉项（位姿偏置） | 步 2 的一组 $\varphi_y,e_y$ | 同号 $s$ 小于反号（§3.5） |
| 交叉项（协方差探针） | 步 2，位姿均值仍为真值 | $\rho>0$ 时椭圆倾斜，$s(\mathbf{r}_+)<s(\mathbf{r}_-)$ |
| 时延 | 步 3 的可选失败诊断 | 残差场像 $\mathbf{J}_f(\boldsymbol{\omega},\mathbf{v})\Delta t$；诊断后回到步 3；主路径不把 $\Delta t$ 写入 $P$；加宽 $P$ 会漏慢物体 |
| 召回 / 掩码 IoU | 步 4 为上界，步 5 为 IMU 代价 | 步 4 背景保持步 1；步 5 背景过杀对齐步 3；通过与否看 FP、FN 与 IoU |

固定 $\tau$、对角 $P$、完整 $P_{\mathrm{cam}}$ 的对照可以在步 3 与步 5 各加一列（§4.1 掩码行），不改变五步顺序，也不代替逐轴注入。双流跳过、边界清理、$\sigma_{t,\min}=1$ vs $2\,\mathrm{cm}$（只比近处）放在步 1 / 步 3 的静态数据上；近处地板的解释见 §3.5。

**不可用（单独列表，不平均进主表）：**

- 步 1 地板已远大于 $\tau_0$（视差失效、运动模糊、RAFT 大面积失败）。
- 步 2 某一轴与 $f\|\varphi\|$、$r_{\mathrm{px}}\|\varphi_z\|$、$f\|e\|/Z$ 对不上，却用步 3 解释 IMU。
- 步 3 整幅动态：缩短积分窗之后仍整幅动态（做了时间戳诊断时，残差场也已对上 $\mathbf{J}_f(\boldsymbol{\omega},\mathbf{v})\Delta t$）。停止，不做步 4–5，也不放宽门控。
- 长窗口开环（$\Delta t_{\mathrm{kf}}\gtrsim 400\,\mathrm{ms}$）或高速（$3\,\mathrm{m/s}$、$60^\circ/\mathrm{s}$）下 $\boldsymbol{\xi}$ 映到的光流已远大于椭圆半轴。
- 去掉 $\sigma_{t,\min}$ 且 $P^-$ 已收到毫米：近处静态过杀。这是近场核对，不是步 3 失败后的补救。
- 全球 $P$ 漂大：门控偏宽，弱动态漏检。
- 无动态标签且无臂上物体 GT：步 4–5 不报绝对 IoU。
- 步 3 的时间戳诊断上 $|\Delta t|\cdot\|\mathbf{v}\|$ 大：全局残差场，不是局部动态。主路径不把 $\Delta t$ 写进 $P$。

#### 4.7.4 数据集

不预写测得的误差或 IoU。下表只说明序列和真值从哪来。

| 用途 | 序列 / 采集 | 具备的真值 |
|------|-------------|------------|
| 步 1–3 | 臂：静态桌面慢扫、中速、快转 | $T^{\mathrm{gt}}$，$P_{\mathrm{arm}}$ |
| 步 2 逐轴注入 | 同上；软件改 $T$，图像与 RAFT 不动 | 注入的 $(\boldsymbol{\varphi},\mathbf{e})$ 已知 |
| 步 2 协方差探针 | 同上静态序列；软件改 $P_{\mathrm{cam}}$，位姿均值仍为 $T^{\mathrm{gt}}$ | 注入的单轴方差与一个交叉项已知 |
| 步 3 时间戳诊断 | 同一已对齐 bag，只平移 IMU 戳；可选，做完回到步 3 | 臂真值仍对齐快门 |
| 步 3 的公开位姿 | EuRoC MH_01 / MH_05、V1_03；TUM VI 室内 | 轨迹 GT。通常无稠密动态标签，只报各轴误差与静态过杀 |
| 步 4–5 | 臂 + 滑台 / 第二臂 / 已知轨迹物体；VKITTI 2；KITTI raw 交通 | 臂侧投影掩码；VKITTI 语义 / 光流 GT；KITTI 粗动态 |
| 高速 | 臂可达上界；KITTI 高速路段 | 注明是否达到 $3\,\mathrm{m/s}$ / $60^\circ/\mathrm{s}$ |

#### 4.7.5 可选：相机静止且场景有动态

真值位姿下相机不动，则 $\mathbf{f}_{\mathrm{geo}}\approx\mathbf{0}$，残差几乎就是 RAFT，动态召回上界比步 4 更干净；有数据时当作更紧的上界记一笔，没有这组也不阻塞步 1–5。

#### 4.7.6 交付物

1. 步 1：静态残差直方图、按深度的 EPE。真值位姿协方差 $P_{\mathrm{arm}}$ 下 $s$ 对 $\chi^{2}_{2}$。后文增量都相对这组。
2. 步 2：六轴 $\delta\mathbf{f}_{\mathrm{geo}}$ 对 $f\|\varphi_x\|$、$f\|\varphi_y\|$、$r_{\mathrm{px}}\|\varphi_z\|$、$f\|e_{x,y}\|/Z$ 以及 $e_z$ 的离轴项；横轴为分量。另有一组 $\varphi_y,e_y$ 的 $s(\mathbf{r}_+)$ 对 $s(\mathbf{r}_-)$。曲线与 [flow_pose_error_sensitivity.md](flow_pose_error_sensitivity.md) 的一阶式对齐。协方差探针：单轴 $P$ 的远近与一对交叉项下 $s(\mathbf{r}_+)$ 对 $s(\mathbf{r}_-)$。
3. 步 3：各轴 IMU 误差标在步 2 曲线上；换 $T^{\mathrm{imu}}$ 后的静态 $s$ 与清理后 FPR。若做了时间戳诊断，记下平移的 IMU 戳、残差场与 $\mathbf{J}_f(\boldsymbol{\omega},\mathbf{v})\Delta t$ 的对照，以及缩短前后的 $\Delta t_{\mathrm{kf}}$，并写明主路径没有把 $\Delta t$ 加进 $P$、门控参数没有为过关而改。
4. 步 4：背景残差对步 1；召回 / 掩码 IoU（上界）。弱 / 强分表。
5. 步 5：背景 FPR 对步 3；召回与漏检相对步 4 的下降；掩码 IoU。通过与否只看 FP、FN 与 IoU。
6. 缺口：缺臂上动态标签、缺 VKITTI、未达高速，写入补充材料。步 3 整幅动态却未做时间戳诊断，也写入。公开集上的静态过杀不要写成掩码 SOTA。

## 5. 结论

1. 用关键帧相对位姿协方差诱导的光流椭圆做马氏门控，比固定像素阈值更符合双目-惯性几何；椭圆由 $\Sigma_f=\mathbf{J}_f P_{\mathrm{cam}}\mathbf{J}_f^\top$ 给出，可相对像素轴倾斜。$P_{\mathrm{cam}}$ 来自关键帧处置零后再经同一套预测与 GICP 更新的并行协方差，即 $\mathrm{Cov}(T_{c\leftarrow k})$。一阶量级上旋转映为 $\sigma_f\sim f\sigma_\omega$（与 $Z$ 无关）、平移映为 $\sigma_f\sim f\sigma_t/Z$，故门限保持 $s\ge 1$，自适应体现在半轴随 $P_{\mathrm{cam}}$ 与深度变化。
2. Z-buffer 几何光流与神经光流残差提供了不依赖语义类别的动态线索；双流跳过与边界清理是降低漏检/伪影的必要工程补充；交叉项使「像位姿误差」的静态斜残差更松、「非刚体」方向更紧。
3. 动态滤除是否成立，看残差、门控、静态过杀与动态召回 / 掩码 IoU。通过与否不看轨迹误差。
4. 局限：依赖视差与 IMU 先验质量；纯旋转小运动或 RAFT 失效时仍可能误判。$P_{\mathrm{cam}}$ 不含关键帧之前的绝对不确定度（并行协方差在关键帧处置零，不减 $P_k$）；关键帧之后随预测增长，并被中间的 GICP 更新压回，不是开环 IMU 预积分。可用性只按 §4.7 五步验证。步 1：相机运动、场景静止、$T^{\mathrm{gt}}$、不注入位姿误差，几何流相对 RAFT 的残差是深度与 RAFT 的地板；同一遍上门控用真值位姿协方差，静态 $s$ 应接近 $\chi^{2}_{2}$（均值约 $2$），手眼底噪不是动态。步 2：同一静止序列上先逐轴注入相机系旋转向量 $\boldsymbol{\varphi}=\mathrm{Log}(R_{\mathrm{gt}}^{\top} R)$ 与平移 $\mathbf{e}$（分轴报告，不用欧拉角），再加一组多轴偏置；光流变化应对上 $f\|\varphi_x\|$、$f\|\varphi_y\|$、边缘 $r_{\mathrm{px}}\|\varphi_z\|$、$f\|e_{x,y}\|/Z$。位姿均值仍为真值时另注入已知协方差，核对椭圆随深度与交叉项变化。扫幅盖住步 3 的 IMU 误差。步 3：换成 $T^{\mathrm{imu}}$，门控用当前位姿协方差，各轴误差在步 2 曲线上读取；静态过杀须留在椭圆内。整幅被标动态则缩短 IMU 积分窗，可选在同一 bag 上平移时间戳做失败诊断后再回到本步；不放宽门控，主路径不把 $\Delta t$ 写进 $P$。静态像素留不住则不进动态场景。步 4：动态场景上 $T^{\mathrm{gt}}$，背景残差保持步 1 的地板，召回 / 掩码 IoU 为上界。步 5：同一动态序列、名义位姿 $T^{\mathrm{imu}}$、门控只用当前位姿协方差，背景过杀对齐步 3，召回相对步 4 的下降是 IMU 位姿误差的代价；通过与否看过杀、漏检与 IoU。相机静止且场景有动态只记一句话，缺数据不阻塞这五步。长窗口、高速或同步偏差标为不可用区，不与成功区平均。
5. 后续：与语义融合、时序跟踪动态物体、把静态置信度传给鲁棒核权重。主路径门控用 §3.5 的关键帧相对协方差，不把绝对 $P_k$ 减进 $P_{\mathrm{cam}}$。执行 §4.7 步 4–5 时需要动态标签（臂上物体投影或 VKITTI 类光流 / 语义 GT）；没有标签时掩码只报相对 oracle，不报绝对 IoU。
