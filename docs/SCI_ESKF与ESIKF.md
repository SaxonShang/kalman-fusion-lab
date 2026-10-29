# 松耦合 ESKF 与紧耦合几何-光度 ESIKF

> 五篇 SCI 之 **Paper 3** · [索引](SCI_五篇论文索引.md) · 工程：`StateEstimator::ESKF` / `ESIKF`，`utils.h` 协方差对齐

## 建议题目

Loosely and Tightly Coupled Error-State Filtering with Stereo-GICP Geometry and Photometric Virtual Observations

## 1. Introduction

### 1.1 研究背景

双目-惯性前端需要在统一的概率状态里融合两类信息：IMU 提供的高频短时运动，以及视觉点云配准提供的相对几何。实际系统很少对每一帧都跑同等代价的紧耦合：非关键帧更适合“整姿观测 + 一次更新”的松耦合 ESKF，以维持实时；关键帧则适合把逐点几何（及光度）信息吃透的紧耦合 ESIKF，以抬高精度并重建 target 缓存。若两套更新使用不同的误差定义或协方差坐标，创新会偏、NEES 会爆，模块也无法干净拼接。

因此本文的核心不是再发明一种配准，而是把**已经得到的几何观测**正确嵌入同一混合误差状态，并分场景选择松/紧耦合。

### 1.2 现有方法问题

1. **协方差错配。** GICP 给出的是相对位姿在某一扰动约定下的 $\Sigma_{\mathrm{gicp}}$，直接当作世界系绝对位姿噪声，会忽略关键帧旋转、相对平移杠杆及左右扰动差异。  
2. **雅可比错套。** 独立 Stereo-GICP 的 $J_{\mathrm{left}}=[-[p']_\times\mid I]$ 对应目标系左扰动；滤波器若对机体做右乘 $\mathrm{Exp}(\delta\theta)$、对位置加 $\delta p_w$，存在平移时 $J_{\mathrm{geo}}\neq J_{\mathrm{left}}$。错套会导致更新方向错误甚至发散。  
3. **点级信息难进卡尔曼。** 逐点 $J^\top W J$ 累加后如何变成标准的 $R_{\mathrm{obs}},r_v$ 并与 $H_{\mathrm{sel}}$、Joseph 形式衔接，文献中常一笔带过，工程易实现成“只更新均值、协方差随意缩小”。

### 1.3 本文思路

以**状态估计**为主贡献：

1. 定义混合误差：$R\leftarrow R\,\mathrm{Exp}(\delta\theta)$，$p\leftarrow p+\delta p_w$，完整 $\delta x\in\mathbb{R}^{21}$，观测用 $H_{\mathrm{sel}}$ 只看前 6 维；  
2. **非关键帧 ESKF**：IMU 传播 → Stereo-GICP/GICP 升世界系位姿 → 创新 $z$ → $R_{\mathrm{obs}}=J_{g\to e}\Sigma_{\mathrm{gicp}}J_{g\to e}^\top$ → 一次卡尔曼 + Joseph；  
3. **关键帧 ESIKF**：锚点固定；相机系投影关联（命中像素中心光线）+ 马氏几何（Huber）+ 切平面光度 → 聚合 $H_{\mathrm{obs}},b_{\mathrm{obs}}$ → 虚拟观测 $(R_{\mathrm{obs}},r_v)$ → iESKF 重线性化。

几何关联、立体 $\Sigma$、匹配噪声 $\sigma_u^{\mathrm{match}}=0.5\,\mathrm{px}$ 与 Huber 引用姊妹篇 [投影 Stereo-GICP](SCI_投影StereoGICP.md)，本文不重复 $J_{uvd}$。公式见第 3 节。

### 1.4 主要贡献

1. 统一松/紧耦合的混合误差与 $H_{\mathrm{sel}}$ 接口，保证概率意义一致。  
2. 给出相对 GICP 协方差到 ESKF 观测噪声的 $J_{g\to e}$ 推导。  
3. 逐步导出 ESIKF 的 $J_{\mathrm{geo}}$，并证明有平移时与 $J_{\mathrm{left}}$ 不等价。  
4. 给出几何-光度信息聚合、虚拟观测与 Joseph/iESKF 流程，用 ATE 与 NEES/NIS 验证。

## 2. 方法说明

### 2.1 混合误差状态（为何这样定义）

旋转用机体右乘、平移用世界系加法，与常见 IMU 误差状态滤波一致，也便于和预积分、外参链式法则衔接。完整误差还含速度、偏置、重力等；视觉位姿观测不直接写这些量的显式雅可比，而是通过 $P$ 的交叉协方差间接修正——这正是 $H_{\mathrm{sel}}$ 只取 6 维的原因。

### 2.2 非关键帧 ESKF（低成本路径）

每帧：IMU 预测先验 → 在静态关键帧点云上配准得相对 $T_{k\leftarrow c}$ → 左乘关键帧世界位姿升为绝对观测 → 算创新 $z$ → 用 $J_{g\to e}$ 对齐噪声 → 一次更新。适合高频维护轨迹，不在每帧上堆点级光度。

### 2.3 关键帧 ESIKF（高精度路径）

固定锚点关键帧。源点先变到锚点机体，再变到关键帧左目相机；在深度图上做自适应半径投影关联，目标点取命中像素的中心光线，再映回机体。每个内点贡献马氏几何信息（目标横向用 $0.5\,\mathrm{px}$ 匹配噪声，白化残差过大时 Huber 降权），可选切平面光度；全部累加后正则成六维虚拟观测，再迭代重关联/重线性化。代价高，但关键帧位姿与 target cache 质量直接影响后续许多帧的松耦合。

### 2.4 与姊妹模块的接口

- 动态滤波 → 静态索引，减少动态污染；  
- Stereo-GICP → 非关键帧相对 $T$ 与左扰动 $\Sigma_{\mathrm{gicp}}$；关键帧点级关联 $\mathbf{q}$、$\Sigma_j$、$W$ 与 Huber；  
- 本模块 → 统一误差状态下的 $P^+$ 与名义位姿，供后端 3DGS 关键帧使用。

## 3. 理论推导

每个显示公式后给出符号含义；$H_{\mathrm{sel}}$、$J_{g\to e}$、$J_{\mathrm{geo}}$、$J_{\mathrm{color}}$ 细导见 §3.2–§3.4。姊妹篇：[动态观测滤波](SCI_动态观测滤波.md)、[投影 Stereo-GICP](SCI_投影StereoGICP.md)。

### 3.1 名义状态与混合误差状态

#### 名义状态

$$
\mathbf{x}
=
\bigl(
R_{wb},\,
\mathbf{p}_{wb},\,
\mathbf{v}_{wb},\,
\mathbf{b}_g,\,
\mathbf{b}_a,\,
\mathbf{g}_w
\bigr).
$$

其中：

- $R_{wb}$：机体→世界旋转；
- $\mathbf{p}_{wb}$：机体世界系位置（m）；
- $\mathbf{v}_{wb}$：世界系速度（m/s）；
- $\mathbf{b}_g,\mathbf{b}_a$：陀螺 / 加计偏置；
- $\mathbf{g}_w$：世界系重力。

#### 混合误差注入

$$
R_{wb}^{+}
=
R_{wb}\,\mathrm{Exp}(\delta\boldsymbol{\theta}),
\qquad
\mathbf{p}_{wb}^{+}
=
\mathbf{p}_{wb}+\delta\mathbf{p}_w.
$$

其中：

- $\delta\boldsymbol{\theta}$：机体**右**切空间旋转误差（rad）；
- $\delta\mathbf{p}_w$：世界系位置误差（m）；
- $\mathrm{Exp}$：$\mathfrak{so}(3)\to\mathrm{SO}(3)$。

位姿误差子向量：

$$
\delta\boldsymbol{\xi}_{\mathrm{pose}}
=
\begin{bmatrix}\delta\boldsymbol{\theta}\\\delta\mathbf{p}_w\end{bmatrix}
\in\mathbb{R}^6.
$$

完整误差状态 $\delta\mathbf{x}\in\mathbb{R}^{21}$（工程维数）还含速度、偏置、重力等增量。

#### 反对称恒等式（后文反复使用）

$$
[\mathbf{v}]_\times
=
\begin{bmatrix}
0 & -v_z & v_y \\
v_z & 0 & -v_x \\
-v_y & v_x & 0
\end{bmatrix},
\qquad
\mathbf{a}\times\mathbf{b}=[\mathbf{a}]_\times\mathbf{b}=-[\mathbf{b}]_\times\mathbf{a}.
$$

以及旋转作用：

$$
R(\delta\boldsymbol{\theta}\times\mathbf{p})
=
R[\delta\boldsymbol{\theta}]_\times\mathbf{p}
=
-R[\mathbf{p}]_\times\delta\boldsymbol{\theta}.
$$

---

### 3.2 松耦合 ESKF（非关键帧）

#### 流程

1. IMU 传播 → 先验 $(\hat R,\hat{\mathbf{p}})$、$P_{\mathrm{pred}}$；
2. 在动态滤波静态关键帧点云上，以 IMU 相对位姿为初值跑 Stereo-GICP（投影关联；失败则 small_gicp）→ $T_{k\leftarrow c}$；
3. 升绝对观测 $T_{w\leftarrow c}^{\mathrm{gicp}}=T_{w\leftarrow k}T_{k\leftarrow c}$；
4. 按配准器扰动约定对齐 $R_{\mathrm{obs}}$（Stereo-GICP 左扰动见下；small_gicp 右扰动见 §3.3），一次误差状态卡尔曼更新。

其中 $P_{\mathrm{pred}}\in\mathbb{R}^{21\times 21}$ 为预测协方差；$T_{k\leftarrow c}$ 为「关键帧←当前」相对位姿（前端已映到 body）。

#### 观测方程与创新

$$
\mathbf{z}
=
\begin{bmatrix}
\mathrm{Log}(\hat R^\top R_{\mathrm{gicp}}) \\
\mathbf{p}_{\mathrm{gicp}}-\hat{\mathbf{p}}
\end{bmatrix}
=
H_{\mathrm{sel}}\,\delta\mathbf{x}
+\mathbf{n},
\qquad
\mathbf{n}\sim\mathcal{N}(\mathbf{0},R_{\mathrm{obs}}).
$$

其中：

- $\mathbf{z}\in\mathbb{R}^6$：位姿创新（旋转轴角 + 平移）；
- $R_{\mathrm{gicp}},\mathbf{p}_{\mathrm{gicp}}$：GICP 世界系观测；
- $\mathrm{Log}$：$\mathrm{SO}(3)\to\mathbb{R}^3$；
- $R_{\mathrm{obs}}$：观测噪声协方差（§3.3 对齐得到）。

在线性化点 $\delta\mathbf{x}=\mathbf{0}$ 处，创新即 $\mathbf{z}$ 本身。

#### 观测雅可比 $H_{\mathrm{sel}}$

位姿观测只直接依赖前 6 维误差：

$$
H_{\mathrm{sel}}
=
\begin{bmatrix}
I_3 & 0 & 0_{3\times 15} \\
0 & I_3 & 0_{3\times 15}
\end{bmatrix}
\in\mathbb{R}^{6\times 21}.
$$

其中：

- 左上 $I_3$：$\partial\mathbf{z}_{\mathrm{rot}}/\partial\delta\boldsymbol{\theta}$；
- 中块 $I_3$：$\partial\mathbf{z}_{\mathrm{pos}}/\partial\delta\mathbf{p}_w$；
- 其余为 0。速度/偏置通过 $P_{\mathrm{pred}}$ 的**交叉协方差**间接修正。

#### 建立方程与解算（松耦合，一次更新）

**步骤 1：观测与噪声。** 配准给出世界系 $R_{\mathrm{gicp}},\mathbf{p}_{\mathrm{gicp}}$ 与相对 $\Sigma_{\mathrm{gicp}}$。创新

$$
\mathbf{z}
=
\begin{bmatrix}
\mathrm{Log}(\hat R^\top R_{\mathrm{gicp}}) \\
\mathbf{p}_{\mathrm{gicp}}-\hat{\mathbf{p}}
\end{bmatrix}.
$$

Stereo-GICP 用 $R_{\mathrm{obs}}=J_{\mathrm{obs}}\Sigma_{\mathrm{gicp}}J_{\mathrm{obs}}^\top$；small_gicp 用 $R_{\mathrm{obs}}=J_{g\to e}\Sigma_{\mathrm{gicp}}J_{g\to e}^\top$（§3.3）。

**步骤 2：创新协方差与增益。** 观测方程 $\mathbf{z}=H_{\mathrm{sel}}\delta\mathbf{x}+\mathbf{n}$，故

$$
S=H_{\mathrm{sel}}P_{\mathrm{pred}}H_{\mathrm{sel}}^\top+R_{\mathrm{obs}}
=
P_{\mathrm{pose}}+R_{\mathrm{obs}}.
$$

对 $S$ 加小阻尼 $\varepsilon=10^{-6}\max(1,\mathrm{tr}(S)/6)$ 后 LDLT，得

$$
K=P_{\mathrm{pred}}H_{\mathrm{sel}}^\top\,S^{-1}\in\mathbb{R}^{21\times 6}.
$$

其中 $S$ 是「先验位姿不确定度 + 观测噪声」；$K$ 把 6 维创新映到 21 维误差修正。

**步骤 3：解增量并注入。**

$$
\delta\hat{\mathbf{x}}=K\mathbf{z},
\qquad
R\leftarrow R\,\mathrm{Exp}(\delta\hat{\boldsymbol{\theta}}),\ 
\mathbf{p}\leftarrow\mathbf{p}+\delta\hat{\mathbf{p}},\ 
\mathbf{v},\mathbf{b}_g,\mathbf{b}_a,\mathbf{g}_w\leftarrow\ldots
$$

**步骤 4：Joseph 协方差。**

$$
P^{+}
=
(I-KH_{\mathrm{sel}})P_{\mathrm{pred}}(I-KH_{\mathrm{sel}})^\top
+
K R_{\mathrm{obs}} K^\top.
$$

其中 $I$ 为 $21\times 21$ 单位阵。非关键帧只做这一次，不重关联。

---

### 3.3 GICP 协方差 → ESKF 混合坐标（逐步）

#### 相对右扰动与绝对位姿

`small_gicp` 给出相对 $T_{kc}=(R_{kc},\mathbf{t}_k)$ 在**右扰动**下的 $\Sigma_{\mathrm{gicp}}$：

$$
T_{kc}^{+}
=
T_{kc}\circ\exp(\delta\boldsymbol{\xi}_r),
\qquad
\delta\boldsymbol{\xi}_r
=
\begin{bmatrix}\delta\boldsymbol{\theta}_{\mathrm{curr}}\\ \delta\mathbf{t}_{\mathrm{curr}}\end{bmatrix}.
$$

其中 $\delta\boldsymbol{\xi}_r$ 在 source/当前切空间。绝对位姿：

$$
R_{wc}=R_{wk}R_{kc},
\qquad
\mathbf{p}_{wc}=\mathbf{p}_{wk}+R_{wk}\mathbf{t}_k.
$$

其中 $R_{wk},\mathbf{p}_{wk}$ 为关键帧世界位姿。

#### 旋转误差传播

$$
R_{wc}^{+}
=
R_{wk}R_{kc}\,\mathrm{Exp}(\delta\boldsymbol{\theta}_{\mathrm{curr}})
=
R_{wc}\,\mathrm{Exp}(\delta\boldsymbol{\theta}_{\mathrm{curr}}).
$$

与 ESKF 右乘约定一致，故

$$
\delta\boldsymbol{\theta}_b=\delta\boldsymbol{\theta}_{\mathrm{curr}}
\quad\Rightarrow\quad
\frac{\partial\delta\boldsymbol{\theta}_b}{\partial\delta\boldsymbol{\theta}_{\mathrm{curr}}}=I_3,\quad
\frac{\partial\delta\boldsymbol{\theta}_b}{\partial\delta\mathbf{t}_{\mathrm{curr}}}=0.
$$

#### 平移误差传播

相对右扰动下，目标系中平移增量含旋转耦合：

$$
\delta\mathbf{t}_{\mathrm{tgt}}
=
\delta\boldsymbol{\theta}_{\mathrm{curr}}\times\mathbf{t}_k
+
R_{kc}\,\delta\mathbf{t}_{\mathrm{curr}}.
$$

映到世界系：

$$
\begin{aligned}
\delta\mathbf{p}_w
&=
R_{wk}\,\delta\mathbf{t}_{\mathrm{tgt}}
\\
&=
R_{wk}(\delta\boldsymbol{\theta}_{\mathrm{curr}}\times\mathbf{t}_k)
+
R_{wk}R_{kc}\,\delta\mathbf{t}_{\mathrm{curr}}
\\
&=
R_{wk}[\mathbf{t}_k]_\times\delta\boldsymbol{\theta}_{\mathrm{curr}}
+
R_{wk}R_{kc}\,\delta\mathbf{t}_{\mathrm{curr}}.
\end{aligned}
$$

（此处用 $\boldsymbol{\omega}\times\mathbf{t}=[\mathbf{t}]_\times\boldsymbol{\omega}$ 的符号约定与代码 `R_wk * Skew(t_k)` 对齐。）

#### 对齐雅可比（为何必须变、怎么变）

GICP 的 $\Sigma_{\mathrm{gicp}}$ 与 ESKF 的 $P$、$H_{\mathrm{sel}}$ **不在同一切空间**：

| | GICP（`small_gicp`） | ESKF 混合误差 |
|--|---------------------|----------------|
| 对象 | 相对 $T_{kc}=T_{k\leftarrow c}$ | 当前绝对位姿 $T_{w\leftarrow c}$ |
| 扰动 | 右乘 $\delta\boldsymbol{\xi}_r=(\delta\boldsymbol{\theta}_{\mathrm{curr}},\delta\mathbf{t}_{\mathrm{curr}})$，在当前/source 切空间 | $R\leftarrow R\,\mathrm{Exp}(\delta\boldsymbol{\theta}_b)$，$\mathbf{p}\leftarrow\mathbf{p}+\delta\mathbf{p}_w$ |
| 平移坐标 | 相对平移的右扰动（机体系） | 世界系位置 |

创新 $\mathbf{z}$ 已按混合误差写（$\mathrm{Log}(\hat R^\top R_{\mathrm{gicp}})$ 与 $\mathbf{p}_{\mathrm{gicp}}-\hat{\mathbf{p}}$）。若把 $\Sigma_{\mathrm{gicp}}$ 原样当作 $R_{\mathrm{obs}}$，等于假设「相对右扰动的方差 = 世界系位置方差」，卡尔曼增益 $K=P H^\top(HPH^\top+R_{\mathrm{obs}})^{-1}$ 会在错误的椭圆上收缩 $P$，NEES 会偏。对齐就是一阶映射

$$
\begin{bmatrix}\delta\boldsymbol{\theta}_b\\ \delta\mathbf{p}_w\end{bmatrix}
=
J_{g\to e}\,
\begin{bmatrix}\delta\boldsymbol{\theta}_{\mathrm{curr}}\\ \delta\mathbf{t}_{\mathrm{curr}}\end{bmatrix},
\qquad
R_{\mathrm{obs}}
=
J_{g\to e}\,\Sigma_{\mathrm{gicp}}\,J_{g\to e}^\top.
$$

由上面旋转 / 平移传播，按列写出：

$$
J_{g\to e}
=
\begin{bmatrix}
I_3 & 0 \\
R_{wk}[\mathbf{t}_k]_\times & R_{wk}R_{kc}
\end{bmatrix}.
$$

分块含义：

- **左上 $I_3$**：相对右乘旋转与机体右乘旋转同约定，$\delta\boldsymbol{\theta}_b=\delta\boldsymbol{\theta}_{\mathrm{curr}}$，不放大也不换轴。  
- **右上 $0$**：相对平移不改旋转。  
- **右下 $R_{wk}R_{kc}$**：把当前机体系的 $\delta\mathbf{t}_{\mathrm{curr}}$ 经 $R_{wc}=R_{wk}R_{kc}$ 转到世界系，才与 $\delta\mathbf{p}_w$ 同坐标。  
- **左下 $R_{wk}[\mathbf{t}_k]_\times$**：相对旋转会通过杠杆 $\mathbf{t}_k$ 变成目标系平移，再转到世界系。$\lVert\mathbf{t}_k\rVert$ 越大（关键帧与当前拉得越开），同样的 yaw 不确定度会在 $R_{\mathrm{obs}}$ 的位置块上放得越大，并写出 $P_{\theta p}$ 型交叉。代码 `R_wk * Skew(t_k)` 与 $\boldsymbol{\omega}\times\mathbf{t}=[\mathbf{t}]_\times\boldsymbol{\omega}$ 一致。

实现：`BuildGicpToEskfPoseJacobian` 组 $J_{g\to e}$，`AlignGicpCovarianceToEskfPose` 做 $J\Sigma J^\top$ 再对称化。前端在机体系配准时直接调用后者（`gicp_out.T` 即 $T_{kc}$，$R_{wk}$ 取关键帧 `rot_end`）。

**不对齐（消融「$J_{g\to e}$ 关」）**：令 $R_{\mathrm{obs}}=\Sigma_{\mathrm{gicp}}$。旋转块偶然还接近，位置块差一个 $R_{wk}R_{kc}$，且丢掉杠杆交叉。预期：ATE 可能仍可用，但 NIS / NEES 变差，$3\sigma$ 包络与误差对不齐；$\lVert\mathbf{t}_k\rVert$ 大的帧上更明显。

**配准在相机系时的两步对齐。** $\Sigma_{\mathrm{gicp}}$ 先在相机 se(3) 右扰动下。先共轭到机体系，再套 $J_{g\to e}$：

$$
T_{kc}^{\mathrm{body}}
=
T_{\mathrm{imu}\leftarrow\mathrm{cam}}\,
T_{kc}^{\mathrm{cam}}\,
T_{\mathrm{imu}\leftarrow\mathrm{cam}}^{-1},
\qquad
\Sigma_{\mathrm{body}}
=
\mathrm{Ad}_{T_{b\leftarrow c}}\,
\Sigma_{\mathrm{gicp}}^{\mathrm{cam}}\,
\mathrm{Ad}_{T_{b\leftarrow c}}^{\top},
$$

$$
R_{\mathrm{obs}}
=
J_{g\to e}(R_{wk},T_{kc}^{\mathrm{body}})\,
\Sigma_{\mathrm{body}}\,
J_{g\to e}^{\top}.
$$

其中 $\mathrm{Ad}$ 为 se(3) 右扰动伴随（`Se3Adjoint`，旋转在前）。`AlignGicpCovarianceCameraToEskfPose` 实现这两步，供 demo 等相机系配准路径使用。注意：这里的 $\mathrm{Ad}$ 是把**相对配准噪声**从相机拧到机体，与动态滤波里把 **IMU 状态 $P$** 变到相机系的 $A$ 方向相反，不要混用。

#### Stereo-GICP 左扰动协方差 → 混合坐标

独立配准器 `AlignProjective` 的 $\Sigma_{\mathrm{gicp}}$ 是**目标系左扰动**下的 $H^{-1}$，再经 `CameraPoseCovToBody` 映到 body（仍是左扰动坐标）。它与 §3.3 的 small_gicp **右扰动**不是同一约定，不能套 `AlignGicpCovarianceToEskfPose`。

相对位姿 $T_{kc}=(R_{kc},\mathbf{t}_k)$ 已在 body。左扰动增量 $\delta\boldsymbol{\xi}_L=(\delta\boldsymbol{\omega}_L,\delta\mathbf{t}_L)$ 在关键帧/目标机体系。映到 ESKF 混合误差：

$$
\delta\boldsymbol{\theta}_b
=
R_{kc}^\top\,\delta\boldsymbol{\omega}_L,
\qquad
\delta\mathbf{p}_w
=
R_{wk}\,\delta\mathbf{t}_L.
$$

其中：

- $R_{kc}^\top$：把目标系左切旋转变到当前机体右切（与创新 $\mathrm{Log}(\hat R^\top R_{\mathrm{gicp}})$ 同轴）；
- $R_{wk}$：把目标机体系平移变到世界系位置。

故

$$
J_{\mathrm{obs}}
=
\begin{bmatrix}
R_{kc}^\top & 0 \\
0 & R_{wk}
\end{bmatrix},
\qquad
R_{\mathrm{obs}}
=
J_{\mathrm{obs}}\,\Sigma_{\mathrm{gicp}}^{\mathrm{body}}\,J_{\mathrm{obs}}^\top.
$$

前端在 `used_stereo_gicp` 为真时组该 $J_{\mathrm{obs}}$；否则走上一节的右扰动 $J_{g\to e}$。

---

### 3.4 紧耦合 ESIKF（关键帧）

#### 相对变换

锚点关键帧 $(R_a,\mathbf{t}_a)$ 固定，当前名义 $(R_c,\mathbf{p}_c)$：

$$
R_{st}=R_a^\top R_c,
\qquad
\mathbf{t}_{st}=R_a^\top(\mathbf{p}_c-\mathbf{t}_a).
$$

其中：

- $R_{st}$：当前→锚点的相对旋转；
- $\mathbf{t}_{st}$：当前原点在锚点机体系中的位置。

源点（当前机体系）$\mathbf{p}_{\mathrm{src}}$：

$$
\mathbf{p}'=R_{st}\mathbf{p}_{\mathrm{src}}+\mathbf{t}_{st}.
$$

其中 $\mathbf{p}'$ 在锚点机体系。

#### 投影关联（相机系，与文档二同一实现）

有关键帧深度图且内参有效时，优先投影关联；否则欧氏最近邻。关联在关键帧**左目相机系**完成：

1. $\mathbf{p}'$ 经 $T_{\mathrm{body}\leftarrow\mathrm{cam}}^{-1}$ 变到相机系得 $\mathbf{p}'_{\mathrm{cam}}$；
2. 调用 `ProjectiveAssociate`：自适应半径 $r(Z)=\mathrm{clamp}(\lceil f_x\cdot 0.10/Z\rceil,2,15)$，边缘只看 $1\,\mathrm{px}$ 邻域，$3\sigma_Z$ 门控；
3. 命中像素 $(b_u,b_v)$、深度 $Z_j$ 时，目标点取**像素中心光线**

$$
\mathbf{q}_{\mathrm{cam}}
=
\begin{pmatrix}
(b_u-c_x)Z_j/f_x \\
(b_v-c_y)Z_j/f_y \\
Z_j
\end{pmatrix},
$$

再映回锚点机体得 $\mathbf{q}$。投影正好落在像素中心时残差沿视线；偏离半个像素时横向残差保留。细节与门控见 [投影 Stereo-GICP §3.3](SCI_投影StereoGICP.md)。

#### 几何残差（嵌入误差状态）

$$
\mathbf{e}=\mathbf{p}'-\mathbf{q},
\qquad
C=R_{st}\Sigma_{\mathrm{src}}R_{st}^\top+\Sigma_{\mathrm{tgt}},
\qquad
W=C^{-1}.
$$

其中：

- $\mathbf{e}\in\mathbb{R}^3$：锚点机体系点到点残差（m）；
- $\Sigma_{\mathrm{src}}$：源点立体协方差（已在当前机体系；三角化噪声 $\sigma_u=\sigma_v=0.5\,\mathrm{px}$，$\sigma_d=0.3\,\mathrm{px}$）；
- $\Sigma_{\mathrm{tgt}}$：目标点协方差，在 $\mathbf{q}_{\mathrm{cam}}$ 上用匹配噪声 $\sigma_u^{\mathrm{match}}=\sigma_v^{\mathrm{match}}=0.5\,\mathrm{px}$ 与 $\sigma_d=0.3\,\mathrm{px}$ 现算，再旋到机体；
- 一次线性化内视 $\mathbf{q}$ 与 $\delta\boldsymbol{\xi}_{\mathrm{pose}}$ 无关。

#### 几何雅可比逐步推导

**步骤 A.** 扰动名义：

$$
R_c^{+}=R_c\,\mathrm{Exp}(\delta\boldsymbol{\theta}),
\qquad
\mathbf{p}_c^{+}=\mathbf{p}_c+\delta\mathbf{p}_w.
$$

**步骤 B.** 相对量变化：

$$
\begin{aligned}
R_{st}^{+}
&=
R_a^\top R_c\,\mathrm{Exp}(\delta\boldsymbol{\theta})
=
R_{st}\,\mathrm{Exp}(\delta\boldsymbol{\theta}),
\\
\mathbf{t}_{st}^{+}
&=
R_a^\top(\mathbf{p}_c+\delta\mathbf{p}_w-\mathbf{t}_a)
=
\mathbf{t}_{st}+R_a^\top\delta\mathbf{p}_w.
\end{aligned}
$$

**步骤 C.** 变换点：

$$
\begin{aligned}
\mathbf{p}'^{+}
&=
R_{st}\,\mathrm{Exp}(\delta\boldsymbol{\theta})\,\mathbf{p}_{\mathrm{src}}
+\mathbf{t}_{st}+R_a^\top\delta\mathbf{p}_w
\\
&\approx
R_{st}(\mathbf{p}_{\mathrm{src}}+\delta\boldsymbol{\theta}\times\mathbf{p}_{\mathrm{src}})
+\mathbf{t}_{st}+R_a^\top\delta\mathbf{p}_w
\\
&=
\mathbf{p}'
+R_{st}(\delta\boldsymbol{\theta}\times\mathbf{p}_{\mathrm{src}})
+R_a^\top\delta\mathbf{p}_w.
\end{aligned}
$$

**步骤 D.** 用反对称恒等式：

$$
R_{st}(\delta\boldsymbol{\theta}\times\mathbf{p}_{\mathrm{src}})
=
-R_{st}[\mathbf{p}_{\mathrm{src}}]_\times\delta\boldsymbol{\theta}.
$$

故

$$
\mathrm{d}\mathbf{p}'
=
-R_{st}[\mathbf{p}_{\mathrm{src}}]_\times\delta\boldsymbol{\theta}
+R_a^\top\delta\mathbf{p}_w,
$$

$$
J_{\mathrm{geo}}
=
\begin{bmatrix}
-R_{st}[\mathbf{p}_{\mathrm{src}}]_\times & R_a^\top
\end{bmatrix}
\in\mathbb{R}^{3\times 6},
\qquad
\mathrm{d}\mathbf{e}=J_{\mathrm{geo}}\,\delta\boldsymbol{\xi}_{\mathrm{pose}}.
$$

其中左块对旋转误差，右块对世界系平移误差。

#### 与 $J_{\mathrm{left}}$ 不等价（证明）

若对相对 $T_{st}$ 做目标系左扰动，则

$$
J_{\mathrm{left}}=\bigl[-[\mathbf{p}']_\times\ \big|\ I\bigr].
$$

混合误差与左扰动增量的一阶关系为

$$
\delta\boldsymbol{\omega}_{\mathrm{left}}=R_{st}\delta\boldsymbol{\theta},
\qquad
\delta\mathbf{t}_{\mathrm{left}}=R_a^\top\delta\mathbf{p}_w.
$$

复合：

$$
J_{\mathrm{left}}
\begin{bmatrix} R_{st} & 0 \\ 0 & R_a^\top \end{bmatrix}
=
\begin{bmatrix}
-[\mathbf{p}']_\times R_{st} & R_a^\top
\end{bmatrix}.
$$

正确 ESIKF 结果是 $-R_{st}[\mathbf{p}_{\mathrm{src}}]_\times$。二者相等当且仅当

$$
[\mathbf{p}']_\times R_{st}=R_{st}[\mathbf{p}_{\mathrm{src}}]_\times,
$$

即 $\mathbf{p}'=R_{st}\mathbf{p}_{\mathrm{src}}$（**无平移**）。一般 $\mathbf{p}'=R_{st}\mathbf{p}_{\mathrm{src}}+\mathbf{t}_{st}$，故

$$
-[\mathbf{p}']_\times R_{st}
\;\neq\;
-R_{st}[\mathbf{p}_{\mathrm{src}}]_\times.
$$

**结论**：ESIKF 必须用上文的 $J_{\mathrm{geo}}$，不能套用 [投影 Stereo-GICP](SCI_投影StereoGICP.md) 的 $J_{\mathrm{left}}$。

#### 几何 / 光度雅可比的实现对照

代码（`BuildPointObservationFactor`）：

```text
J_geo_mat.block<3,3>(0,0) = -R_state_to_target * SkewSymmetric(p_src);
J_geo_mat.block<3,3>(0,3) = R_target_to_world.transpose();  // = R_aᵀ
```

其中 $R_{\mathrm{state\_to\_target}}=R_{st}$，$R_{\mathrm{target\_to\_world}}=R_a$；  
$\mathbf{p}_{\mathrm{src}}=R_{st}^\top(\mathbf{p}'-\mathbf{t}_{st})$（`sensor_origin_in_state=0` 时）。

| 公式量 | 代码入口 |
|--------|----------|
| $H_{\mathrm{sel}}$、$K$、Joseph | `StateEstimator::ESKF` / `ESIKF` |
| small_gicp $J_{g\to e}$、$R_{\mathrm{obs}}$ | `BuildGicpToEskfPoseJacobian` / `AlignGicpCovarianceToEskfPose`（`utils.h`） |
| Stereo-GICP 左扰动 $J_{\mathrm{obs}}$ | `frontend.cpp`：`J_obs=diag(R_{kc}^\top,R_{wk})$ |
| 相机系 $\Sigma$ 先共轭到 body | `CameraPoseCovToBody` / `AlignGicpCovarianceCameraToEskfPose` |
| $J_{\mathrm{geo}}$、$J_{\mathrm{color}}$、$C,W$ | `BuildPointObservationFactor` |
| Huber、$H_{\mathrm{obs}},b_{\mathrm{obs}}$ 累加 | `ColorPointObserver`（`HuberWeight`） |
| 投影关联 | `BuildProjectiveObservationSubclouds` → `ProjectiveAssociate` |

#### 光度残差与雅可比（逐步）

切平面颜色梯度：

$$
\mathbf{d}_M
=
\bigl(I-\hat{\mathbf{n}}\hat{\mathbf{n}}^\top\bigr)\nabla I.
$$

其中 $\hat{\mathbf{n}}$ 为单位法向；$\nabla I$ 为强度梯度；$I$ 为单位阵；$\mathbf{d}_M$ 去掉法向分量。

光度残差：

$$
r_{\mathrm{color}}
=
I(\mathbf{q})+\mathbf{d}_M^\top(\mathbf{p}'-\mathbf{q})-I(\mathbf{p}').
$$

其中 $I(\cdot)$ 为点处强度（与单位阵符号同形，依上下文区分）。

微分（仅 $\mathbf{p}'$ 随状态变）：

$$
\mathrm{d}r_{\mathrm{color}}
=
\mathbf{d}_M^\top\mathrm{d}\mathbf{p}'
=
\mathbf{d}_M^\top\bigl(-R_{st}[\mathbf{p}_{\mathrm{src}}]_\times\delta\boldsymbol{\theta}+R_a^\top\delta\mathbf{p}_w\bigr).
$$

旋转项整理：

$$
\begin{aligned}
\mathbf{d}_M^\top\bigl(-R_{st}[\mathbf{p}_{\mathrm{src}}]_\times\delta\boldsymbol{\theta}\bigr)
&=
-(R_{st}^\top\mathbf{d}_M)^\top[\mathbf{p}_{\mathrm{src}}]_\times\delta\boldsymbol{\theta}
\\
&=
\delta\boldsymbol{\theta}^\top
\bigl(\mathbf{p}_{\mathrm{src}}\times(R_{st}^\top\mathbf{d}_M)\bigr).
\end{aligned}
$$

平移项：

$$
\mathbf{d}_M^\top R_a^\top\delta\mathbf{p}_w
=
(R_a\mathbf{d}_M)^\top\delta\mathbf{p}_w.
$$

故

$$
J_{\mathrm{color}}^\top
=
\begin{bmatrix}
\mathbf{p}_{\mathrm{src}}\times(R_{st}^\top\mathbf{d}_M) \\
R_a\mathbf{d}_M
\end{bmatrix}
\in\mathbb{R}^6.
$$

光度权重 $w_{\mathrm{color}}>0$ 由位置协方差在 $\mathbf{d}_M$ 方向的投影及纹理强度自适应给出。

#### 建立观测方程与解算（逐步）

以下对应 `ESIKF` 的一轮 iESKF 迭代。$P_{\mathrm{pred}}$ 在整段迭代中固定；名义位姿每轮更新。对应数少于 $20$ 则本轮跳过。

**步骤 1：重线性化相对位姿与关联。** 由当前名义 $(R_c,\mathbf{p}_c)$ 算 $R_{st},\mathbf{t}_{st}$，把源点变到锚点机体得 $\mathbf{p}'$，再相机系投影关联得到 $\mathbf{q}$（像素中心光线）。内层视 $\mathbf{q}$ 固定。

**步骤 2：单点几何 / 光度因子。** 对每个对应（`BuildPointObservationFactor`）：

$$
\mathbf{e}=\mathbf{p}'-\mathbf{q},
\quad
C=R_{st}\Sigma_{\mathrm{src}}R_{st}^\top+\Sigma_{\mathrm{tgt}},
\quad
W=C^{-1},
\quad
J_{\mathrm{geo}}=\bigl[-R_{st}[\mathbf{p}_{\mathrm{src}}]_\times\ \big|\ R_a^\top\bigr],
$$

以及可选的 $r_{\mathrm{color}},J_{\mathrm{color}},w_{\mathrm{color}}$。白化残差 $s=\mathbf{e}^\top W\mathbf{e}$，Huber 权重 $w=w(s)$（与文档二同核，$\delta=1.345$）。

**步骤 3：累加 6×6 观测信息。** 几何权重 $\lambda\in[0,1]$（前端默认 $0.5$）：

$$
\begin{aligned}
H_{\mathrm{obs}}
&\mathrel{+}=
\lambda\,w\,J_{\mathrm{geo}}^\top W J_{\mathrm{geo}}
+(1-\lambda)\,w_{\mathrm{color}}\,J_{\mathrm{color}}J_{\mathrm{color}}^\top,
\\
\mathbf{b}_{\mathrm{obs}}
&\mathrel{+}=
\lambda\,w\,J_{\mathrm{geo}}^\top W\mathbf{e}
+(1-\lambda)\,w_{\mathrm{color}}\,J_{\mathrm{color}}\,r_{\mathrm{color}}.
\end{aligned}
$$

其中：

- $H_{\mathrm{obs}}\in\mathbb{R}^{6\times 6}$：所有点堆完后的观测信息矩阵；
- $\mathbf{b}_{\mathrm{obs}}\in\mathbb{R}^6$：观测信息向量；
- 几何项的 $J^\top W\mathbf{e}$ 与独立配准的 $\mathbf{g}$ 同形，只是 $J$ 换成 $J_{\mathrm{geo}}$。

半像素取整残差通常仍在 Huber 核内；邻像素误配使 $s$ 变大、$w$ 下降。

**步骤 4：压成虚拟六维观测。** 点级二次型 $\tfrac12\delta^\top H_{\mathrm{obs}}\delta+\mathbf{b}_{\mathrm{obs}}^\top\delta$ 的驻点满足

$$
H_{\mathrm{obs}}\,\delta+\mathbf{b}_{\mathrm{obs}}=\mathbf{0}
\quad\Leftrightarrow\quad
H_{\mathrm{obs}}\,\delta=-\mathbf{b}_{\mathrm{obs}}.
$$

这与独立 Stereo-GICP 的 $H\delta=-\mathbf{g}$ 同形（此处 $\mathbf{b}_{\mathrm{obs}}$ 对应 $\mathbf{g}$）。加阻尼后

$$
\varepsilon=10^{-6}\max\bigl(1,\mathrm{tr}(H_{\mathrm{obs}})/6\bigr),
\qquad
H_{\mathrm{reg}}=H_{\mathrm{obs}}+\varepsilon I_6.
$$

对 $H_{\mathrm{reg}}$ 做 LDLT，解

$$
\boxed{H_{\mathrm{reg}}\,\mathbf{r}_v=-\mathbf{b}_{\mathrm{obs}},}
\qquad
R_{\mathrm{obs}}=H_{\mathrm{reg}}^{-1}.
$$

其中：

- $\mathbf{r}_v=-H_{\mathrm{reg}}^{-1}\mathbf{b}_{\mathrm{obs}}\in\mathbb{R}^6$：虚拟创新，即「只看点云时」混合误差坐标下的 Gauss–Newton 步；
- $R_{\mathrm{obs}}\in\mathbb{R}^{6\times 6}$：虚拟观测噪声（信息矩阵的逆）。

$\mathbf{r}_v$ 解出后不再经 $H$ 作用；状态增量由下一步的卡尔曼增益给出：$\delta\hat{\mathbf{x}}=K\mathbf{r}_v$。观测模型为 $\mathbf{r}_v=H_{\mathrm{sel}}\delta\mathbf{x}+\mathbf{n}$，$\mathbf{n}\sim\mathcal{N}(\mathbf{0},R_{\mathrm{obs}})$。

**步骤 5：与 IMU 先验合成卡尔曼增益。** 预计算

$$
P_{\mathrm{pose}}=P_{\mathrm{pred}}[0:6,0:6],
\qquad
P_{\mathrm{pred}}H_{\mathrm{sel}}^\top=P_{\mathrm{pred}}[:,0:6].
$$

创新协方差与增益：

$$
S=P_{\mathrm{pose}}+R_{\mathrm{obs}},
\qquad
K=P_{\mathrm{pred}}H_{\mathrm{sel}}^\top\,S^{-1}
\in\mathbb{R}^{21\times 6}.
$$

实现中对 $S$ 做 LDLT，用 $S^{-1}$ 右乘得到 $K$。$K$ 的后 15 行非零，是因为 $P_{\mathrm{pred}}$ 里位姿与速度/偏置/重力的交叉协方差：位姿被 $\mathbf{r}_v$ 拉动时，这些量跟着修正。

**步骤 6：注入名义状态。** 状态增量不是 $H^{-1}$ 再作用一次，而是

$$
\delta\hat{\mathbf{x}}=K\mathbf{r}_v,
\qquad
\begin{aligned}
R_c&\leftarrow R_c\,\mathrm{Exp}(\delta\hat{\boldsymbol{\theta}}),\\
\mathbf{p}_c&\leftarrow\mathbf{p}_c+\delta\hat{\mathbf{p}}_w,\\
\mathbf{v},\mathbf{b}_g,\mathbf{b}_a,\mathbf{g}_w&\leftarrow\text{各加对应增量}.
\end{aligned}
$$

**步骤 7：迭代与 Joseph。** 若 $\|\delta\hat{\boldsymbol{\xi}}_{\mathrm{pose}}\|<10^{-3}$ 则收敛；否则回到步骤 1（最多 $8$ 轮）。全部迭代结束后，用最后一轮的 $K,R_{\mathrm{obs}}$ 做 Joseph：

$$
P^{+}
=
(I-KH_{\mathrm{sel}})P_{\mathrm{pred}}(I-KH_{\mathrm{sel}})^\top
+
K R_{\mathrm{obs}} K^\top.
$$

注意：迭代过程中均值在变，$P_{\mathrm{pred}}$ 不变；只有退出循环后才写 $P^{+}$。

---

### 3.5 ESKF 与 ESIKF 对照（含雅可比）

| 项目 | Stereo-GICP（独立，姊妹篇） | ESKF | ESIKF 几何项 |
|------|---------------------------|------|----------------|
| 残差维数 | $e\in\mathbb{R}^3$ | $z\in\mathbb{R}^6$（位姿创新） | $e\in\mathbb{R}^3$（+ 标量光度） |
| 参数扰动 | $\delta\xi$ 目标系左 SE(3) | $\delta x$（21），观测看前 6 | 同 ESKF 混合前 6 |
| 几何 / 观测 $J$ | $J_{\mathrm{left}}=[-[p']_\times\mid I]$ | $H_{\mathrm{sel}}=[I\mid 0]$ | $J_{\mathrm{geo}}=[-R_{st}[p_{\mathrm{src}}]_\times\mid R_a^{\top}]$ |
| 权重 | $w\,W$，目标 $\sigma^{\mathrm{match}}=0.5\,\mathrm{px}$ | Stereo：$J_{\mathrm{obs}}\Sigma J_{\mathrm{obs}}^{\top}$；small_gicp：$J_{g\to e}\Sigma J_{g\to e}^{\top}$ | 同 $w\,W$ + 光度 $w_c$ |
| 关联 | 相机系投影：像素中心光线 | 无（整姿观测） | 同一 `ProjectiveAssociate`；失败则欧氏 |
| 代码 | `AlignProjective`（`stereo_gicp.cpp`） | `ESKF` + `J_{\mathrm{obs}}$ / `AlignGicpCovariance*` | `BuildProjectiveObservationSubclouds` + `ColorPointObserver` + `ESIKF` |

**一句话：**

- **Stereo-GICP**：目标系左扰动，$J_{\mathrm{left}}=[-[p']_\times\mid I]$，代价 $\rho(e^{\top}C^{-1}e)$（Huber）；$\mathbf{q}$ 在命中像素中心光线上。  
- **ESKF**：位姿观测雅可比是选择矩阵 $H_{\mathrm{sel}}$；Stereo-GICP 左扰动噪声经 $J_{\mathrm{obs}}$、small_gicp 右扰动噪声经 $J_{g\to e}$ 变到 $[\delta\theta_b,\delta p_w]$。  
- **ESIKF**：几何项沿用同一关联、$e$、$C$ 与 Huber，但雅可比必须按机体右扰动 + 世界系平移推导为 $J_{\mathrm{geo}}$，不能照搬 $J_{\mathrm{left}}$。

二者共享混合误差与 $H_{\mathrm{sel}}$，保证松/紧耦合统计一致。

---

### 3.6 符号总表（本篇）

| 符号 | 含义 |
|------|------|
| $\mathbf{x},\delta\mathbf{x},\delta\boldsymbol{\theta},\delta\mathbf{p}_w$ | 名义 / 误差 / 旋转误差 / 平移误差 |
| $H_{\mathrm{sel}},P_{\mathrm{pred}},K,S,R_{\mathrm{obs}},\mathbf{z}$ | 选择矩阵、预测协方差、增益、创新协方差、观测噪声、创新 |
| $J_{g\to e},J_{\mathrm{obs}},\Sigma_{\mathrm{gicp}}$ | 右扰动/左扰动→ESKF 雅可比、配准协方差 |
| $R_{st},\mathbf{t}_{st},\mathbf{p}',\mathbf{q},(b_u,b_v)$ | 相对位姿、变换点、像素中心光线目标点、命中像素 |
| $\sigma_u^{\mathrm{match}},w(s)$ | 目标横向匹配噪声（$0.5\,\mathrm{px}$）、Huber 权重 |
| $J_{\mathrm{geo}},J_{\mathrm{color}},W,\mathbf{e},r_{\mathrm{color}}$ | 几何/光度雅可比、信息、残差 |
| $H_{\mathrm{obs}},\mathbf{b}_{\mathrm{obs}},\mathbf{r}_v,\lambda,\varepsilon$ | 聚合信息、信息向量（同形于 Stereo-GICP 的 $\mathbf{g}$）、虚拟创新（解 $H_{\mathrm{reg}}\mathbf{r}_v=-\mathbf{b}_{\mathrm{obs}}$）、几何权重、正则 |

---

### 3.7 与前两篇的接口

- 静态 target：文档一；
- 投影关联（像素中心光线）、立体 $\Sigma$、匹配噪声、Huber、$J_{\mathrm{left}}$：文档二；
- 本篇完成混合误差下的 $J_{\mathrm{geo}}$、左/右扰动 $R_{\mathrm{obs}}$ 对齐与协方差闭环。

## 4. 实验

主问题：协方差对齐与正确 $J_{\mathrm{geo}}$ 是否改善**一致性**与精度；松/紧耦合分工是否划算。ATE/RPE 只说明均值准不准；本篇单独评融合时，以滤波器算出的 $P$ 是否匹配真实误差为主判据。

### 4.1 控制变量：动态滤波全程固定

动态滤波会改送进配准的静态点，从而改内点、$R_{\mathrm{obs}}$、创新 $z$，最后改 $P^+$ 与 NEES。评 ESKF / ESIKF 时，**动态滤波本身不作为变量**：整段序列、所有融合对照组使用同一套开关与同一套门控参数，中途不切换、不对难片段单独加强。

- 主表建议关闭动态滤波（或全部打开且参数完全相同），避免把「点云被滤干净」记成融合收益。  
- 「接/不接动态滤波」只作为现系统上的**单独一行**消融，不与松/紧耦合对照搅在一起。  
- 3DGS 后端关闭；IMU 噪声与关键帧策略四组共用。

「全程固定」指配置不变，不是掩码每帧相同：同一门控下 $\mathcal{S}$ 仍随光流残差变化。

### 4.2 融合对照组

| 组 | 更新规则 | 用来回答 |
|----|----------|----------|
| IMU-only | 不视觉更新 | 过程模型的 $P$ 是否按真实漂移长大 |
| ESKF-only | 每帧松耦合，关键帧也不走 ESIKF | 整姿观测 + Stereo $J_{\mathrm{obs}}$ / small_gicp $J_{g\to e}$ 是否一致 |
| 仅 KF-ESIKF | 仅关键帧紧耦合，中间帧只 IMU | 点级虚拟观测是否比整姿更校准 |
| 现系统 | 非 KF→ESKF，KF→ESIKF | 分工是否既准又一致 |

外部 baseline（轨迹对比，不替代一致性）：ORB-SLAM3、VINS-Fusion；固定噪声 ESKF；ICP+ESKF；**故意错套 $J_{\mathrm{left}}$ 的 ESIKF**（应更差或更不稳）。

### 4.3 一致性指标（重点）

每帧在混合误差下取（轨迹先 SE(3) 对齐，再按帧算 $e$）

$$
e_{\theta}=\mathrm{Log}(\hat R^{\top}R_{\mathrm{gt}}),
\qquad
e_{p}=p_{\mathrm{gt}}-\hat p,
\qquad
e=\begin{bmatrix}e_{\theta}\\ e_{p}\end{bmatrix}.
$$

| 量 | 定义 | 健康区间 |
|----|------|----------|
| NEES | $\varepsilon=e^{\top}(P^{+})^{-1}e$（需真值） | 6 维均值约 $4$–$9$；长期 $\gg 6$ 过度自信，$\ll 6$ 过于保守 |
| NIS | $\nu=z^{\top}S^{-1}z$，$S=H_{\mathrm{sel}}P^{-}H_{\mathrm{sel}}^{\top}+R_{\mathrm{obs}}$ | 均值约 $4$–$9$（不必真值） |
| $3\sigma$ 覆盖 | $\lvert e_i\rvert\le 3\sqrt{P_{ii}^{+}}$ | 约 $99\%$ 量级 |
| 收缩是否诚实 | $\mathrm{tr}(P^{+})/\mathrm{tr}(P^{-})$ 与 ATE 同向 | $P$ 缩小且误差也缩小 |

只报 $\mathrm{tr}(P)$ 或 `gicp_cov_trace` 不够。ESKF 与 ESIKF 的 NEES/NIS **分色、分表**，不要混成全程一个平均数。

### 4.4 每帧记录

现有 `fusion_csv` 含创新幅度与 `gicp_cov_trace`，不足以评 $P$。补记（ESKF / ESIKF 分列）：

- 更新前/后：$P_{\mathrm{pose}}$ 六个对角、$\mathrm{tr}(P_{\theta\theta})$、$\mathrm{tr}(P_{pp})$、$\lVert P_{\theta p}\rVert_{F}$；  
- $R_{\mathrm{obs}}$、$S$、$z$、NIS；  
- 对齐后的 $e$、旋转/平移分开的 NEES；  
- `method=eskf|esikf`、是否关键帧、内点数、耗时。

建议顺序：先补 NIS 与 $P$ 对角（不必真值）→ 确认更新后 $P$ 收缩合理 → 有 EuRoC GT 再加 NEES 与 $3\sigma$。

### 4.5 数据、消融与主图

1. **数据**：EuRoC 用 MH_01_easy / MH_05_difficult、V1_03、V2_03 拉开难易；TUM VI、KITTI 补 NIS 与 $3\sigma$（KITTI 上 NEES 较弱）。统一分辨率、IMU 频率与标定。  
2. **消融**（每组只改一项）：  
   - **$J_{g\to e}$ / $J_{\mathrm{obs}}$ 开/关**：关则 $R_{\mathrm{obs}}=\Sigma_{\mathrm{gicp}}$，不经相对扰动→混合误差的映射。Stereo 路径关 $J_{\mathrm{obs}}$，small_gicp 路径关 $J_{g\to e}$。应重点看 NIS / NEES 与 $3\sigma$，而不是只看 ATE；按 $\lVert\mathbf{t}_k\rVert$ 分桶，右扰动路径的杠杆项 $R_{wk}[\mathbf{t}_k]_\times$ 在大相对平移上更关键。相机系配准路径同时关 $\mathrm{Ad}_{T_{b\leftarrow c}}$ 作为第二档。  
   - $R_{\mathrm{obs}}$ 改为各向同性（丢掉 GICP 各向异性，但保留坐标变换）。  
   - 更新时把 $P_{\mathrm{pose}}$ 强制对角（丢掉滤波交叉 $P_{\theta p}$，与「不对齐观测噪声」分开）。  
   - 几何-only / 光度-only / 联合；Huber 开/关；iESKF 迭代次数；完整点 $\Sigma$ vs 各向同性；$\sigma^{\mathrm{match}}$ 过小消融。  
   - **IMU–图像时延**（与动态滤波档 D 同一套 $\Delta t$）：创新均值应出现 $\approx(\boldsymbol{\omega},\mathbf{v})\Delta t$ 的系统偏；NIS 涨、NEES 变差。动态门控配置仍固定，避免把过杀点云记成融合不一致。  
3. **调度**：关键帧密度变化时 ESKF-only vs 现系统的精度–耗时曲线。  
4. **主图**：  
   - 误差 vs $3\sigma$ 时间序列（位置三轴、yaw）：融合有效则误差下降且大多仍在包络内；包络远宽于误差为太保守，经常穿出为过度自信；  
   - NEES / NIS 直方图（对照 $\chi^{2}_{6}$），ESKF 与 ESIKF 分开；  
   - $\mathrm{tr}(P^{+})$ 对比 IMU-only：应下降且与 ATE 同向；只缩小 $P$、ATE 不动则协方差被拧坏；  
   - 更新前后 $P_{\mathrm{pose}}$ 热图（关键帧平均）：应能看到 $P_{\theta p}$；强制对角后 NEES 变差即可写入正文。  
5. **读数注意**：$P_{\theta p}$ 非零只说明滤波认为相关存在，好不好看对角消融后的 NEES；动态滤波主表与「+动态」行分开报；另报数值失败次数与实时率。

## 5. 结论

1. 松耦合 ESKF 与紧耦合 ESIKF 共享混合误差与 $H_{\mathrm{sel}}$，才能在同一前端里切换而不破坏协方差。  
2. Stereo-GICP 左扰动需 $J_{\mathrm{obs}}$、small_gicp 右扰动需 $J_{g\to e}$；点级更新必须用 $J_{\mathrm{geo}}$，不能套独立配准的 $J_{\mathrm{left}}$。  
3. ESIKF 与文档二共用像素中心光线关联、$0.5\,\mathrm{px}$ 匹配噪声与 Huber；虚拟观测把点级信息压进标准卡尔曼形式。  
4. 局限：依赖关联与静态假设；关键帧策略影响紧耦合收益上限。  
5. 后续：与后端 3DGS 关键帧协方差传递、自适应 $\lambda$ 几何–光度权重。
