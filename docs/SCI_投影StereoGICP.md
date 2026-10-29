# 基于立体点协方差的投影关联 Stereo-GICP

> 五篇 SCI 之 **Paper 2** · [索引](SCI_五篇论文索引.md) · 本文描述原投影实现。2026-10-28 配准已升级为 [robotic-front main `744ac2d`](../modules/registration/README.md)，独立求解器位于 `modules/registration/src/stereo_gicp.cpp`；`src/stereo_gicp.cpp` 保留 ESIKF 的观测工具。本文下方关于共用关联及旧函数名的描述对应升级前版本，当前实现以模块源码与版本说明为准。

## 建议题目

Projective Stereo-GICP with Full Triangulation Covariance for Uncertainty-Aware Point Cloud Registration

## 1. Introduction

### 1.1 研究背景

双目 SLAM 与视觉里程计中，短时相对运动估计大量依赖点云配准。与激光不同，双目点由视差三角化得到：沿视线的深度噪声通常远大于图像平面内的横向噪声，且深度标准差近似随 $Z^2$ 增长（方差随 $Z^4$）。若配准代价仍把三个坐标轴等权对待，远处点会在视线方向产生很强的虚假拉力，表现为 $t_z$ 被噪声拧动、旋转被远点“撬动”，近处本应更可信的切向结构反而权重不足。

因此，配准需要同时回答两件事：**谁和谁对应**（关联），以及**这条残差有多可信**（度量）。前者在深度不确定大时不能只靠欧氏最近邻；后者必须进入与双目物理一致的协方差。

### 1.2 现有方法问题

1. **关联与噪声物理脱节。** 欧氏最近邻在深度一抖时可能配到邻面，残差方向不再沿视线，后续加权难以挽回。  
2. **度量过于粗糙。** 仅对深度乘经验系数或假设对角 $\Sigma$，忽略 $\mathrm{Cov}(X,Z)$ 等耦合，与完整一阶传播不等价。  
3. **像素量化被漏写。** 深度图只在整数像素上给深度。对应点若放在像素中心光线上，会留下至多 $0.5\,\mathrm{px}$ 的横向残差；横向标准差若远小于这半个像素，量化会被过强约束。许多工作未写清 $J_{uvd}$，也未把这段匹配不确定度放进 $\Sigma_j$。

### 1.3 本文思路

本文聚焦**相对位姿配准本身**（不为滤波器做主贡献）：

1. 由 $(u,v,d)$ 经 $J_{uvd}$ 传播完整 $3\times 3$ 立体协方差 $\Sigma$；  
2. 投影关联：按深度自适应搜索，目标点取命中像素的中心光线；边缘只看 $1\,\mathrm{px}$ 邻域，并用 $3\sigma_Z$ 门控；  
3. Segal 型马氏点到点 $e^\top C^{-1}e$，$C=R\Sigma_i R^\top+\Sigma_j$；目标横向用 $0.5\,\mathrm{px}$ 匹配噪声，外加 Huber；  
4. 左 SE(3) 扰动下 GN/LM，外层重关联。

动态剔除、ESKF 嵌入见姊妹篇；嵌入滤波器时须改用混合误差雅可比，不能照搬 $J_{\mathrm{left}}$。细节见第 3 节。

### 1.4 主要贡献

1. 给出双目 $J_{uvd}$ 全元素及 $\mathrm{Cov}(X,Z)$ 耦合模型，说明为何必须用完整 $\Sigma$。  
2. 投影关联落到命中像素中心光线，并用 $0.5\,\mathrm{px}$ 匹配噪声描述取整：半像素横向残差等于目标项的 $1\sigma$。  
3. 推导左 SE(3) 马氏 GICP 雅可比，并给出 Huber 加权；解释视线降权与切向约束的含义。  
4. 在相对位姿精度、收敛与深度分桶误差上对比各向同性 ICP / 点到面，以及二者加上点协方差后的加权变体。

## 2. 方法说明

### 2.1 立体三角化协方差（在做什么）

对有效视差点设 $\Sigma_{\mathrm{meas}}=\mathrm{diag}(\sigma_u^2,\sigma_v^2,\sigma_d^2)$，三角化后 $\Sigma=J_{uvd}\Sigma_{\mathrm{meas}}J_{uvd}^\top+\varepsilon I$。$\Sigma$ 大特征值大致沿视线，远点更“胖”。源点用三角化噪声 $\sigma_u=\sigma_v=0.5\,\mathrm{px}$、$\sigma_d=0.3\,\mathrm{px}$。目标点在关联像素上现算：深度仍用 $\sigma_d=0.3\,\mathrm{px}$，横向改用匹配噪声 $\sigma_u^{\mathrm{match}}=\sigma_v^{\mathrm{match}}=0.5\,\mathrm{px}$（最近像素取整的最大偏差，当作 $1\sigma$）。$\varepsilon=(0.004\,\mathrm{m})^2$。

### 2.2 投影关联（在做什么）

源点经 $T$ 得 $\mathbf{p}'$，连续投影到 $(u,v)$ 后在目标深度图上搜索。搜索半径随深度变化：

$$
r(Z)=\mathrm{clamp}\!\left(\left\lceil f_x\cdot 0.10\,\mathrm{m}/Z\right\rceil,\,2,\,15\right)\ \mathrm{px}.
$$

其中：

- $r(Z)$：以投影像素为中心的搜索半径（px，切比雪夫距离）；
- $Z$：用来估计半径的深度（m），实现里取 $p'_z$；
- $f_x$：水平焦距（px）；
- $0.10\,\mathrm{m}$：与 GICP 对应距离同量级的空间搜索半径；
- $2,15$：半径的下限与上限（px）。

近处窗口大，远处缩小。边缘跳变只在切比雪夫半径 $1\,\mathrm{px}$ 的邻域里判断；中心像素若已满足 $|p'_z-Z|\le 3\sigma_Z$ 则优先采用，否则在窗口内取与 $p'_z$ 最接近且仍过 $3\sigma_Z$ 门控的深度。量程为 $0.5$–$15\,\mathrm{m}$。

命中像素为整数坐标 $(b_u,b_v)$、深度 $Z_j$ 时，目标点放在该像素的中心光线上，而不是 $\mathbf{p}'$ 自己的连续光线上。投影落在像素中心时残差沿视线；偏离最多半个像素时，横向残差留在 $\mathbf{e}$ 里，并由 $\sigma^{\mathrm{match}}=0.5\,\mathrm{px}$ 告诉优化器这段偏差有多可信。

### 2.3 马氏点到点目标（在做什么）

$\mathbf{e}=\mathbf{p}'-\mathbf{q}_j$，$C=R\Sigma_i R^\top+\Sigma_j$，最小化 $\sum \rho(e^\top C^{-1}e)$。$W=C^{-1}$ 在视线方向变小，削弱靠视差噪声拧 $t_z$。切向仍较紧：半像素横向残差等于目标项的 $1\sigma$；再叠上源点的 $0.5\,\mathrm{px}$ 后，相对完整 $C$ 大约是 $0.7\sigma$。白化残差过大时用 Huber（$\delta=1.345$）降权。

### 2.4 左 SE(3) 优化（在做什么）

内层固定关联，$T\leftarrow\exp(\delta\xi)T$，$J_{\mathrm{left}}=[-[p']_\times\mid I]$。法方程按 Huber 权重累加，阻尼为 $10^{-6}\max(1,\mathrm{tr}(H)/6)$。$\|\delta\boldsymbol{\omega}\|$ 与 $\|\delta\mathbf{t}\|$ 都小于 $10^{-4}$ 时停止，否则外层重关联。默认最多 $20$ 轮，源点先以 $0.05\,\mathrm{m}$ 体素下采样，有效对应少于 $20$ 则本轮放弃。此为独立配准器 `AlignProjective`。

### 2.5 在系统中的位置

- 相对 $T$ 与 $\Sigma_{\mathrm{gicp}}$ → 松耦合 ESKF（需 $J_{g\to e}$，姊妹篇 3）；  
- 点级 $(e,C)$ → ESIKF 重导 $J_{\mathrm{geo}}$（勿直接用 $J_{\mathrm{left}}$）；  
- target 建议先经动态滤波静态索引（姊妹篇 1）。

## 3. 理论推导

每个显示公式后给出符号含义；左 SE(3) 雅可比细导与实现对照见 §3.5。姊妹篇：[动态观测滤波](SCI_动态观测滤波.md)、[ESKF 与 ESIKF](SCI_ESKF与ESIKF.md)。

### 3.1 问题与假设

#### 问题表述

给定源点云 $\{\mathbf{p}_i,\Sigma_i\}$ 与目标几何（深度图或点云 $\{\mathbf{q}_j,\Sigma_j\}$），估计短时相对位姿 $T=(R,t)$。经典各向同性点到点 ICP 忽视双目深度噪声远大于横向噪声；欧氏最近邻在深度不确定大时易配到邻面。

#### 方法要点

1. **投影关联**：回答「谁与谁配对」——命中像素的中心光线；
2. **马氏点到点代价**：回答「残差多可信」——$\rho(e^\top C^{-1}e)$，$C=R\Sigma_i R^\top+\Sigma_j$，目标横向为 $0.5\,\mathrm{px}$ 匹配噪声；
3. **独立优化**用左 SE(3) 雅可比 $J_{\mathrm{left}}$；嵌入滤波器时改用混合误差雅可比（见文档三），几何项同样乘 Huber 权重。

---

### 3.2 立体三角化与协方差传播（逐步）

#### 测量与三角化

左目像素 $(u,v)$、视差 $d$、基线 $B$：

$$
Z=\frac{f_x B}{d},\qquad
X=\frac{(u-c_x)Z}{f_x},\qquad
Y=\frac{(v-c_y)Z}{f_y}.
$$

其中：

- $(u,v)$：左目像素（px）；
- $d$：视差（px），$d>0$；
- $B$：基线（m）；
- $f_x,f_y,c_x,c_y$：内参；
- $(X,Y,Z)$：左目系三维点（m）。

#### 测量噪声

假设独立：

$$
\Sigma_{\mathrm{meas}}
=
\mathrm{diag}(\sigma_u^2,\sigma_v^2,\sigma_d^2).
$$

其中 $\sigma_u,\sigma_v,\sigma_d$ 为 $u,v,d$ 的标准差（px）。

#### 三角化雅可比 $J_{uvd}$

深度对视差：

$$
\frac{\partial Z}{\partial d}
=
-\frac{f_x B}{d^2}
=
-\frac{Z^2}{f_x B},
\qquad
\frac{\partial Z}{\partial u}
=
\frac{\partial Z}{\partial v}
=
0.
$$

其中第一式由 $Z=f_x B/d$ 直接微分；第二式表明深度不显含像素坐标。

对 $X=(u-c_x)Z/f_x$ 用乘积法则：

$$
\frac{\partial X}{\partial u}
=
\frac{Z}{f_x},
\qquad
\frac{\partial X}{\partial v}=0,
\qquad
\frac{\partial X}{\partial d}
=
\frac{u-c_x}{f_x}\frac{\partial Z}{\partial d}
=
\frac{X}{Z}\frac{\partial Z}{\partial d}.
$$

同理

$$
\frac{\partial Y}{\partial v}
=
\frac{Z}{f_y},
\qquad
\frac{\partial Y}{\partial d}
=
\frac{Y}{Z}\frac{\partial Z}{\partial d},
\qquad
\frac{\partial Y}{\partial u}=0.
$$

组装（列依次对应 $\partial/\partial u,\partial/\partial v,\partial/\partial d$）：

$$
J_{uvd}
=
\begin{pmatrix}
Z/f_x & 0 & (X/Z)\,\partial Z/\partial d \\
0 & Z/f_y & (Y/Z)\,\partial Z/\partial d \\
0 & 0 & \partial Z/\partial d
\end{pmatrix}.
$$

其中 $J_{uvd}=\partial(X,Y,Z)/\partial(u,v,d)\in\mathbb{R}^{3\times 3}$。

#### 一阶协方差传播

$$
\Sigma
=
J_{uvd}\,\Sigma_{\mathrm{meas}}\,J_{uvd}^\top
+\varepsilon I_3.
$$

其中：

- $\Sigma\in\mathbb{R}^{3\times 3}$：点位置协方差（$\mathrm{m}^2$）；
- $\varepsilon>0$：数值正则，保证正定；
- $I_3$：单位阵。

#### 尺度与耦合（为何必须用完整 $3\times 3$）

由 $\partial Z/\partial d\propto -Z^2$：

$$
\sigma_Z
\sim
\frac{Z^2}{f_x B}\,\sigma_d
\quad\Rightarrow\quad
\mathrm{Var}(Z)\sim Z^4.
$$

横向：

$$
\sigma_X\sim\frac{Z}{f_x}\sigma_u,\qquad
\sigma_Y\sim\frac{Z}{f_y}\sigma_v.
$$

交叉项（不可忽略）：

$$
\mathrm{Cov}(X,Z)
=
\frac{X}{Z}\Bigl(\frac{\partial Z}{\partial d}\Bigr)^2\sigma_d^2,
\qquad
\mathrm{Cov}(Y,Z)
=
\frac{Y}{Z}\Bigl(\frac{\partial Z}{\partial d}\Bigr)^2\sigma_d^2.
$$

因此 $\Sigma$ 主轴近似沿视线，且非对角；仅对 $e_z$ 降权不能等价于完整马氏度量。

源点记 $\Sigma_i$（三角化噪声）。目标点记 $\Sigma_j$：同一套 $J_{uvd}$，但 $\sigma_u,\sigma_v$ 换成匹配噪声 $0.5\,\mathrm{px}$，$\sigma_d$ 仍为 $0.3\,\mathrm{px}$。

---

### 3.3 投影关联（逐步）

#### 变换与投影

当前估计 $T=(R,t)$，源点 $\mathbf{p}_i$：

$$
\mathbf{p}'=R\mathbf{p}_i+t=(p'_x,p'_y,p'_z)^\top,
\qquad
(u,v)=\pi(\mathbf{p}').
$$

其中：

- $T=(R,t)$：当前相对位姿，把源点变到目标左目系；
- $\mathbf{p}_i$：源点在源相机系的坐标（m）；
- $\mathbf{p}'$：变到目标左目系后的源点（m）；
- $\pi$：针孔投影，$(u,v)$ 为连续像素；
- $(u_i,v_i)=(\mathrm{round}(u),\mathrm{round}(v))$：搜索窗口中心。

#### 自适应搜索与门控

搜索半径（像素，切比雪夫）：

$$
r(Z)=\mathrm{clamp}\!\left(\left\lceil f_x \cdot 0.10 / Z\right\rceil,\,2,\,15\right).
$$

其中：

- $r(Z)$：搜索半径（px）；$Z$ 小则窗口大；
- $f_x$：水平焦距（px）；
- $0.10$：空间搜索半径（m），与 GICP 对应距离同量级；
- $Z$：此处取 $p'_z$（m）；
- $2$ 与 $15$：夹紧后的最小、最大半径（px）。

在该窗口内读取深度，并要求：

1. 像素在像面内，$Z$ 落在 $[0.5,15]\,\mathrm{m}$；
2. **边缘**：只在半径 $1\,\mathrm{px}$ 的邻域里统计 $Z_{\max}-Z_{\min}$。邻域多于一个有效深度且跨度大于 $3\sigma_Z(Z_{\mathrm{ref}})$ 则丢弃（遮挡或深度跳变）。大搜索窗里的另一层深度不参与这次判断；
3. **中心优先**：中心像素深度若满足 $|p'_z-Z_{\mathrm{center}}|\le 3\sigma_Z(Z_{\mathrm{center}})$，就用它；
4. 否则在窗口中取 $|Z-p'_z|$ 最小、且 $|p'_z-Z_j|\le 3\sigma_Z(Z_j)$ 的像素。$\sigma_Z$ 用立体模型与 $\sigma_d=0.3\,\mathrm{px}$ 在该深度处计算。

#### 沿命中像素中心光线反投影

命中整数像素 $(b_u,b_v)$、深度 $Z_j$：

$$
\mathbf{q}_j
=
\begin{pmatrix}
\dfrac{b_u-c_x}{f_x}Z_j \\[4pt]
\dfrac{b_v-c_y}{f_y}Z_j \\[4pt]
Z_j
\end{pmatrix}.
$$

其中：

- $(b_u,b_v)$：最终采用的整数像素（px）；
- $Z_j$：该像素上的深度（m）；
- $c_x,c_y,f_x,f_y$：主点与焦距（px）；
- $\mathbf{q}_j$：目标左目系中的对应点（m），位于该像素的中心光线上。

内层迭代固定关联时 $\mathbf{q}_j$ 不变。连续投影与命中像素之差记 $\delta u=u-b_u$、$\delta v=v-b_v$（px）。选中最近像素时 $|\delta u|,|\delta v|\le 0.5$。

#### 残差里的视线项与横向项

$$
\mathbf{e}=\mathbf{p}'-\mathbf{q}_j.
$$

其中 $\mathbf{e}$ 为点到点残差（m）；$\mathbf{p}'$ 为变到目标系的源点；$\mathbf{q}_j$ 为像素中心光线上的对应点。

投影正好落在像素中心（$\delta u=\delta v=0$）时 $\mathbf{q}_j\parallel\mathbf{p}'$，残差沿视线：

$$
\mathbf{e}=(p'_z-Z_j)\,\hat{\mathbf{r}}.
$$

其中 $p'_z$、$Z_j$ 分别为源点与对应点的深度（m）；$\hat{\mathbf{r}}=\mathbf{p}'/\|\mathbf{p}'\|$ 为单位视线。

偏离像素中心时，横向大约是

$$
e_x\approx \delta u\cdot\frac{Z_j}{f_x},\qquad
e_y\approx \delta v\cdot\frac{Z_j}{f_y}.
$$

其中：

- $e_x,e_y$：残差的相机系横向分量（m）；
- $\delta u,\delta v$：连续投影相对命中像素的偏差（px）；
- $Z_j/f_x$、$Z_j/f_y$：该深度处 $1\,\mathrm{px}$ 对应的横向长度（m/px）。

目标协方差的横向标准差来自同一比例：

$$
\sigma_{X,j}=\frac{Z_j}{f_x}\,\sigma_u^{\mathrm{match}}.
$$

其中 $\sigma_{X,j}$ 只来自 $\Sigma_j$（目标项，m）；$\sigma_u^{\mathrm{match}}=0.5\,\mathrm{px}$ 为匹配噪声标准差。取 $|\delta u|=0.5$ 时

$$
\frac{|e_x|}{\sigma_{X,j}}
=
\frac{0.5}{\sigma_u^{\mathrm{match}}}
=
1.
$$

其中分子 $0.5$ 是最近像素取整的最大偏差（px）。该比值等于 $1$ 表示：这段横向残差与 $\Sigma_j$ 声明的横向标准差一样大，即目标项的 $1\sigma$。只看目标项时，对应的马氏贡献约为 $(e_x/\sigma_{X,j})^2=1$。

源点横向标准差另有 $\sigma_{u,i}=0.5\,\mathrm{px}$。两项独立，完整残差协方差 $C=R\Sigma_i R^\top+\Sigma_j$ 的横向标准差为

$$
\sigma_{X,C}
=
\frac{Z_j}{f_x}\sqrt{\sigma_{u,i}^2+(\sigma_u^{\mathrm{match}})^2}
\approx
\frac{Z_j}{f_x}\cdot 0.71.
$$

其中 $\sigma_{X,C}$ 是 $C$ 的横向标准差（m），已经同时包含源点三角化噪声和目标点匹配噪声。同一段半像素残差相对完整 $C$ 为

$$
\frac{|e_x|}{\sigma_{X,C}}
\approx
\frac{0.5}{0.71}
\approx
0.7.
$$

其中 $0.7$ 是 $|e_x|$ 与 $\sigma_{X,C}$ 的比值，单位是“$C$ 的标准差”。马氏贡献约为 $0.7^2\approx 0.5$，小于只看目标项时的 $1$。视线方向的 $\sigma_Z\propto Z^2$ 通常仍远大于这半个像素，马氏度量继续压低视差噪声对 $t_z$ 的拉力。

最近像素的取整误差落在 $[-0.5,0.5]\,\mathrm{px}$。把它当作均匀分布时标准差约为 $1/\sqrt{12}\approx 0.29\,\mathrm{px}$。实现把最大偏差 $0.5\,\mathrm{px}$ 用作 $1\sigma$，比均匀模型保守，又不会把半像素残差放到 $1\,\mathrm{px}$ 那么松。搜索窗选到邻像素时偏差可以到若干像素，超出这段量化模型，交给下一节的 Huber。

---

### 3.4 马氏目标函数

#### 残差与残差协方差

$$
\mathbf{e}=\mathbf{p}'-\mathbf{q}_j,
\qquad
C=R\Sigma_i R^\top+\Sigma_j,
\qquad
W=C^{-1}.
$$

其中：

- $\mathbf{e}\in\mathbb{R}^3$：点到点残差（m）；
- $C$：残差协方差（源旋到目标系后与目标相加）；
- $W$：信息矩阵。

推导：$R\mathbf{p}_i$ 的协方差为 $R\Sigma_i R^\top$（仿射变换）；$\mathbf{q}_j$ 独立则方差相加。$\Sigma_i$ 的 $\sigma_u,\sigma_v,\sigma_d$ 为 $0.5,0.5,0.3\,\mathrm{px}$；$\Sigma_j$ 的横向改为匹配噪声 $0.5\,\mathrm{px}$，视差噪声仍是 $0.3\,\mathrm{px}$。

#### 目标

$$
\min_{T}
E(T)
=
\sum_{(i,j)\in\mathcal{A}}
\rho\!\left(\mathbf{e}_{ij}^\top W_{ij}\mathbf{e}_{ij}\right).
$$

核内 $\rho(s)=s$，因此小残差仍是马氏平方。记 $s=\mathbf{e}^\top W\mathbf{e}$。Huber 阈值 $\delta=1.345$，核内权重为 $1$，核外为 $\delta/\sqrt{s}$：

$$
w(s)=
\begin{cases}
1, & s\le \delta^2,\\
\delta/\sqrt{s}, & s>\delta^2,
\end{cases}
\qquad
\rho(s)=
\begin{cases}
s, & s\le \delta^2,\\
2\delta\sqrt{s}-\delta^2, & s>\delta^2.
\end{cases}
$$

其中：

- $s=\mathbf{e}^\top W\mathbf{e}$：白化后的马氏平方，无量纲；
- $\delta=1.345$：Huber 阈值，作用在 $\sqrt{s}$ 上；$\delta^2$ 是核内、核外的分界；
- $w(s)$：乘到该点 $J^\top W J$ 与 $J^\top W\mathbf{e}$ 上的权重；核内为 $1$，核外为 $\delta/\sqrt{s}$；
- $\rho(s)$：同一尺度下记录的鲁棒代价；核内等于 $s$，核外为 $2\delta\sqrt{s}-\delta^2$。

$s$ 非有限或为负时该点权重为 $0$。邻像素误配使白化残差变大，从而被降权；半像素以内的取整残差通常仍在核内。

---

### 3.5 左 SE(3) 扰动与雅可比（逐步）

本节用于**独立相对位姿优化**（`AlignProjective`）。嵌入 ESKF/ESIKF 时见文档三；ESIKF 的几何项使用同一 $e$、$C$ 与 Huber 权重，雅可比改为 $J_{\mathrm{geo}}$。

#### 左扰动定义

$$
T\leftarrow\exp(\delta\boldsymbol{\xi})\,T,
\qquad
\delta\boldsymbol{\xi}
=
\begin{bmatrix}\delta\boldsymbol{\omega}\\\delta\mathbf{t}\end{bmatrix}
\in\mathbb{R}^6.
$$

其中：

- $\delta\boldsymbol{\omega}$：目标系左切旋转增量（rad）；
- $\delta\mathbf{t}$：目标系平移增量（m）；
- $\exp$：$\mathrm{SE}(3)$ 指数映射。

#### 对已变换点的作用

$$
\mathbf{p}'^{+}
=
\exp(\delta\boldsymbol{\xi})\,\mathbf{p}'
\approx
\mathbf{p}'+\delta\boldsymbol{\omega}\times\mathbf{p}'+\delta\mathbf{t}.
$$

内层迭代固定关联时，$\mathbf{q}_j$ 对 $\delta\boldsymbol{\xi}$ 一阶贡献为 0，故

$$
\mathrm{d}\mathbf{e}
=
\mathrm{d}\mathbf{p}'
=
\delta\boldsymbol{\omega}\times\mathbf{p}'+\delta\mathbf{t}
=
-[\mathbf{p}']_\times\delta\boldsymbol{\omega}
+\delta\mathbf{t}.
$$

反对称矩阵定义：

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

工程中独立配准的反对称矩阵在 `src/stereo_gicp.cpp` 内构造；ESIKF 使用 `svio_utils::SkewSymmetric`。

#### 矩阵形式（左扰动雅可比）

固定对应时 $q_j$ 对 $\delta\boldsymbol{\xi}$ 一阶贡献为 0，故 $\mathrm{d}e=\mathrm{d}p'$：

$$
\boxed{
J_{\mathrm{left}}
=
\begin{bmatrix}
-[\mathbf{p}']_\times & I_3
\end{bmatrix}
=
\begin{bmatrix}
0 & p'_z & -p'_y & 1 & 0 & 0 \\
-p'_z & 0 & p'_x & 0 & 1 & 0 \\
p'_y & -p'_x & 0 & 0 & 0 & 1
\end{bmatrix},
\qquad
\mathrm{d}\mathbf{e}=J_{\mathrm{left}}\delta\boldsymbol{\xi}.
}
$$

其中：

- $J_{\mathrm{left}}\in\mathbb{R}^{3\times 6}$：残差对左扰动的雅可比；
- 前三列对应 $\delta\boldsymbol{\omega}$，后三列对应 $\delta\mathbf{t}$；
- $\delta\boldsymbol{\omega}$ 在**目标系左切**，与 ESKF 的机体右切 $\delta\boldsymbol{\theta}$ **不能直接互换**。

#### 实现对照（本篇雅可比）

| 公式量 | 代码 |
|--------|------|
| 搜索半径 $r(Z)$、Huber $w(s)$ | `SearchRadiusPx`、`HuberWeight`（`include/stereo_gicp.h`） |
| 点协方差 $\Sigma$ | `PointCovarianceFromXyz` |
| 投影关联 $\mathbf{q}_j$ | `ProjectiveAssociate` |
| $J_{\mathrm{left}}$、残差 $\mathbf{e}$、$H,\mathbf{g}$ | `AlignProjective`（`src/stereo_gicp.cpp`） |
| ESIKF 关联 | `StateEstimator::BuildProjectiveObservationSubclouds` 调用 `ProjectiveAssociate` |
| ESIKF 几何 Huber | `ColorPointObserver` 中的 `HuberWeight` |

#### 建立法方程与解算（逐步）

以下对应 `AlignProjective` 的一轮内层迭代。外层先固定当前 $T=(R,t)$ 做投影关联，得到对应集 $\mathcal{A}$；对应数少于 $20$ 则本轮放弃。

符号约定（与代码一致）：

| 符号 | 含义 | 代码 |
|------|------|------|
| $\mathbf{e}\in\mathbb{R}^3$ | 点残差 $\mathbf{p}'-\mathbf{q}$ | `e` |
| $\mathbf{g}\in\mathbb{R}^6$ | 梯度信息 $\sum w J^\top W\mathbf{e}$ | `g` |
| $H\in\mathbb{R}^{6\times 6}$ | 近似 Hessian $\sum w J^\top W J$ | `H` |
| $\delta\boldsymbol{\xi}\in\mathbb{R}^6$ | 左扰动增量 | `dx` |

正规方程（本模块唯一采用的求解形式）是

$$
\boxed{H_{\mathrm{reg}}\,\delta\boldsymbol{\xi}=-\mathbf{g}.}
$$

残差 $\mathbf{e}\in\mathbb{R}^3$ 只进入 $\mathbf{g}=\sum w\,J^\top W\mathbf{e}$；右端是 $-\mathbf{g}$，不是 $\mathbf{e}$。

---

**步骤 0：从代价到正规方程（推导）。** 固定关联后，核内（$w=1$）单点代价为 $s=\mathbf{e}^\top W\mathbf{e}$。左扰动线性化

$$
\mathbf{e}(\delta\boldsymbol{\xi})\approx\mathbf{e}_0+J\delta\boldsymbol{\xi},\qquad J=J_{\mathrm{left}}.
$$

代入得

$$
\begin{aligned}
s(\delta\boldsymbol{\xi})
&\approx
(\mathbf{e}_0+J\delta\boldsymbol{\xi})^\top W(\mathbf{e}_0+J\delta\boldsymbol{\xi})
\\
&=
\mathbf{e}_0^\top W\mathbf{e}_0
+2\delta\boldsymbol{\xi}^\top(J^\top W\mathbf{e}_0)
+\delta\boldsymbol{\xi}^\top(J^\top W J)\delta\boldsymbol{\xi}.
\end{aligned}
$$

对 $\delta\boldsymbol{\xi}$ 求导并令为零：

$$
\frac{\partial s}{\partial\delta\boldsymbol{\xi}}
=
2\,J^\top W\mathbf{e}_0
+2\,J^\top W J\,\delta\boldsymbol{\xi}
=\mathbf{0}
\quad\Rightarrow\quad
(J^\top W J)\,\delta\boldsymbol{\xi}
=
-(J^\top W\mathbf{e}_0).
$$

Huber 核外用权重 $w$ 缩放该点的信息与梯度（IRLS）：把上式两边同乘 $w$，再对所有点求和，即得

$$
H\delta\boldsymbol{\xi}=-\mathbf{g},
\qquad
H=\sum_i w_i J_i^\top W_i J_i,
\qquad
\mathbf{g}=\sum_i w_i J_i^\top W_i\mathbf{e}_i.
$$

二次型写法：局部近似 $E\approx\mathrm{const}+\mathbf{g}^\top\delta\boldsymbol{\xi}+\tfrac12\delta\boldsymbol{\xi}^\top H\delta\boldsymbol{\xi}$，驻点满足 $\nabla E=H\delta\boldsymbol{\xi}+\mathbf{g}=\mathbf{0}$，故右端必须是 $-\mathbf{g}$。

---

**步骤 1：单点残差与权重。** 对 $(i,j)\in\mathcal{A}$：

$$
\mathbf{p}'=R\mathbf{p}_i+\mathbf{t},
\qquad
\mathbf{e}=\mathbf{p}'-\mathbf{q}_j,
\qquad
C=R\Sigma_i R^\top+\Sigma_j.
$$

用 LDLT 解 $C X=I$ 得 $W=C^{-1}$（不显式求逆）。再

$$
s=\mathbf{e}^\top W\mathbf{e},
\qquad
w=w(s)
=
\begin{cases}
1, & s\le\delta^2,\\
\delta/\sqrt{s}, & s>\delta^2,
\end{cases}
\quad(\delta=1.345).
$$

其中 $\Sigma_i$ 用三角化噪声 $0.5,0.5,0.3\,\mathrm{px}$；$\Sigma_j$ 用匹配噪声 $0.5,0.5\,\mathrm{px}$ 与 $\sigma_d=0.3\,\mathrm{px}$。$C$ 分解失败、$W$ 非有限、$s$ 非法或 $w\le 0$ 时跳过该点。

同时记录鲁棒代价（只用于日志，不进法方程）：

$$
\rho(s)=
\begin{cases}
s, & s\le\delta^2,\\
2\delta\sqrt{s}-\delta^2, & s>\delta^2.
\end{cases}
$$

---

**步骤 2：单点雅可比与贡献块。**

$$
J
=
J_{\mathrm{left}}
=
\begin{bmatrix}-[\mathbf{p}']_\times & I_3\end{bmatrix}
\in\mathbb{R}^{3\times 6}.
$$

该点贡献

$$
\Delta H_i=w\,J^\top W J\in\mathbb{R}^{6\times 6},
\qquad
\Delta\mathbf{g}_i=w\,J^\top W\mathbf{e}\in\mathbb{R}^6.
$$

展开含义：

- $W\mathbf{e}\in\mathbb{R}^3$：白化后的三维拉力；
- $J^\top(W\mathbf{e})\in\mathbb{R}^6$：经 $J^\top$ 分配到 $(\delta\boldsymbol{\omega},\delta\mathbf{t})$——前三维靠 $-[\mathbf{p}']_\times^\top$ 把拉力变成力矩，后三维原样进平移；
- $J^\top W J$：6 个增量之间的信息耦合（例如绕某轴转与沿某轴移的交叉项）。

---

**步骤 3：累加全部内点。**

$$
H=\sum_{i}\Delta H_i,
\qquad
\mathbf{g}=\sum_{i}\Delta\mathbf{g}_i,
\qquad
E=\sum_{i}\rho(s_i).
$$

实现：OpenMP 下每线程各累一份再 `critical` 归并。归并后

$$
H\leftarrow\tfrac12(H+H^\top).
$$

$H$ 或 $\mathbf{g}$ 含非有限值则中止本轮。

---

**步骤 4：写成待解线性方程。** 加阻尼（固定比例 LM，不做阻尼线搜索）：

$$
\lambda=10^{-6}\max\bigl(1,\mathrm{tr}(H)/6\bigr),
\qquad
H_{\mathrm{reg}}=H+\lambda I_6.
$$

待解方程：

$$
H_{\mathrm{reg}}\,\delta\boldsymbol{\xi}=-\mathbf{g}.
$$

$\mathbf{g}$ 是代价对 $\delta\boldsymbol{\xi}$ 的梯度方向，故增量取负号。

---

**步骤 5：LDLT 解算。**

$$
H_{\mathrm{reg}}=L\,D\,L^\top
\quad\Rightarrow\quad
\delta\boldsymbol{\xi}=-H_{\mathrm{reg}}^{-1}\mathbf{g}.
$$

代码：`dx = -H_ldlt.solve(g)`。分解失败或 `dx` 非有限则 `break`。

$$
\delta\boldsymbol{\xi}
=
\begin{bmatrix}\delta\boldsymbol{\omega}\\ \delta\mathbf{t}\end{bmatrix},
\quad
\delta\boldsymbol{\omega}\in\mathbb{R}^3\ (\mathrm{rad}),\ 
\delta\mathbf{t}\in\mathbb{R}^3\ (\mathrm{m}),
$$

均在**目标相机系**左切空间。

---

**步骤 6：左乘更新 $T$。**

$$
R\leftarrow\mathrm{Exp}(\delta\boldsymbol{\omega})\,R,
\qquad
\mathbf{t}\leftarrow\mathrm{Exp}(\delta\boldsymbol{\omega})\,\mathbf{t}+\delta\mathbf{t}.
$$

等价于 $T\leftarrow\exp(\delta\boldsymbol{\xi})\,T$：小运动加在目标系，乘在当前 $T$ 左边。

---

**步骤 7：收敛与外层重关联。** 若

$$
\|\delta\boldsymbol{\omega}\|<10^{-4}
\quad\text{且}\quad
\|\delta\mathbf{t}\|<10^{-4},
$$

则 `converged=true` 并停止。否则用新 $T$ **重新投影关联**，回到步骤 1。默认最多 $20$ 轮外层。

---

**步骤 8：输出位姿协方差。** 用最后一轮**未加阻尼**的 $H$（不是 $H_{\mathrm{reg}}$）：

$$
\Sigma_{\mathrm{gicp}}=H^{-1}
$$

（LDLT 求逆后对称化）。该 $\Sigma$ 与 $\delta\boldsymbol{\xi}$ 同坐标（目标系左扰动）。映到 body 用 `CameraPoseCovToBody`；送 ESKF 前再乘 $J_{\mathrm{obs}}$（见文档三）。

#### 信息几何含义（$t_z$ 弱约束）

设视线方向为 $\hat{\mathbf{r}}$。$C$ 在 $\hat{\mathbf{r}}$ 上特征值大 $\Rightarrow$ $W$ 在该方向特征值小 $\Rightarrow$ 沿视线的 $\mathbf{e}$ 对 $E$ 贡献小 $\Rightarrow$ **不靠视差噪声拧紧 $t_z$**。切向结构对应大信息，约束旋转与横向平移。

---

### 3.6 与滤波器几何项的关系（预告）

独立优化使用 $J_{\mathrm{left}}$（对 $\mathbf{p}'$ 左扰动）。系统滤波器采用

$$
R\leftarrow R\,\mathrm{Exp}(\delta\boldsymbol{\theta}),
\qquad
\mathbf{p}\leftarrow\mathbf{p}+\delta\mathbf{p}_w,
$$

此时几何雅可比为

$$
J_{\mathrm{geo}}
=
\begin{bmatrix}
-R_{st}[\mathbf{p}_{\mathrm{src}}]_\times & R_a^\top
\end{bmatrix}.
$$

当存在平移 $t_{st}\neq\mathbf{0}$ 时，$J_{\mathrm{geo}}\neq J_{\mathrm{left}}$（在相同残差 $\mathbf{e}=\mathbf{p}'-\mathbf{q}$ 下）。完整证明见 [§3.4](SCI_ESKF与ESIKF.md)。

---

### 3.7 算法流程

1. 源点做 $0.05\,\mathrm{m}$ 体素下采样，并按 §3.2 准备 $\Sigma_i$；目标有静态点云时只把它溅射成深度图，否则用给定深度图；
2. 外层循环：由当前 $T$ 做 §3.3 的投影关联，得到 $\mathcal{A}$（$|\mathcal{A}|<20$ 则停止）；
3. 内层按 §3.5「建立法方程与解算」：逐点算 $\mathbf{e},W,w,J_{\mathrm{left}}$ → 累加 $H,\mathbf{g}$ → LDLT 解 $H_{\mathrm{reg}}\delta\boldsymbol{\xi}=-\mathbf{g}$ → 左乘更新 $T$；
4. 未收敛则回到步骤 2 重关联；收敛后输出 $T$ 与 $\Sigma_{\mathrm{gicp}}\approx H^{-1}$。

动态场景下目标点集应先经 [文档一](SCI_动态观测滤波.md) 静态筛选。

---

### 3.8 符号总表（本篇）

| 符号 | 含义 |
|------|------|
| $(u,v,d),B,J_{uvd},\Sigma$ | 测量、基线、三角化雅可比、点协方差 |
| $\sigma_u^{\mathrm{match}},\sigma_{X,j},\sigma_{X,C}$ | 目标横向匹配噪声（$0.5\,\mathrm{px}$）、只含 $\Sigma_j$ 的横向标准差、完整 $C$ 的横向标准差 |
| $r(Z),\delta$ | 自适应搜索半径、Huber 阈值（$1.345$） |
| $\mathbf{p}',\mathbf{q}_j,(b_u,b_v),\mathbf{e},C,W$ | 变换点、像素中心光线对应、命中像素、残差、协方差、信息 |
| $w(s),\rho(s)$ | Huber 权重与鲁棒代价 |
| $\delta\boldsymbol{\xi},J_{\mathrm{left}},H,\mathbf{g}$ | 左扰动、雅可比、Hessian、梯度信息（$\mathbf{g}=\sum w J^\top W\mathbf{e}$；解 $H\delta=-\mathbf{g}$） |
| $\mathcal{A},E,\Sigma_{\mathrm{gicp}}$ | 关联集、鲁棒总代价、位姿协方差 $H^{-1}$ |

---

### 3.9 与相邻文档的接口

- 输入静态 target：文档一；
- 输出相对位姿给松耦合 ESKF，或点级 $J_{\mathrm{geo}},W,\mathbf{e}$ 嵌入 ESIKF：文档三。

## 4. 实验

主问题三句：**谁和谁配对**对不对；**完整 $\Sigma$** 有没有压住远点视线噪声；相对位姿（尤其 $t_z$）是否因此更稳。本篇评**独立配准器**，不评 ESKF/NEES（姊妹篇 3），也不把动态滤波当变量（姊妹篇 1：全程固定同一套开关）。ATE 只作为可选下游一行。

机械臂给相对位姿真值 $T_{c\leftarrow k}^{\mathrm{arm}}$ 和可控深度分布；合成双目隔离 $J_{uvd}$；公开集用来和别人比。

### 4.1 一览

| 实验 | 做哪些 | 考察什么 | 期望结论 |
|------|--------|----------|----------|
| **S 合成** | 平面/多平面深度图；已知 $T$；只在 $(u,v,d)$ 上加已知噪声 | $J_{uvd}$ 与马氏代价是否按模型工作 | 完整 $\Sigma$ 的 $t_z$ 误差不随远点噪声线性恶化；对角/各向同性会。不上公开集主表。 |
| **G1 关联** | 同一点云、同一初值；投影 vs 欧氏 NN；像素中心光线+$0.5\,\mathrm{px}$ 匹配噪声 vs 连续光线（横向残差被消掉）vs 像素中心但 $\sigma^{\mathrm{match}}$ 过小 | 配对是否落到正确像素、半像素残差有没有被过强约束 | 欧氏 NN 在远点/邻面处配错，后续加权救不回。$\sigma^{\mathrm{match}}\ll 0.5\,\mathrm{px}$ 时横向 RPE 变差；连续光线则丢掉切向的亚像素信息。 |
| **G2 度量** | 关联固定为：相机系自适应半径投影，命中像素中心光线反投影，$3\sigma_Z$/$1\,\mathrm{px}$ 边缘门控；对照：Iso-P2P、P2Plane、**Cov-P2P**（完整 $C$ 马氏点到点）、**Cov-P2Plane**（法向标量 $\sigma_n^2=n^\top C n$）、Diag-GICP、完整 $\Sigma$+Huber | 把点协方差写进 P2P/P2Plane 后能否逼近完整方法；$C$ 与 Huber 是否仍必要 | Cov-P2P 相对 Iso-P2P 在 far/$t_z$ 上明显改善，但仍低于完整方法（缺 Huber；与 Diag 比则完整 $C$ 更稳）。Cov-P2Plane 相对 P2Plane 有增益，但一维法向投影吃不满视线椭球，弱平面/远噪下旋转仍易拧。对角 $\Sigma$ 不能等价 $\mathrm{Cov}(X,Z)$。Huber 主要降邻像素误配失败率。 |
| **G3 分桶** | 按目标深度 near / mid / far 分别报 RPE（旋转、$t_{xy}$、$t_z$） | 优势是不是出在远点 | 近处各方法接近；远处完整 $\Sigma$+投影明显稳。这是相对各向同性的主图，不是分轴 ATE。 |
| **G4 初值** | 在臂真值或 IMU 初值上加 $\delta\boldsymbol{\xi}$（平移 $1$–$10\,\mathrm{cm}$，旋转 $0.5^\circ$–$3^\circ$） | 外层重关联的收敛盆 | 投影关联对沿视线的初值误差更耐；欧氏 NN 对横向错位更脆。报迭代次数与失败率。 |
| **G5 臂上真值** | 全静态眼在手上；短时帧对；$T$ 真值用臂。动态滤波固定 | 真实双目上 G1–G3 是否还成立 | 与 S 同方向：$t_z$、far 桶改善。手眼误差进底噪，不当成方法差。 |
| **公开集** | EuRoC / TUM VI / KITTI 相邻或对关键帧的短时 RPE | 不掉点 | 弱纹理/远景（MH_05、高速）上 $t_z$ 与失败率优于各向同性。KITTI 远车/路面分桶。 |
| **工程** | 点数、外层次数、耗时；$3\sigma_Z$ 门控与 $1\,\mathrm{px}$ 边缘门控开/关；搜索半径夹紧范围 | 实时与边缘点 | 帧率可接受；边缘门控关则深度跳变处失败率升。近处 $r(Z)$ 到 $15\,\mathrm{px}$，远处收到 $2\,\mathrm{px}$。 |
| **可选下游** | 同一 $\Sigma_{\mathrm{gicp}}$ 送固定 ESKF（$J_{g\to e}$ 开，动态滤波固定） | 配准均值准了，松耦合 ATE 会不会跟 | 只报一行。NEES / $J_{g\to e}$ 仍归姊妹篇 3。 |

顺序：S（模型）→ G1（关联）→ G2（度量）→ G3（分桶）→ G4（初值）→ G5（臂）→ 公开集。G1 与 G2 **不要同时改**：先锁关联再改 $C$，否则说不清是配对还是权重。

不做：六轴位姿析因、分轴 ATE 主表、把开/关动态滤波记成本篇收益、用 $J_{\mathrm{left}}$ 直接评滤波器。

### 4.2 控制变量与对照

| 固定 | 做法 |
|------|------|
| 动态滤波 | 全程同一套（建议本篇主表**关**，或全开且参数不变） |
| 点云与视差 | 同一帧对、同一稠密重建 |
| 初值 | 除 G4 外，一律 IMU 相对位姿或臂真值加固定小扰动 |
| 优化 | 同一套 LM、外层次数上限、收敛阈值 |

对照（G2，关联已锁为：自适应半径投影 + 命中像素中心光线；$C=R\Sigma_i R^\top+\Sigma_j$ 中 $\Sigma$ 均由同一套 $J_{uvd}$ 得到，目标横向 $\sigma^{\mathrm{match}}=0.5\,\mathrm{px}$，除非另行注明）：

| 组 | 关联 | 代价 |
|----|------|------|
| Iso-P2P | 自适应半径投影，命中像素中心光线 | $e^\top e$（各向同性点到点） |
| P2Plane | 同上 | $(n^\top e)^2$（点到面，目标法向） |
| **Cov-P2P** | 同上 | $e^\top C^{-1}e$（点到点 + 完整三维点协方差；无 Huber） |
| **Cov-P2Plane** | 同上 | $(n^\top e)^2/(n^\top C n)$（点到面 + 法向投影方差） |
| Diag-GICP | 同上 | $e^\top C_{\mathrm{diag}}^{-1}e$（对角 $\Sigma$，丢掉 $\mathrm{Cov}(X,Z)$；无 Huber） |
| **完整方法** | 同上 | 完整 $C$ + Huber（相对 Cov-P2P 只开鲁棒核） |
| 消融：NN | 欧氏最近邻 | 完整 $C$（给 G1） |
| 消融：连续光线 | $\mathbf{q}\parallel\mathbf{p}'$，横向残差被消掉 | 完整 $C$（给 G1） |
| 消融：过小 $\sigma^{\mathrm{match}}$ | 像素中心光线 | $\sigma_u^{\mathrm{match}}\ll 0.5\,\mathrm{px}$（给 G1） |

读表顺序建议：Iso-P2P → Cov-P2P → 完整方法（协方差与 Huber 的增量）；P2Plane → Cov-P2Plane（法向残差上加权是否够用）；Cov-P2P ↔ Diag-GICP（交叉项是否必要）。

外部叙事（公开集，不替代上面）：Open3D/small_gicp 各向同性、通用 GICP。不把 ORB-SLAM3 整条轨迹当本篇主对照。

### 4.3 真值、分桶与指标

**相对位姿真值。** 臂：$T_{c\leftarrow k}^{\mathrm{gt}}=T_{w\leftarrow c,c}^{\mathrm{arm}\,{-1}}T_{w\leftarrow c,k}^{\mathrm{arm}}$（或等价复合）。公开集：数据集 GT 的短时相对位姿。评的是 $\tilde T=T_{\mathrm{est}}^{-1}T_{\mathrm{gt}}$ 的 $\mathrm{Log}$，旋转与平移分开，平移再拆 $t_{xy}$ 与 $t_z$（目标相机系）。

**深度桶**（按目标点 $Z_j$ 或帧对中位深度）：near $<3\,\mathrm{m}$，mid $3$–$8\,\mathrm{m}$，far $>8\,\mathrm{m}$（KITTI 可改为 $<15$ / $15$–$40$ / $>40$）。每桶单独 RPE，避免近点把平均数拉平。

| 量 | 用途 |
|----|------|
| 相对旋转 / $t_{xy}$ / $t_z$ RMSE | 主精度；far+$t_z$ 是完整 $\Sigma$ 的试金石 |
| 外层迭代、失败率（不收敛或 $\lVert t\rVert$ 爆） | G4、远景 |
| 内点率、命中像素与 $\lvert\delta u\rvert,\lvert\delta v\rvert$（合成可算） | G1 |
| 单次配准耗时、点数 | 工程 |
| 可选：$\varepsilon=\xi^\top\Sigma_{\mathrm{gicp}}^{-1}\xi$ | $H^{-1}$ 是否大致罩住误差；只作附录，一致性主文在姊妹篇 3 |

### 4.4 合成（S）怎么做

固定几何（一块平板 + 一块成角的第二平面），真值 $T$ 已知。只对 $(u,v,d)$ 加零均值高斯，$\sigma_d$ 取两档（例如 $0.1\,\mathrm{px}$ 与 $0.5\,\mathrm{px}$），$\sigma_u,\sigma_v$ 保持较小。远点 $\mathrm{Var}(Z)\sim Z^4$，各向同性会被视线噪声拧 $t_z$；完整 $\Sigma$ 应主要吃切向。第二平面用来暴露欧氏 NN 跨面错配。不在这里扫机械臂轨迹。

### 4.5 臂上（G5）与公开集

与姊妹篇 1 同一套手眼与同步。本篇用**全静态**短时帧对即可，不需要独立运动物体（物体是动态滤波的事）。臂走：近桌面、中距、再看向远纹理，保证三桶都有点。初值用 IMU 复合（档 B 风格），真值用臂；不要把 $P_{\mathrm{arm}}$ 写进配准 $C$。

公开集：EuRoC 用 MH_01 与 MH_05（远+难）、V1_03；TUM VI 补暗光；KITTI 补远路面。只报短时帧对 RPE 与失败率，不把整段 ATE 当成配准器分数。

时延 $\Delta t$ 会偏 IMU 初值，属 G4 的一种现场来源；系统时延扫描归姊妹篇 1 档 D，本篇不单独做 $\Delta t$ 主表。

### 4.6 建议顺序

1. S：确认 $J_{uvd}$ 与完整 $\Sigma$ 在可控噪声下压 $t_z$。  
2. G1：只改关联/反投影，锁完整 $\Sigma$。  
3. G2：锁像素中心光线与 $0.5\,\mathrm{px}$ 匹配噪声，只改代价：Iso-P2P / P2Plane / Cov-P2P / Cov-P2Plane / Diag-GICP / 完整方法（+Huber）。  
4. G3：G2 的结果按深度桶重报。  
5. G4：加初值扰动，报收敛盆。  
6. G5：臂上重复 G1–G3 的主对照（不必再重扫 $\sigma^{\mathrm{match}}$ 的全部幅度）。  
7. 公开集短时 RPE；可选一行固定 ESKF 的 ATE。

## 5. 结论

1. 双目配准的关联用投影到深度图像素，代价用完整 $\Sigma$；否则远点视线噪声会主导 $t_z$，半像素取整残差也会在过小的横向方差下被拧紧。  
2. 目标点取命中像素的中心光线，横向匹配噪声取 $0.5\,\mathrm{px}$：半像素取整残差等于目标项的 $1\sigma$，避免量化残差被过强约束，也保留切向的亚像素信息。  
3. $J_{\mathrm{left}}$ 仅服务独立相对位姿优化；嵌入混合误差滤波器须改雅可比，几何项仍用同一 Huber（姊妹篇 3）。  
4. 局限：依赖深度图质量与静态场景；无独立运动模型。  
5. 后续：与动态滤波置信度加权、多尺度关联。
