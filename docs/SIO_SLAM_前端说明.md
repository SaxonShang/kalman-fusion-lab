# SIO_SLAM 前端方法说明

## 摘要

本文档面向当前工程实现，对 SIO_SLAM 前端的状态估计流程进行论文化描述。系统以双目图像和惯性测量为输入，先由 IMU 完成短时运动传播，再结合稠密点云配准完成状态校正；在普通帧采用 GICP 与误差状态卡尔曼滤波实现松耦合融合，在关键帧采用几何—光度联合残差的迭代误差状态卡尔曼滤波实现紧耦合优化。同时，系统在配准前引入动态点剔除模块，以抑制场景中运动物体对位姿估计的干扰。全文重点说明其状态定义、坐标变换关系、滤波公式及协方差对齐原则。

**关键词：** 双目视觉，IMU 预积分，ESKF，ESIKF，GICP，动态点剔除

**SCI 五篇论文（理论已写入各篇 §3）：** [SCI_五篇论文索引.md](SCI_五篇论文索引.md)

- [动态观测滤波](SCI_动态观测滤波.md)
- [投影 Stereo-GICP](SCI_投影StereoGICP.md)
- [ESKF 与 ESIKF](SCI_ESKF与ESIKF.md)
- [3DGS-IMU 后端](SCI_3DGS_IMU后端.md)
- [系统集成](SCI_系统集成.md)

**雅可比推导：** 已写入各论文 §3（[动态滤波 §3.4](SCI_动态观测滤波.md) · [Stereo-GICP §3.5](SCI_投影StereoGICP.md) · [ESKF/ESIKF §3.2–3.5](SCI_ESKF与ESIKF.md)）。


---

## 1. 方法概述

SIO_SLAM 前端旨在在连续图像流中稳定估计载体位姿。其核心思想是：利用 IMU 提供高频、短时间尺度内的运动先验，利用双目重建得到的点云提供几何观测，并在关键阶段引入颜色梯度信息增强配准稳定性。整体处理链可概括为如下步骤：

1. 读取当前图像与对应时间段的 IMU 数据；
2. 由双目视差恢复当前时刻三维点云；
3. 基于上一时刻状态执行 IMU 传播，获得当前帧先验；
4. 使用预测相对位姿完成动态点剔除；
5. 对非关键帧执行 GICP + ESKF 松耦合更新；
6. 对关键帧执行 ESIKF 紧耦合优化与关键帧管理。

该过程的主控制逻辑位于 [../src/frontend.cpp](../src/frontend.cpp)。

---

## 2. 系统状态与误差状态建模

### 2.1 名义状态

系统采用的名义状态可表示为

$$
\mathbf{x} = \left( \mathbf{R}_{wb},\ \mathbf{p}_{wb},\ \mathbf{v}_{wb},\ \mathbf{b}_g,\ \mathbf{b}_a,\ \mathbf{g}_w \right)
$$

其中，$\mathbf{R}_{wb}$ 为本体系到世界系的旋转矩阵，$\mathbf{p}_{wb}$ 和 $\mathbf{v}_{wb}$ 分别表示本体在世界系下的位置和速度，$\mathbf{b}_g$ 与 $\mathbf{b}_a$ 分别为陀螺仪与加速度计偏置，$\mathbf{g}_w$ 为世界坐标系下的重力向量。

### 2.2 误差状态定义

滤波器实际传播和更新的是误差状态。结合当前工程实现，其核心部分写为

$$
\delta \mathbf{x} =
\begin{bmatrix}
\delta \boldsymbol\theta_b \\
\delta \mathbf{p}_w \\
\delta \mathbf{v}_w \\
\delta \mathbf{b}_g \\
\delta \mathbf{b}_a \\
\delta \mathbf{g}_w
\end{bmatrix}
$$

该定义并非统一坐标下的纯局部误差，而是**混合误差坐标**：旋转误差 $\delta \boldsymbol\theta_b$ 定义在当前机体系，位置、速度与重力误差定义在世界系，偏置项定义在参数空间。这一约定直接决定了观测残差和观测协方差的表达方式。

### 2.3 旋转扰动模型

系统对旋转采用右扰动形式：

$$
\mathbf{R} = \hat{\mathbf{R}}\,\mathrm{Exp}(\delta \boldsymbol\theta_b)
$$

因此，旋转创新 $\delta \boldsymbol\theta_b$ 的物理含义是“当前机体系下的小角度修正量”。凡是进入滤波器更新的旋转误差，都必须与这一约定严格一致。

### 2.4 坐标系定义与变换关系

前端处理涉及**多个关键坐标系**的协调变换，理解这些关系是掌握整个系统一致性的核心。

#### 坐标系清单

| 坐标系           | 符号  | 说明                             | 备注                                                                |
| ------------- | --- | ------------------------------ | ----------------------------------------------------------------- |
| **世界坐标系**     | W   | 全局参考系，由第一帧 IMU 初始位姿固定          | 状态中 $\mathbf{R}_{wb}$, $\mathbf{p}_{wb}$, $\mathbf{v}_{wb}$ 定义在此系 |
| **IMU 本体坐标系** | B   | IMU 传感器原点的坐标系；在 IMU 和相机之间充当中介  | 陀螺/加速度计测量定义在此系                                                    |
| **关键帧坐标系**    | K   | 上一处理的前端关键帧的参考坐标系（原点同当时 IMU 位置） | 用于 GICP 点云配准的参考                                                   |
| **当前帧坐标系**    | C   | 当前处理帧对应的 IMU 坐标系（原点同当前 IMU 位置） | 新到达的观测所处坐标系                                                       |

#### 坐标系之间的变换矩阵

系统维护以下关键的位姿变换：

$$
\begin{align}
\mathbf{T}_{w}^{b} &= \left(\mathbf{R}_{wb},\ \mathbf{p}_{wb}\right) & \text{IMU 在世界系中的位姿（状态变量）} \\
\mathbf{T}_{w}^{k} &= \left(\mathbf{R}_{wk},\ \mathbf{p}_{wk}\right) & \text{上一关键帧（过去 IMU 位置）在世界系中的位姿} \\
\mathbf{T}_{w}^{c,pred} &= \left(\mathbf{R}_{wc}^{pred},\ \mathbf{p}_{wc}^{pred}\right) & \text{当前帧 IMU 的先验位姿（IMU 预积分传播得到）}
\end{align}
$$

#### 相对位姿的计算

**第1步：从世界系转换到局部相对位姿**

当前帧相对于上一关键帧的**预测相对位姿**为：

$$
\mathbf{T}_{k}^{c,pred} = \left(\mathbf{T}_{w}^{k}\right)^{-1} \mathbf{T}_{w}^{c,pred} = \left(\mathbf{R}_{kc}^{pred},\ \mathbf{t}_{c \text{ in } k}^{pred}\right)
$$

其中：
$$
\mathbf{R}_{kc}^{pred} = \left(\mathbf{R}_{wk}\right)^T \mathbf{R}_{wc}^{pred}
$
$$
\mathbf{t}_{c \text{ in } k}^{pred} = \left(\mathbf{R}_{wk}\right)^T \left(\mathbf{p}_{wc}^{pred} - \mathbf{p}_{wk}\right)
$

这个相对位姿作为 GICP 配准的**初值**。

**第2步：点的坐标变换**

将一个点从当前帧坐标系 C 变换到关键帧坐标系 K：

$$
\mathbf{p}_k = \mathbf{R}_{kc} \mathbf{p}_c + \mathbf{t}_{c \text{ in } k}
$$

其反变换（关键帧点投影到当前帧，用于动态点剔除）为：

$$
\mathbf{p}_c = \left(\mathbf{R}_{kc}\right)^T (\mathbf{p}_k - \mathbf{t}_{c \text{ in } k})
$$

**第3步：GICP 输出的坐标系对齐**

GICP 返回的相对位姿 $\mathbf{T}_{k}^{c,gicp}$ 定义在**局部坐标系 K-C**。为了用于 ESKF 更新，必须先转换到世界系：

$$
\mathbf{T}_{w}^{c,gicp} = \mathbf{T}_{w}^{k} \mathbf{T}_{k}^{c,gicp}
$$

#### 协方差的坐标系对齐

预测协方差的前 6×6 块对应的误差坐标为**混合形式**：

$$
[\delta \boldsymbol\theta_b,\ \delta \mathbf{p}_w]
$$

- $\delta \boldsymbol\theta_b$：在当前 IMU 机体系中定义（右扰动）
- $\delta \mathbf{p}_w$：在世界坐标系中定义

而 GICP 返回的协方差对应相对位姿的误差，即**关键帧局部坐标系**中的表达：

$$
[\delta \boldsymbol\theta_k,\ \delta \mathbf{t}_k]
$$

两者不在同一坐标定义下，**不能直接相加**。需使用雅可比变换：

$$
\mathbf{J}_{cov} = \begin{pmatrix}
\left(\mathbf{R}_{kc}\right)^T & \mathbf{0} \\
\mathbf{0} & \mathbf{R}_{wk}
\end{pmatrix}
$$

将 GICP 协方差变换为滤波器坐标：

$$
\Sigma_{obs} = \mathbf{J}_{cov} \Sigma_{gicp} \mathbf{J}_{cov}^T
$$

#### 坐标系与损失计算的关系总结

| 处理阶段           | 主要坐标系     | 说明                                     |
| -------------- | --------- | -------------------------------------- |
| **双目重建**       | 相机系 C（左目） | 点云初始在左目相机坐标系中                          |
| **IMU 预积分**    | IMU 本体系 B | 陀螺/加速度计在本体系测量                          |
| **GICP 配准**    | 相对坐标系 K→C | 局部点云匹配，输出相对位姿                          |
| **GICP 转换**    | 世界系 W     | GICP 输出需转到世界系进入 ESKF                   |
| **ESKF 预测**    | 混合误差坐标    | $(\delta\theta_b, \delta\mathbf{p}_w)$ |
| **ESIKF 几何残差** | 关键帧系 K    | 点变换到关键帧完成配准                            |

---

## 3. 双目重建模型

### 3.1 视差到深度的转换

在双目模块中，系统首先根据左右图像估计视差图，然后将每个有效像素恢复为三维点。设像素坐标为 $(u,v)$，视差为 $d$，相机参数为 $(f_x,f_y,c_x,c_y,B)$，则深度满足

$$
z = \frac{f_x B}{d}
$$

### 3.2 相机坐标系中的三维重建

进一步依据针孔成像关系，可得三维坐标（在**左目相机坐标系 C** 中）：

$$
\begin{aligned}
x_c &= \frac{(u-c_x)z}{f_x}, \\
y_c &= \frac{(v-c_y)z}{f_y}, \\
z_c &= \frac{f_x B}{d}
\end{aligned}
$$

其中 $(x_c, y_c, z_c)$ 是点在**相机坐标系 C** 中的三维坐标。

### 3.3 从相机系到 IMU 本体系的变换

为了与后端处理保持一致，系统需要将相机系中的点变换到 **IMU 本体坐标系 B**。这涉及左目相机与 IMU 之间的外参标定关系：

$$
\mathbf{T}_{b}^{c} = \left(\mathbf{R}_{bc},\ \mathbf{t}_{bc}\right)
$$

其中 $\mathbf{R}_{bc}$ 和 $\mathbf{t}_{bc}$ 分别为相机相对于 IMU 的旋转和位移，通常在系统标定时获得。

点在 IMU 本体系中的表示为：

$$
\mathbf{p}_b = \mathbf{R}_{bc}^T (\mathbf{p}_c - \mathbf{t}_{bc})
$$

或等价地：

$$
\mathbf{p}_b = \mathbf{R}_{cb}^T \mathbf{p}_c - \mathbf{R}_{cb}^T \mathbf{t}_{bc}
$$

但在前端流程中，点云通常先在相机坐标系完成配准，用到的仍是**相机系坐标**。坐标系变换主要用于与后端同步时。

### 3.4 点云坐标系的处理链

```
图像像素 (u,v) 
        ↓
    视差估计 
        ↓
深度计算 z = fxB/d
        ↓
相机系三维坐标 (x_c, y_c, z_c)  [坐标系: C]
        ↓
可选: 变换到 IMU 本体系 (x_b, y_b, z_b)  [坐标系: B]
        ↓
用于 GICP 配准和点云处理
```

由此可见，双目模块本质上完成了从图像域到三维空间的反投影。根据后续处理需要，点云可维持在相机系或配准到 IMU 本体系，但**必须明确记录点云当前所在的坐标系**，以避免在多坐标系变换中产生方向混淆。

---

## 4. IMU 先验传播模型

### 4.1 IMU 测量与坐标系

IMU 的陀螺仪和加速度计都在 **IMU 本体坐标系 B** 中进行测量。设从时刻 $k-1$ 到时刻 $k$ 的 IMU 原始测量为：

$$
\boldsymbol\omega_m \in \mathbb{R}^3 \quad \text{（本体系中的角速度）}
$$

$$
\mathbf{a}_m \in \mathbb{R}^3 \quad \text{（本体系中的比力加速度）}
$$

去偏后的测量值（移除陀螺零偏 $\mathbf{b}_g$ 和加速度计零偏 $\mathbf{b}_a$）为：

$$
\tilde{\boldsymbol\omega} = \boldsymbol\omega_m - \mathbf{b}_g, \qquad \tilde{\mathbf{a}} = \mathbf{a}_m - \mathbf{b}_a
$$

### 4.2 名义状态传播（在世界坐标系中）

系统维护的名义状态位置和速度都表示在**世界坐标系 W** 中。IMU 的旋转矩阵 $\mathbf{R}_{wb}$ 表示将**本体系向量变换到世界系**。

离散时间下的状态传播近似写为：

$$
\mathbf{R}_k = \mathbf{R}_{k-1}\,\mathrm{Exp}(\tilde{\boldsymbol\omega}\Delta t)
$$

$$
\mathbf{v}_k = \mathbf{v}_{k-1} + \left(\mathbf{R}_{k-1}\tilde{\mathbf{a}} + \mathbf{g}_w\right)\Delta t
$$

$$
\mathbf{p}_k = \mathbf{p}_{k-1} + \mathbf{v}_{k-1}\Delta t + \frac{1}{2}\left(\mathbf{R}_{k-1}\tilde{\mathbf{a}} + \mathbf{g}_w\right)\Delta t^2
$$

其中 $\mathbf{g}_w = [0, 0, g]^T$ 是**世界坐标系中的重力向量**。

**关键点**：本体系中测得的加速度 $\tilde{\mathbf{a}}$ 需先通过 $\mathbf{R}_{k-1}$ 变换到世界系，再与重力加速度 $\mathbf{g}_w$ 相加。

### 4.3 协方差传播（混合误差坐标）

误差协方差的传播采用标准的线性化形式：

$$
\mathbf{P}_k = \mathbf{F}_k\mathbf{P}_{k-1}\mathbf{F}_k^T + \mathbf{G}_k\mathbf{Q}_k\mathbf{G}_k^T
$$

其中：

- $\mathbf{F}_k$ 为误差状态转移矩阵
- $\mathbf{G}_k$ 为噪声输入矩阵
- $\mathbf{Q}_k$ 为离散噪声协方差

关键注意：传播过程中，旋转误差 $\delta\boldsymbol\theta_b$ 始终定义在**当前本体系**，而位置误差 $\delta\mathbf{p}_w$ 定义在**世界系**。这种**混合误差坐标**定义贯穿整个前端。

当前工程中，传播后的状态和协方差保存在 `imu_preintegration_` 中，用作后续视觉校正的预测量。

---

## 5. 坐标系综合参考

第2.4节已介绍了前端中的四个主要坐标系及其相对关系。本章进一步总结其在各处理流程中的具体应用。

### 5.1 坐标系记号总结

| 记号                        | 完整说明                   | 用途               |
| ------------------------- | ---------------------- | ---------------- |
| $\mathbf{T}_{w}^{b}$      | IMU 在世界系中的绝对位姿（状态向量）   | 滤波器主状态，由IMU预积分传播 |
| $\mathbf{T}_{w}^{c,pred}$ | 当前帧 IMU 的先验位姿（预积分得）    | 提前知道的部分观测信息      |
| $\mathbf{T}_{w}^{k}$      | 上一关键帧 IMU 的位姿（过去状态）    | 相对位姿的参考基准        |
| $\mathbf{T}_{k}^{c,pred}$ | **相对位姿预测**（从关键帧看当前帧）   | GICP 配准的初值       |
| $\mathbf{T}_{k}^{c,gicp}$ | **GICP 输出的相对位姿**（局部坐标） | 必须转到世界系才能用于ESKF  |
| $\mathbf{T}_{w}^{c,gicp}$ | GICP 观测转为世界系（用于ESKF）   | 滤波器的观测，与预测同系     |

### 5.2 相对位姿的计算与转换

#### 从世界系推出相对位姿

当前帧相对于上一关键帧的**预测相对位姿**由以下公式计算：

$$
\mathbf{T}_{k}^{c,pred} = \left(\mathbf{T}_{w}^{k}\right)^{-1} \mathbf{T}_{w}^{c,pred}
$$

展开为旋转和平移：

$$
\mathbf{R}_{kc}^{pred} = \left(\mathbf{R}_{wk}\right)^T \mathbf{R}_{wc}^{pred} \quad \text{（关键帧看当前帧的旋转）}
$$

$$
\mathbf{t}_{c \text{ in } k}^{pred} = \left(\mathbf{R}_{wk}\right)^T \left(\mathbf{p}_{wc}^{pred} - \mathbf{p}_{wk}\right) \quad \text{（当前帧在关键帧系中的位置）}
$$

这个相对位姿正是 **GICP 配准的初值**。

#### GICP 观测的坐标转换

GICP 配准返回的相对位姿 $\mathbf{T}_{k}^{c,gicp}$ 定义在**关键帧-当前帧的局部坐标系对**中。为了用于 ESKF 更新（此时的观测需与预测同系），必须转回**世界坐标系**：

$$
\mathbf{T}_{w}^{c,gicp} = \mathbf{T}_{w}^{k} \mathbf{T}_{k}^{c,gicp}
$$

即：

$$
\mathbf{R}_{wc}^{gicp} = \mathbf{R}_{wk} \mathbf{R}_{kc}^{gicp}
$$

$$
\mathbf{p}_{wc}^{gicp} = \mathbf{p}_{wk} + \mathbf{R}_{wk} \mathbf{t}_{kc}^{gicp}
$$

---

### 5.3 点云坐标变换

#### 关键帧点投影到当前帧

用于**动态点剔除**：计算关键帧中每个点在当前帧的位置，生成几何光流预期。

设关键帧点在世界系中为 $\mathbf{p}_{w}^{key}$，先变换到关键帧系：

$$
\mathbf{p}_{k}^{key} = \mathbf{R}_{wk}^T (\mathbf{p}_{w}^{key} - \mathbf{p}_{wk})
$$

再用预测的相对位姿投影到当前帧系：

$$
\mathbf{p}_{c}^{pred} = \left(\mathbf{R}_{kc}^{pred}\right)^T (\mathbf{p}_{k}^{key} - \mathbf{t}_{c \text{ in } k}^{pred})
$$

#### 当前帧点投影到关键帧

用于 **GICP 和 ESIKF 配准**：当前帧点需变换到关键帧进行比对。

设当前帧点在世界系中为 $\mathbf{p}_{w}^{curr}$（按世界系位姿反推），或直接用相对位姿：

$$
\mathbf{p}_{k} = \mathbf{R}_{kc}^{gicp} \mathbf{p}_{c} + \mathbf{t}_{c \text{ in } k}^{gicp}
$$

这是 GICP 和 ESIKF 中几何残差计算的关键步骤。

### 5.4 旋转与平移的混合误差坐标

前端预测协方差前6维的误差定义为：

$$
[\delta \boldsymbol\theta_b,\ \delta \mathbf{p}_w] \in \mathbb{R}^{3+3}
$$

其中：

- $\delta \boldsymbol\theta_b$：在**当前 IMU 机体系 B**（右扰动）
- $\delta \mathbf{p}_w$：在**世界系 W**

这种**混合定义**贯穿传播和更新全程。

#### 来自 GICP 的协方差转换

GICP 输出的协方差 $\Sigma_{gicp}$ 对应相对位姿：$[\delta\boldsymbol\theta_{k}, \delta\mathbf{t}_k]$（文献中常见的形式）。

为与预测协方差对齐，需使用雅可比变换矩阵：

$$
\mathbf{J}_{cov} = \begin{pmatrix}
\left(\mathbf{R}_{kc}\right)^T & \mathbf{0} \\
\mathbf{0} & \mathbf{R}_{wk}
\end{pmatrix}
$$

变换后的观测协方差：

$$
\Sigma_{obs}^{aligned} = \mathbf{J}_{cov} \Sigma_{gicp} \mathbf{J}_{cov}^T
$$

**变换的物理含义**：

- 旋转块 $(\mathbf{R}_{kc})^T$ ：将"从关键帧角度看的左扰动"转为"从当前帧角度看的右扰动"
- 平移块 $\mathbf{R}_{wk}$：将"相对位置误差"旋转到世界系的对齐方向

只有完成这一对齐，创新协方差才能正确合成：

$$
\mathbf{S} = \mathbf{P}_{pred}^{pose} + \Sigma_{obs}^{aligned}
$$

### 5.5 ESIKF 中的坐标系说明

关键帧优化（ESIKF）中，所有点和梯度都需在**关键帧坐标系 K** 中计算：

$$
\mathbf{p}_{k} = \mathbf{R}_{kc} \mathbf{p}_{c} + \mathbf{t}_{c \text{ in } k}
$$

在此基础上，几何/光度残差与其雅可比都遵循**右扰动、混合误差坐标**的约定，与 ESKF 的定义保持一致（详见第8章）。Stereo-GICP 几何项嵌入 ESIKF 后的逐步雅可比推导见 [SCI_ESKF与ESIKF.md §3.4](SCI_ESKF与ESIKF.md)。

---

---

## 6. 非关键帧更新：GICP 与 ESKF 松耦合融合

### 6.1 GICP 配准模型与坐标系

在普通帧情形下，系统使用 GICP 估计当前点云相对于上一关键帧点云的**相对位姿**：

$$
\mathbf{T}_{k}^{c,gicp} = \left(\mathbf{R}_{kc}^{gicp},\ \mathbf{t}_{kc}^{gicp}\right)
$$

这是在**关键帧坐标系 K** 和**当前帧坐标系 C** 之间的相对变换，代表"从关键帧看当前帧"的相对位姿。

**坐标系变换至关重要**：GICP 配准的初值由第5.2节给出的预测相对位姿提供，确保迭代收敛的稳定性。

### 6.2 相对位姿转为绝对世界系位姿

ESKF 更新需要的是**世界系下的绝对观测位姿**，而不是相对位姿。因此必须进行坐标转换：

$$
\mathbf{T}_{w}^{c,gicp} = \mathbf{T}_{w}^{k} \mathbf{T}_{k}^{c,gicp}
$$

展开为：

$$
\mathbf{R}_{wc}^{gicp} = \mathbf{R}_{wk} \mathbf{R}_{kc}^{gicp}
$$

$$
\mathbf{p}_{wc}^{gicp} = \mathbf{p}_{wk} + \mathbf{R}_{wk} \mathbf{t}_{kc}^{gicp}
$$

**代码实现**（frontend.cpp 第 374 行）：

```cpp
const Eigen::Matrix4d T_w_curr_gicp = T_w_prevkf * gicp_out.T.cast<double>();
```

其中 `T_w_prevkf` 是关键帧的世界系位姿，`gicp_out.T` 是 GICP 返回的相对位姿。

只有进行此变换，$\mathbf{T}_{w}^{c,gicp}$ 才能与 ESKF 预测的 $\mathbf{T}_{w}^{c,pred}$ 在同一坐标系中进行比较，确保滤波器的观测与预测在同一参考系下。

### 6.3 创新项构造（混合误差坐标）

根据前端定义的**右扰动和混合误差坐标**，旋转创新可表示为

$$
\delta \boldsymbol\theta = \log\left(\left(\mathbf{R}_{wc}^{pred}\right)^T \mathbf{R}_{wc}^{gicp}\right)
$$

位置创新为

$$
\delta \mathbf{p}_w = \mathbf{p}_{wc}^{gicp} - \mathbf{p}_{wc}^{pred}
$$

从而形成观测残差向量

$$
\mathbf{z}_{innov} =
\begin{bmatrix}
\delta \boldsymbol\theta \\
\delta \mathbf{p}_w
\end{bmatrix}
$$

其中 $\delta \boldsymbol\theta$ 是李代数表示，$\delta \mathbf{p}_w$ 是世界系下的位置差异。

### 6.4 卡尔曼更新公式

当前实现仅使用位姿前 6 维参与视觉更新，因此定义选择矩阵

$$
\mathbf{H}_{sel} =
\begin{bmatrix}
\mathbf{I}_3 & \mathbf{0} & \cdots \\
\mathbf{0} & \mathbf{I}_3 & \cdots
\end{bmatrix}
$$

则卡尔曼增益为

$$
\mathbf{K} = \mathbf{P}_{pred}\mathbf{H}_{sel}^T\left(\mathbf{H}_{sel}\mathbf{P}_{pred}\mathbf{H}_{sel}^T + \mathbf{R}_{obs}\right)^{-1}
$$

后验修正量为

$$
\delta \mathbf{x} = \mathbf{K}\mathbf{z}_{innov}
$$

协方差采用 Joseph 形式更新：

$$
\mathbf{P}_{post} = (\mathbf{I}-\mathbf{K}\mathbf{H})\mathbf{P}_{pred}(\mathbf{I}-\mathbf{K}\mathbf{H})^T + \mathbf{K}\mathbf{R}_{obs}\mathbf{K}^T
$$

该更新方式兼顾数值稳定性和对称正定性。

### 6.5 坐标对齐总结

```
GICP 配准获得
  └─→ 相对位姿: T_k^{c,gicp}  [坐标系: K-C 相对]
       └─→ × T_w^k 
            └─→ 绝对位姿: T_w^{c,gicp}  [坐标系: W 绝对]
                 └─→ 与预测 T_w^{c,pred} 比较
                      └─→ 计算创新 δ(·)
                           └─→ 送入 ESKF 更新
```

---

## 7. 观测协方差的坐标对齐问题

### 7.0 观测协方差的四层坐标系关系

点云观测协方差在系统中经历多层坐标变换。理解这一变换链条是确保滤波器数值精确性的关键。

**第一层：相机坐标系 C**

点的位置不确定性首先在双目相机坐标系中定义：$\boldsymbol{\Sigma}^C = \text{diag}(\sigma_x^2, \sigma_y^2, \sigma_z^2)$

这是各向异性的对角矩阵，反映：

- 沿焦距方向的像素噪声投影（$\sigma_x^2$、$\sigma_y^2$）
- 沿深度方向的视差噪声投影（$\sigma_z^2$，最大）

**第二层：IMU 体系 B**

将相机系协方差旋转到 IMU 体系（用外参 $\mathbf{R}_{BC}$、$\mathbf{t}_{BC}$）：
$\boldsymbol{\Sigma}^B = \mathbf{R}_{BC} \boldsymbol{\Sigma}^C (\mathbf{R}_{BC})^T$

此时点在体系中的坐标为 $\mathbf{p}_b = \mathbf{R}_{BC} (\mathbf{p}_c - \mathbf{t}_{BC})$，协方差仍为 $3 \times 3$ 正定矩阵（一般非对角）。

**第三层：关键帧体系 K**

在 ESIKF 中，点被变换到关键帧坐标系：$\mathbf{q} = \mathbf{R}_{kb} \mathbf{p}_b + \mathbf{t}_{kb}$

相应的协方差通过雅可比变换：
$$\boldsymbol{\Sigma}^K = \mathbf{R}_{kb} \boldsymbol{\Sigma}^B (\mathbf{R}_{kb})^T$

**第四层：误差状态坐标 $[\delta\boldsymbol\theta_b, \delta\mathbf{p}_w]$**

最后，残差在误差状态定义下的雅可比为：
$$\mathbf{J} = \begin{bmatrix} \mathbf{q}_{rel} \times \hat{\mathbf{n}} \\ \mathbf{R}_{wk} \hat{\mathbf{n}} \end{bmatrix}$

点位置不确定性在几何残差方向的投影为：
$$\sigma_{obs}^2 = \mathbf{J}^T \boldsymbol{\Sigma}^K \mathbf{J}$

这给出了观测的总噪声方差，用于信息权重。

**整体流程图**：

```
Point in camera C with Σ_c
           |
           | R_BC rotation (external param)
           v
Point in IMU body B with Σ_b
           |
           | Iterative transform to keyframe K
           v
Point in keyframe K with Σ_k
           |
           | Jacobian projection onto [δθ_b, δp_w]
           v
Geometric residual noise σ²_geo = n̂ᵀ Σ_k n̂
Photometric residual noise σ²_color = d_p⊥ᵀ Σ_k d_p⊥
```

### 7.1 问题本质

系统预测协方差前 6 维对应的误差语义为

$$
[\delta \boldsymbol\theta_b,\ \delta \mathbf{p}_w]
$$

即旋转误差位于当前机体系，而平移误差位于世界系。另一方面，GICP 输出的相对协方差通常对应于关键帧局部坐标下的相对位姿参数，即

$$
[\delta \boldsymbol\theta_k,\ \delta \mathbf{t}_k]
$$

二者不在同一误差坐标定义下，因此不能直接相加。

### 7.2 协方差变换

为实现与滤波器预测协方差的一致对齐，需构造映射矩阵

$$
\mathbf{J}_{obs} = \mathrm{blkdiag}\left(\mathbf{R}_{k\leftarrow c}^T,\ \mathbf{R}_{w\leftarrow k}\right)
$$

**代码实现**（frontend.cpp 第 375-380 行）：

```cpp
const Eigen::Matrix3d R_prevkf_curr = gicp_out.T.block<3, 3>(0, 0).cast<double>();
const Eigen::Matrix3d &R_w_prevkf = prev_keyframe_.imu_states.rot_end;
Eigen::Matrix<double, 6, 6> J_obs = Eigen::Matrix<double, 6, 6>::Zero();
J_obs.block<3, 3>(0, 0) = R_prevkf_curr.transpose(); // rot: prevkf→curr body
J_obs.block<3, 3>(3, 3) = R_w_prevkf;                // pos: prevkf→world
const Eigen::Matrix<double, 6, 6> R_obs_aligned = 
  J_obs * gicp_out.covariance * J_obs.transpose();
```

将 GICP 原始协方差映射为

$$
\Sigma_{obs} = \mathbf{J}_{obs}\Sigma_{gicp}\mathbf{J}_{obs}^T
$$

其中：

- 旋转块：GICP 中"关键帧作为参考的左扰动"转为"当前帧为参考的右扰动"。代码中通过 `R_prevkf_curr.transpose()` 实现
- 平移块：将"关键帧局部坐标系"的平移误差旋转到"世界坐标系"。代码中通过 `R_w_prevkf` 实现

完成对齐后的创新协方差为 $\mathbf{S} = \mathbf{P}_{pred}^{pose} + \Sigma_{obs}^{aligned}$，两项均在 $[\delta\boldsymbol\theta_b, \delta\mathbf{p}_w]$ 坐标框架下。

#### 协方差流向对比

观测协方差在两种滤波器中的完整处理过程对比：

**ESKF（GICP 松耦合）**：  
GICP输出 (R_rel, Σ_gicp) → Jacobian变换 J_obs·Σ_gicp·J_obs^T → 创新协方差 S = P_pred + Σ_aligned → 一次Kalman更新

**ESIKF（点云紧耦合）**：  
双目点云 → 相机系各向异性 [σ²_x, σ²_y, σ²_z]_C → 旋转到体系 R_BC·Σ_C·R_BC^T → 5次迭代重线性化到关键帧 → 法向/梯度投影 → 自适应噪声加权 → 累加到 H_obs, b_obs → 5轮迭代Kalman更新

核心差异：ESKF 采用一次性观测变换，ESIKF 采用逐次迭代与自适应加权。

在关键帧场景下，系统不再使用单个刚体位姿观测，而是直接基于点云对应关系构建联合残差，通过多轮迭代完成状态修正。所有计算都在**关键帧坐标系 K** 中进行。

### 8.1 点云坐标变换与几何残差

#### 坐标变换流程（当前帧 B → 关键帧 K）

在ESIKF中，所有计算都在**关键帧坐标系 K** 中进行。关键变换公式为：

**第一步**：将当前帧（IMU体系 B）点 $\mathbf{p}_b$ 变换到关键帧系中的点 $\mathbf{q}$：

$$
\mathbf{R}_{k \leftarrow b}^{curr} = \mathbf{R}_{wk}^T \cdot \mathbf{R}_{wb}^{curr}
$$

$$
\mathbf{t}_{b \text{ in } k}^{curr} = \mathbf{R}_{wk}^T \cdot (\mathbf{p}_{wb}^{curr} - \mathbf{p}_{wk})
$$

$$
\mathbf{q} = \mathbf{R}_{k \leftarrow b}^{curr} \cdot \mathbf{p}_b + \mathbf{t}_{b \text{ in } k}^{curr}
$$

其中：

- $\mathbf{R}_{wk}$、$\mathbf{p}_{wk}$：上一关键帧的世界系位姿；代码中分别对应 `R_anchor` 与 `t_anchor`
- $\mathbf{R}_{wb}^{curr}$、$\mathbf{p}_{wb}^{curr}$：当前帧（IMU体系）的世界系位姿估计
- $\mathbf{t}_{b \text{ in } k}^{curr}$：当前 IMU 原点在关键帧体系 K 中的坐标
- 结果 $\mathbf{q}$ 坐标在**关键帧体系 K** 中

**代码实现** ([state_estimater.cpp](../src/state_estimater.cpp) lines 413-419)：

```cpp
const Eigen::Matrix3d R_curr_to_prevkf = R_anchor.transpose() * imu_states.rot_end;
const Eigen::Vector3d t_curr_in_prevkf =
  R_anchor.transpose() * (imu_states.pos_end - t_anchor);
for (int i = 0; i < N; ++i) {
  const auto &src = pointcloud_cur->points[i];
  const Eigen::Vector3d p_prevkf =
    R_curr_to_prevkf * Eigen::Vector3d(src.x, src.y, src.z) + t_curr_in_prevkf;
  ...
}
```

### 8.1b ESIKF 中的点云观测协方差 vs ESKF 中的 GICP 协方差

两种滤波器处理观测协方差的方式差异巨大，反映了其不同的残差建模策略。

| 特性           | ESKF（GICP）               | ESIKF（点云）               |
| -------------- | -------------------------- | --------------------------- |
| **观测来源**   | 6-DOF 刚体位姿（单个观测） | 点对配置（多个残差）        |
| **协方差来源** | GICP 算法输出（整体）      | 各点位置不确定性（逐点）    |
| **坐标系**     | 相对位姿局部（关键帧对齐） | 绝对坐标（体系、关键帧系）  |
| **变换方式**   | 一次矩阵变换 $J_{obs}$     | 全过程迭代重线性化          |
| **计算量**     | 低（单一变换）             | 中等（5次迭代，每次全点集） |
| **鲁棒性**     | 依赖GICP收敛质量           | 通过融合与降权增强鲁棒性    |

**关键区别**：

**ESKF的GICP协方差**：
- 来自 GICP EM-ICP 迭代的对应关系不确定性
- 是 6×6 矩阵，表示两帧相对位姿的协方差
- 需要一次性变换到 $[\delta\boldsymbol\theta_b, \delta\mathbf{p}_w]$ 空间
- 使用 **Jacobian 投影**：$\Sigma_{obs} = J_{obs} \Sigma_{gicp} J_{obs}^T$

**ESIKF的点云协方差**：
- 来自双目立体测距噪声与网络置信度
- 每一点都有独立的 3×3 位置协方差 $\Sigma_i^B$
- 在迭代过程中被逐次投影到法向/梯度方向
- 通过**法向投影**：$\sigma^2 = \hat{\mathbf{n}}^T \Sigma_i^B \hat{\mathbf{n}}$ 获得残差方向噪声
- 在每次迭代前被重新计算（因为点坐标发生变化）

#### 几何残差的定义

对变换后的源点 $\mathbf{q}$（当前帧在关键帧系中）、对应的目标点 $\mathbf{p}$（关键帧中）及其单位法向量 $\hat{\mathbf{n}}$（都在**关键帧系 K** 中），定义点到面残差：

$$
r_{geo} = (\mathbf{q} - \mathbf{p})^T\hat{\mathbf{n}}
$$

相对位移向量：

$$
\mathbf{q}_{rel} = \mathbf{q} - \mathbf{t}_{b \text{ in } k}^{curr} = \mathbf{R}_{k \leftarrow b}^{curr}\mathbf{p}_b
$$

**坐标说明**：

- $\mathbf{q} \in K$：当前帧点在关键帧坐标系中的位置
- $\mathbf{p} \in K$：关键帧中的对应目标点
- $\hat{\mathbf{n}} \in K$：关键帧中目标点的单位法向量
- $\mathbf{q}_{rel} \in K$：当前点相对当前 IMU 原点的向量，但其坐标表达在关键帧体系 K 中

#### 几何残差的线性化

在右扰动、混合误差坐标 $[\delta\boldsymbol\theta_b, \delta\mathbf{p}_w]$ 下，一阶泰勒展开为：

$$
r_{geo} \approx r_0 + \frac{\partial r_{geo}}{\partial \delta\boldsymbol\theta_b}\delta\boldsymbol\theta_b + \frac{\partial r_{geo}}{\partial \delta\mathbf{p}_w}\delta\mathbf{p}_w
$$

$$
\frac{\partial r_{geo}}{\partial \delta\boldsymbol\theta_b} = (\mathbf{q}_{rel} \times \hat{\mathbf{n}})^T \in \mathbb{R}^{1 \times 3}
$$

其中 $\mathbf{q}_{rel}$ 与 $\hat{\mathbf{n}}$ 都表达在关键帧体系 K 中，但该项对应的是对当前机体系右扰动 $\delta\boldsymbol\theta_b$ 的偏导。

**平移偏导数**（世界系 W 中的微分）：

$$
\frac{\partial r_{geo}}{\partial \delta\mathbf{p}_w} = (\mathbf{R}_{wk}\hat{\mathbf{n}})^T \in \mathbb{R}^{1 \times 3}
$$

其中 $\mathbf{R}_{wk}$ 为关键帧的世界系旋转。

**几何残差的雅可比矩阵**（$1 \times 6$ 选择向量）：

$$
\mathbf{J}_{geo} = \begin{bmatrix} (\mathbf{q}_{rel} \times \hat{\mathbf{n}})^T & (\mathbf{R}_{wk}\hat{\mathbf{n}})^T \end{bmatrix} \in \mathbb{R}^{1 \times 6}
$$

**代码实现** ([state_estimater.cpp](../src/state_estimater.cpp) lines 134-136)：

```cpp
const Eigen::Vector3d q_rel = q - source_origin_in_target;
factor.J_geo.head<3>() = q_rel.cross(n_unit);        // 旋转块：体系中
factor.J_geo.tail<3>() = R_target_to_world * n_unit; // 平移块：世界系旋转应用
```

### 8.2 光度残差建模

为提高纹理区域的约束能力，系统进一步引入颜色梯度项。

#### 切平面梯度计算

将目标点颜色梯度 $\mathbf{d}_p$（关键帧系 K 中）投影到法向垂直的切平面：

$$
\mathbf{d}_p^{\perp} = \mathbf{d}_p - (\mathbf{d}_p^T\hat{\mathbf{n}})\hat{\mathbf{n}}
$$

该切平面梯度垂直于目标点法向，代表沿曲面的颜色变化方向。

#### 光度残差定义

构造光度残差（在关键帧系 K 中）：

$$
r_{color} = I_p + (\mathbf{d}_p^{\perp})^T(\mathbf{q}-\mathbf{p}) - I_q
$$

其中：

- $I_p$：目标点的灰度值
- $I_q$：源点的灰度值
- $(\mathbf{d}_p^{\perp})^T(\mathbf{q}-\mathbf{p})$ 沿切平面方向的梯度修正

物理含义：目标点灰度应沿梯度方向与源点匹配。

#### 光度残差的雅可比

使用同样的线性化方法，光度残差的雅可比为（$1 \times 6$ 选择向量）：

$\mathbf{J}_{color} = \begin{bmatrix} (\mathbf{q}_{rel} \times \mathbf{d}_p^{\perp})^T & (\mathbf{R}_{wk}\mathbf{d}_p^{\perp})^T \end{bmatrix}$

结构与几何残差完全一致，仅是法向量 $\hat{\mathbf{n}}$ 替换为切平面梯度 $\mathbf{d}_p^{\perp}$。

**代码实现** ([state_estimater.cpp](../src/state_estimater.cpp) lines 138-142)：

```cpp
const Eigen::Vector3d dpM = target_color_gradient - target_color_gradient.dot(n_unit) * n_unit;

factor.r_color = ip + dpM.dot(q - p) - iq;

factor.J_color.head<3>() = q_rel.cross(dpM); // 旋转块

factor.J_color.tail<3>() = R_target_to_world * dpM; // 平移块
```

### 8.2b 点云观测协方差的计算

点云观测中每个点的位置不确定性由双目视差、网络置信度和相机几何特性共同决定。本节详细说明协方差在不同坐标系间的变换关系。

#### 相机坐标系中的各向异性协方差

双目立体视觉点云的位置不确定性模型基于**误差传播定律**。设点深度为 $z = f_x b/d$（其中 $f_x$ 为焦距、$b$ 为基线、$d$ 为视差），则关键的物理模型为：

**视差空间噪声**（由网络置信度 $c \in (0,1]$ 调制）：

$\sigma_d^2(c) = \frac{\sigma_{d,base}^2}{c}$

其中 $\sigma_{d,base}^2$ 为视差测量基础噪声，置信度调制遵循：

- $c \to 1$（高置信）：$\sigma_d^2 \to \sigma_{d,base}^2$（接近仪器噪声下界）

- $c \to 0$（低置信）：$\sigma_d^2 \to \infty$（测量无效）

**深度空间噪声**（通过误差传播从视差传播）：

$\sigma_z^2 = \left(\frac{z^2}{f_x b}\right)^2 \cdot \sigma_d^2 = \frac{\sigma_{d,base}^2}{c} \cdot \left(\frac{z^2}{f_x b}\right)^2, \quad \sigma_z^2 \geq \sigma_{min}^2$

其中 $\sigma_{min}^2 = 10^{-4}$ m$^2$ 为各向异性协方差的下限。

**像素空间噪声**：

$\sigma_u^2 = \sigma_v^2 = \frac{\sigma_{d,base}^2}{c}$

**相机坐标系各向异性协方差**（通过内参投影到3D）：

$\sigma_x^2 = \sigma_{min}^2 + \frac{z^2}{f_x^2}\sigma_u^2 + \left(\frac{x}{z}\right)^2\sigma_z^2$

$\sigma_y^2 = \sigma_{min}^2 + \frac{z^2}{f_y^2}\sigma_v^2 + \left(\frac{y}{z}\right)^2\sigma_z^2$

$(x_c, y_c, z) \in \mathbb{R}^3 \text{ 为点在相机系的坐标}$

相机系中的对角协方差矩阵：

$\boldsymbol{\Sigma}_c = \begin{bmatrix} \sigma_x^2 & 0 & 0 \\ 0 & \sigma_y^2 & 0 \\ 0 & 0 & \sigma_z^2 \end{bmatrix}$

**代码实现** ([state_estimater.cpp](../src/state_estimater.cpp) lines 253-276)：

```cpp

```

// ── 像素空间视差方差 ──

const double sigma_u2 = disp_var_base_ / src_conf;

const double sigma_v2 = disp_var_base_ / src_conf;

// ── 深度噪声 ──

const double ratio = (z * z) / fxb; // z²/(f_x·b)，单位 m/px

const double sigma_z2 = kMinPointVar + disp_var_base_ * ratio * ratio / src_conf;

// ── 3D各向异性传播 ──

const double zx = pc.x() / std::max(z, 1e-6);

const double zy = pc.y() / std::max(z, 1e-6);

// 相机系下各向异性位置方差（由像素噪声与深度噪声共同传播）

const double sigma_x2 = kMinPointVar + (z * z / (fx * fx)) * sigma_u2 + (zx * zx) * sigma_z2;

const double sigma_y2 = kMinPointVar + (z * z / (fy * fy)) * sigma_v2 + (zy * zy) * sigma_z2;

#### 坐标变换：相机系 C → 体系 B

在 ESIKF 中使用的所有点坐标都是在 **IMU 体系 B** 中，因此需要将相机系协方差旋转到体系：

$\boldsymbol{\Sigma}_{pos}^B = \mathbf{R}_{BC} \boldsymbol{\Sigma}_c^C (\mathbf{R}_{BC})^T$

其中 $\mathbf{R}_{BC}$ 为相机到体系的旋转（通常由外参 $T_{body}^{camera}$ 确定）。

此后点位置方差投影到几何/光度残差方向：

$\sigma_{geo}^2 = \hat{\mathbf{n}}^T \boldsymbol{\Sigma}_{pos}^B \hat{\mathbf{n}}$

$\sigma_{color}^2 = (\mathbf{d}_p^{\perp})^T \boldsymbol{\Sigma}_{pos}^B (\mathbf{d}_p^{\perp})$

**代码实现** ([state_estimater.cpp](../src/state_estimater.cpp) lines 166-176)：

```cpp
// ── 为与相机成像模型一致 ──

// 在相机系直接构建各向异性协方差Σ_c，再旋转到体系

const auto point_cov_body = [&](const Eigen::Vector3d &pc, double z, double c,

double &out_trace_mean) -> Eigen::Matrix3d {

double cov_trace_mean_cam = 0.0;

const Eigen::Matrix3d cov_cam = ComputeCameraPointCovariance(pc, z, c, cov_trace_mean_cam);

// 旋转到体系

const Eigen::Matrix3d cov_body = R_body_cam_ * cov_cam * R_body_cam_.transpose();

out_trace_mean = cov_body.trace() / 3.0;

return cov_body;

};
```

#### 源点与目标点协方差的汇总

源点（当前帧）和目标点（关键帧）均有各自的位置协方差。取两者之和作为配对点的总位置不确定度：

$\boldsymbol{\Sigma}_{pos}^{total} = \boldsymbol{\Sigma}_{source}^B + \boldsymbol{\Sigma}_{target}^B$

**代码实现** ([state_estimater.cpp](../src/state_estimater.cpp) lines 177-184)：

```cpp
double src_var_mean = 0.0, tgt_var_mean = 0.0;

const Eigen::Matrix3d src_cov_body = point_cov_body(q_cam, z_src, src_conf, src_var_mean);

const Eigen::Matrix3d tgt_cov_body = point_cov_body(p_cam, z_tgt, tgt_conf, tgt_var_mean);

const Eigen::Matrix3d pos_cov_sum = src_cov_body + tgt_cov_body;



factor.src_pos_var = src_var_mean;

factor.tgt_pos_var = tgt_var_mean;

factor.pos_var_sum = factor.src_pos_var + factor.tgt_pos_var;
```

### 8.2c 残差方向的自适应测量噪声建模

基于点位置协方差，构建几何/光度方向的测量噪声模型。

#### 几何测量噪声

几何残差 $r_{geo}$ 的噪声方差包含两部分：

**1. 点位置不确定度在法向方向的投影**：

$$\sigma_{pos,geo}^2 = \hat{\mathbf{n}}^T \boldsymbol{\Sigma}_{pos}^{total} \hat{\mathbf{n}}$

**2. 距离和残差自适应项**（用于降权外点）：

$$\sigma_{dist}^2 = (1 + \|\mathbf{q} - \mathbf{p}\|^2) \cdot (1 + |r_{geo}|)$

综合的几何测量噪声：

$$\sigma_{geo}^2 = \sigma_{dist}^2 + \sigma_{pos,geo}^2$

**物理含义**：

- 远距离匹配点（$\|\mathbf{q} - \mathbf{p}\|^2$ 大）的权重降低，因为外点概率增大

- 残差大的点（当前线性化点不准确）的权重降低

- 点云本身不确定性低的区域获得高权重

**代码实现** ([state_estimater.cpp](../src/state_estimater.cpp) lines 200-207)：

```cpp
const double pos_var_geo = n_unit.dot(pos_cov_sum * n_unit);



// 几何噪声：随匹配距离和几何残差自适应增大，叠加点位置不确定度

const double geo_noise =

(1.0 + factor.match_dist2) * (1.0 + std::abs(factor.r_geo)) + pos_var_geo;



factor.inv_geo_noise = 1.0 / std::max(geo_noise, 1e-6);
```

#### 光度测量噪声

光度残差 $r_{color}$ 的噪声方差由：

**1. 点位置不确定度在梯度方向的投影**：

$$\sigma_{pos,color}^2 = (\mathbf{d}_p^{\perp})^T \boldsymbol{\Sigma}_{pos}^{total} (\mathbf{d}_p^{\perp})$

**2. 纹理强度自适应项**（梯度弱的区域光度约束弱）：

$$\sigma_{texture}^2 = \left(1 + \frac{1}{\|\mathbf{d}_p^{\perp}\|^2 + \epsilon}\right) \cdot (1 + |r_{color}|)$

综合的光度测量噪声：

$$\sigma_{color}^2 = \sigma_{texture}^2 + \sigma_{pos,color}^2$

**物理含义**：

- 梯度接近零的平坦区域（$\|\mathbf{d}_p^{\perp}\|^2 \to 0$）的光度约束接近无穷（权重趋零）

- 梯度强的纹理区域获得高权重

- 残差大的点权重降低

**代码实现** ([state_estimater.cpp](../src/state_estimater.cpp) lines 208-214)：

```cpp
const double pos_var_color = dpM.dot(pos_cov_sum * dpM);



// 光度噪声：纹理越弱（梯度≈0），光度约束越不可靠，噪声越大

const double color_texture = 1.0 + 1.0 / (factor.grad_strength2 + 1e-6);

const double color_noise = color_texture * (1.0 + std::abs(factor.r_color)) + pos_var_color;



factor.inv_color_noise = 1.0 / std::max(color_noise, 1e-6);
```

#### 信息权重转换

从测量噪声方差转换为信息权重（倒数）：

$$w_{geo}^{-1} = \sigma_{geo}^2, \quad w_{color}^{-1} = \sigma_{color}^2$

$$w_{geo} = \frac{1}{\max(\sigma_{geo}^2, \epsilon)}, \quad w_{color} = \frac{1}{\max(\sigma_{color}^2, \epsilon)}$

其中 $\epsilon = 10^{-6}$ 为数值稳定性的下限夹值。

**代码实现** ([state_estimater.cpp](../src/state_estimater.cpp) lines 215-217)：

```cpp
factor.inv_geo_noise = 1.0 / std::max(geo_noise, 1e-6);

factor.inv_color_noise = 1.0 / std::max(color_noise, 1e-6);
```

### 8.3 观测信息矩阵构建与加权融合

#### 信息矩阵和信息向量累加

将所有点对（几何+光度）的贡献累加，得到 $6 \times 6$ 观测信息矩阵和 $6 \times 1$ 观测信息向量：

$$
\mathbf{H}_{obs} = \sum_i \left[ \lambda \cdot w_{geo,i} \mathbf{J}_{geo,i}^T\mathbf{J}_{geo,i} + (1-\lambda) \cdot w_{color,i} \mathbf{J}_{color,i}^T\mathbf{J}_{color,i} \right]

$$

$$
\mathbf{b}_{obs} = \sum_i \left[ \lambda \cdot w_{geo,i} \mathbf{J}_{geo,i}^T r_{geo,i} + (1-\lambda) \cdot w_{color,i} \mathbf{J}_{color,i}^T r_{color,i} \right]

$$

其中：

- $\lambda \in [0,1]$：几何/光度融合系数
- $w_{geo,i} = 1/\sigma_{geo,i}^2$：第 $i$ 点几何噪声信息权重
- $w_{color,i} = 1/\sigma_{color,i}^2$：第 $i$ 点光度噪声信息权重

**代码实现** ([state_estimater.cpp](../src/state_estimater.cpp) lines 320-328)：

```cpp
const double w_geo_i = lambda > 0.0 ? lambda * factor.inv_geo_noise : 0.0;
const double w_color_i = (1.0 - lambda) > 0.0 ? (1.0 - lambda) * factor.inv_color_noise : 0.0;
H_pose.noalias() += w_geo_i * factor.J_geo * factor.J_geo.transpose()
                    + w_color_i * factor.J_color * factor.J_color.transpose();
b_pose.noalias() += w_geo_i * factor.J_geo * factor.r_geo 
                    + w_color_i * factor.J_color * factor.r_color;
```

#### 虚拟观测的等价化

加入 Tikhonov 阻尼项 $\epsilon\mathbf{I}$ 进行正则化，将观测信息转化为标准卡尔曼滤波中的等价噪声协方差：

$$
\mathbf{R}_{obs} = (\mathbf{H}_{obs}+\epsilon\mathbf{I})^{-1} \quad (6 \times 6)
$$

虚拟观测的残差向量：

$$
\mathbf{r}_v = -(\mathbf{H}_{obs}+\epsilon\mathbf{I})^{-1}\mathbf{b}_{obs} \quad (6 \times 1)
$$

这对应于标准卡尔曼更新中的"测量值"。

### 8.3b 创新协方差与卡尔曼增益

#### 创新协方差

根据矩阵反演引理，创新协方差（innovation covariance）为 $6 \times 6$ 矩阵：

$$
\mathbf{S} = \mathbf{H}_{sel}\mathbf{P}_{pred}\mathbf{H}_{sel}^T + \mathbf{R}_{obs}
$$

其中 $\mathbf{H}_{sel} \in \mathbb{R}^{6 \times 21}$ 为观测选择矩阵（前6列为单位矩阵，其余为0）。展开为：

$$
\mathbf{S} = \mathbf{P}_{pose} + \mathbf{R}_{obs}
$$

其中 $\mathbf{P}_{pose} = \mathbf{P}_{pred}[0:6, 0:6]$ 为预测协方差的位姿子块。

#### 卡尔曼增益（Kalman Gain）

卡尔曼增益矩阵（$21 \times 6$）控制观测对各状态变量的修正强度：

$$
\mathbf{K} = \mathbf{P}_{pred}\mathbf{H}_{sel}^T \mathbf{S}^{-1}
$$

其物理含义：

- $\mathbf{K}[0:3, :]$：旋转对观测残差的响应  
- $\mathbf{K}[3:6, :]$：平移对观测残差的响应
- $\mathbf{K}[6:9, :]$：速度通过交叉协方差的间接修正
- $\mathbf{K}[9:21, :]$：偏置、重力通过交叉协方差的间接修正

**代码实现** ([state_estimater.cpp](../src/state_estimater.cpp) lines 510-522)：

```cpp
const Eigen::Matrix<double, 6, 6> S = P_pose + R_obs; // 创新协方差
const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> S_ldlt(S);
K = P_pred_Ht * S_ldlt.solve(Eigen::Matrix<double, 6, 6>::Identity());
// P_pred_Ht = P_pred · H_selᵀ = P_pred[:, 0:6]  (21×6)
// K = (21×6) · (6×6)^{-1} = (21×6)
```

### 8.3c 迭代误差状态卡尔曼滤波（iEKF）更新

#### 累计误差状态与创新向量

ESIKF 在每次迭代中维护累计误差状态 $\delta\mathbf{x}^{prev}$（相对初始先验的累计修正）。在当前迭代中，计算创新向量：

$$
\text{innovation}^{(iter)} = \mathbf{r}_v - \mathbf{H}_{sel}\delta\mathbf{x}^{(iter-1)}
$$

其中 $\mathbf{H}_{sel}\delta\mathbf{x}^{(iter-1)}$ 为上一轮累计误差在观测子空间中的投影。

#### 状态增量更新

当前迭代得到的新累计误差状态（$21 \times 1$）：

$$
\delta\mathbf{x}^{(iter)} = \mathbf{K} \cdot \text{innovation}^{(iter)}
$$

本次真正注入名义状态的增量：

$$
\Delta\mathbf{x}^{(iter)} = \delta\mathbf{x}^{(iter)} - \delta\mathbf{x}^{(iter-1)}
$$

#### 流形上的状态更新

将增量以右扰动的形式注入名义状态 $\hat{\mathbf{x}}$：

$$
\hat{\mathbf{R}}^{(iter)} = \hat{\mathbf{R}}^{(iter-1)} \cdot \mathrm{Exp}(\Delta\boldsymbol\theta^{(iter)})
$$

$$
\hat{\mathbf{p}}^{(iter)} = \hat{\mathbf{p}}^{(iter-1)} + \Delta\mathbf{p}^{(iter)}
$$

其他状态量直接相加。

**代码实现** ([state_estimater.cpp](../src/state_estimater.cpp) lines 530-543)：

```cpp
const Eigen::Matrix<double, 6, 1> innovation = r_v - H_sel * dx_prev;
const Eigen::Matrix<double, DIM_STATE, 1> dx_new = K * innovation;
const Eigen::Matrix<double, DIM_STATE, 1> delta_x = dx_new - dx_prev;
dx_prev = dx_new;

// 流形更新
imu_states.rot_end = imu_states.rot_end * svio_utils::ExpSO3(delta_x.segment<3>(0));
imu_states.pos_end += delta_x.segment<3>(3);
imu_states.vel_end += delta_x.segment<3>(6);
imu_states.bias_g += delta_x.segment<3>(9);
imu_states.bias_a += delta_x.segment<3>(12);
imu_states.gravity += delta_x.segment<3>(15);
```

#### 收敛判定

当状态增量极小（$|\Delta\mathbf{x}| < 10^{-4}$）或达到最大迭代次数时停止迭代。

### 8.3d 后验协方差更新

所有迭代完成后，使用 Joseph 形式的协方差更新以保证数值稳定性和正定性：

$$
\mathbf{P}_{post} = (\mathbf{I} - \mathbf{K}\mathbf{H}_{sel})\mathbf{P}_{pred}(\mathbf{I} - \mathbf{K}\mathbf{H}_{sel})^T + \mathbf{K}\mathbf{R}_{obs}\mathbf{K}^T
$$

此形式无需直接求 $21 \times 21$ 矩阵逆，而仅需 $6 \times 6$ 分解，更加数值稳定。

**代码实现** ([state_estimater.cpp](../src/state_estimater.cpp) lines 555-560)：

```cpp
const Eigen::Matrix<double, DIM_STATE, DIM_STATE> I_KH = 
  Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Identity() - K * H_sel;
imu_states.cov = I_KH * P_pred * I_KH.transpose() 
                 + K * R_obs * K.transpose(); // Joseph form
```

### 8.4 ESIKF 坐标变换综合流程图

整个 ESIKF 闭环中，坐标系的变换关系可总结为下图：

```
当前帧点云 (当前 IMU 体系 B)
          |
          | [R_curr_to_prevkf, t_curr_in_prevkf]
          v
    关键帧体系中的点 (关键帧 IMU 体系 K)
          |
          +─────────────────────────────────────────┐
          |                                         |
          v                                         v
    相对位姿 (K←B)                          残差计算 (K 中)
    Iterative Update                         r_geo, r_color
          |                                         |
          | δx = K · (r_v - H_sel·δx_prev)         |
          |                                         |
          v                                         |
    流形更新：R_B ← R_B · Exp(δθ_b)              |
             p_B ← p_B + δp_w (混合坐标)        |
             ...                                    |
          |                                         |
          | 收敛? (||δx|| < ε)                      |
        ┌─┴─┐                                       |
    否  |    是                                      |
        |    |                                      |
        └────┼──────────────────────────────────────┘
             v
    后验协方差 P_post
    (Joseph 形式，21×21)
```

**关键坐标变换点**：

1. **输入**：点云 $\mathbf{p}_b \in \mathbb{R}^3$（当前 IMU 体系 B）
2. **第一步重线性化**：变换到关键帧体系 K  
  $$\mathbf{q} = \mathbf{R}_{k \leftarrow b}^{curr} \mathbf{p}_b + \mathbf{t}_{b \text{ in } k}^{curr}$$
3. **几何/光度残差计算**：所有运算在关键帧体系 K 中
   - 源点：$\mathbf{q} \in K$
   - 目标法向：$\hat{\mathbf{n}} \in K$（预计算）
   - 残差：$r_{geo} = (\mathbf{q} - \mathbf{p})^T\hat{\mathbf{n}}$
4. **Jacobian 中的混合坐标**：
  - 旋转增量是当前机体系 B 下的右扰动，但用于构造雅可比的 $\mathbf{q}_{rel}$ 与 $\hat{\mathbf{n}}$ 都表达在关键帧体系 K 中  
  - 平移梯度在世界系 W 中，通过 $\mathbf{R}_{wk} \hat{\mathbf{n}}$ 对齐到误差状态定义
5. **累计信息矩阵**：$\mathbf{H}_{obs} = \sum_i \mathbf{J}_i^T w_i \mathbf{J}_i$（全为 $6 \times 6$）
6. **虚拟观测**：等价化为标准卡尔曼观测
7. **状态更新**：以右扰动形式应用到 SE(3) 流形上
8. **输出**：新的后验状态 $\hat{\mathbf{x}}_{post}$ 和协方差 $\mathbf{P}_{post}$

### 8.5 ESIKF 与 ESKF 对比

| 项目         | ESKF（松耦合 GICP）            | ESIKF（紧耦合几何+光度）    |
| ---------- | ------------------------- | ------------------ |
| **触发场景**   | 普通帧（高速运动）                 | 关键帧（需要精确约束）        |
| **观测类型**   | 6-DOF 刚体位姿（GICP 输出）       | 几何+光度混合残差（点级别）     |
| **观测噪声模型** | 协方差对角线化简 $\\Sigma_{gicp}$ | 各点自适应噪声（距离+梯度+置信度） |
| **迭代**     | 无（观测一次更新）                 | 5 次以内迭代重线性化        |
| **坐标系要求**  | 观测转世界系、协方差对齐              | 所有计算在关键帧系 K 中      |
| **稳定性**    | 依赖 GICP 相对位姿质量            | 多约束融合，收敛稳定         |
| **计算复杂度**  | 较低（单次矩阵运算）                | 中等（迭代+Jacobian 计算） |

**关键取舍**：

- ESKF：适合低纹理、快速运动场景（GICP 是相对位姿的有效约束）
- ESIKF：适合高纹理、缓速运动场景（几何+光度双重约束提升精度）

---

## 9. 动态点剔除机制

为了减弱动态物体对几何配准的影响，系统在状态更新前引入动态点剔除模块。其核心思想是利用先验相对位姿将关键帧点云投影到当前图像平面，生成几何光流预测，再与 RAFT 网络输出的真实光流进行对比。如果两者差异显著，则说明对应区域可能不满足静态场景假设，需从目标点云中剔除。最终保留下来的静态背景点云将被用于后续的 GICP 或 ESIKF 更新，从而提升整体估计鲁棒性。

---

## 10. 关键帧策略

前端关键帧和后端关键帧采用两级管理机制。前端关键帧主要依据平移、旋转、时间间隔及速度变化等指标进行触发，用于保证短时间尺度上的观测稳定性；后端关键帧则在此前基础上采用更稀疏的晋升策略，以控制后端优化规模和地图维护频率。这一设计兼顾了前端实时性与后端优化效率。

---

## 11. 结论

综上，SIO_SLAM 前端本质上是一套以 IMU 预测为先验、以双目点云为观测、以误差状态滤波为核心融合框架的视觉惯性前端。其数学上最关键的部分在于：一方面保持 IMU 传播、视觉残差和状态注入之间的误差定义一致；另一方面确保来自 GICP 的位姿和协方差在进入滤波器之前完成正确的坐标变换与协方差对齐。只要这一一致性得到保证，系统的前端估计链条便能够在动态场景中维持较好的收敛性与稳定性。

### 13.2 数学一致性

阅读和修改本工程时，最重要的四点是：

1. 预测协方差不是纯世界系，也不是纯机体系，而是混合误差坐标；
2. GICP 相对位姿必须先转成世界系绝对观测后才能送入 ESKF；
3. GICP 协方差必须先做坐标对齐后才能和预测协方差相加；
4. ESIKF 中几何和光度雅可比，也必须严格按照右扰动和混合误差定义构造。

只要这四点保持一致，整条前端估计链在数学上就是自洽的。
