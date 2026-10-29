# 平移 / 旋转误差对几何光流的影响（解析推导）

本文不依赖具体帧号或实测序列，从针孔模型与刚体相对位姿一阶展开，推导**位姿误差如何变成像素光流误差**，并给出可换算的量级关系。实现对照：[`flow_pose_sensitivity_model.py`](../flow_pose_sensitivity_model.py) 中的 `linear_flow_delta`；坐标系约定见 [`coordinate_frames.md`](coordinate_frames.md)。

---

## 1. 问题设定

源帧深度反投影得相机系点 $\mathbf{p}_0=(x,y,Z)^\top$，经相对位姿 $T_{\mathrm{target}\leftarrow\mathrm{source}}=(R,\mathbf{t})$：

$$
\mathbf{p}_1 = R\mathbf{p}_0 + \mathbf{t},\qquad
(u',v')=\pi(\mathbf{p}_1),\qquad
\mathbf{f}=(u'-u,\,v'-v).
$$

针孔投影（OpenCV / ZED 左目）：

$$
u = f_x\frac{x}{Z}+c_x,\qquad
v = f_y\frac{y}{Z}+c_y.
$$

关心：相对位姿存在误差 $\boldsymbol{\omega}$（小旋转，rad）与 $\delta\mathbf{t}$（小平移，m）时，几何光流相对真值的一阶变化 $\delta\mathbf{f}=(\delta u,\delta v)$。

### 1.1 坐标系

扰动均在**左目相机系 C**（$X$ 右、$Y$ 下、$Z$ 前）下表达。代码标签 roll / pitch / yaw 表示绕相机 $X/Y/Z$，**不是**机体航空角。

| 代码名 | 相机轴 | 近光心主导光流 |
|--------|--------|----------------|
| roll | 绕 $+X$ | $\delta v \approx -f_y\omega_x$（竖直流） |
| pitch | 绕 $+Y$ | $\delta u \approx +f_x\omega_y$（水平流） |
| yaw | 绕 $+Z$ | 绕光心旋转场；中心弱 |

### 1.2 符号表（全文变量含义）

#### 像素与内参

| 符号 | 单位 | 含义 |
|------|------|------|
| $(u,v)$ | px | 源帧像素坐标；$u$ 向右，$v$ 向下 |
| $(u',v')$ | px | 经相对位姿变换后，投影到目标帧的像素坐标 |
| $f_x,f_y$ | px | 焦距（像素单位） |
| $c_x,c_y$ | px | 主点（光轴与像面交点的像素坐标） |
| $f$ | px | 量级分析时取 $f\approx f_x\approx f_y$ |
| $\pi(\cdot)$ | — | 针孔投影：3D 点 → 像素 |
| $\pi^{-1}(u,v,Z)$ | — | 反投影：像素 + 深度 → 相机系 3D 点 |

#### 3D 点与相对位姿

| 符号 | 单位 | 含义 |
|------|------|------|
| $\mathbf{p}_0=(x,y,Z)^\top$ | m | **源相机系**下的 3D 点 |
| $x$ | m | 点在源相机系的横向坐标（右为正） |
| $y$ | m | 点在源相机系的竖向坐标（下为正） |
| $Z$ | m | 点在源相机系的深度（前 / 光轴为正）；亦作特征深度 |
| $\mathbf{p}_1$ | m | **目标相机系**下的 3D 点 |
| $T_{\mathrm{target}\leftarrow\mathrm{source}}$ | — | 源→目标的相对位姿齐次变换 |
| $R$ | — | 相对旋转（$3\times 3$），$\mathbf{p}_1$ 中的旋转部分 |
| $\mathbf{t}=(t_x,t_y,t_z)^\top$ | m | 相对平移（目标系下，源光心相对目标光心的位移，视实现约定） |
| $\mathbf{f}=(u'-u,\,v'-v)$ | px | 几何光流（像素位移） |

#### 归一化像平面

| 符号 | 单位 | 含义 |
|------|------|------|
| $x_n=x/Z=(u-c_x)/f_x$ | 无量纲 | 归一化坐标：$u$ 方向相对光轴的横向张角近似（$\tan$） |
| $y_n=y/Z=(v-c_y)/f_y$ | 无量纲 | 归一化坐标：$v$ 方向；光心处为 $0$，越靠画面边缘越大 |
| $x_n^2,\,y_n^2,\,x_n y_n$ | 无量纲 | 由 $x_n,y_n$ 构成的二次项，出现在旋转流场的**边缘修正**中 |

#### 位姿误差（本文关心的扰动）

| 符号 | 单位 | 含义 |
|------|------|------|
| $\boldsymbol{\omega}=(\omega_x,\omega_y,\omega_z)^\top$ | rad | 相对旋转的小误差（绕源相机 $X/Y/Z$） |
| $\omega_x$ | rad | 绕相机 $+X$（右）的旋转误差；代码名 roll |
| $\omega_y$ | rad | 绕相机 $+Y$（下）的旋转误差；代码名 pitch |
| $\omega_z$ | rad | 绕相机 $+Z$（光轴）的旋转误差；代码名 yaw |
| $\theta$ | rad | 量级分析时某一轴旋转误差的标量大小 |
| $\theta_{\mathrm{deg}}$ | ° | $\theta$ 的角度制：$180\theta/\pi$ |
| $\delta\mathbf{t}=(\delta t_x,\delta t_y,\delta t_z)^\top$ | m | 相对平移的小误差（沿源相机 $X/Y/Z$） |
| $\delta t_x$ | m | 沿相机右方向的平移误差 |
| $\delta t_y$ | m | 沿相机下方向的平移误差 |
| $\delta t_z$ | m | 沿光轴前向的平移误差 |
| $\delta t_\perp$ | m | 横向平移误差（$\delta t_x$ 或 $\delta t_y$ 的量级代表） |
| $[\boldsymbol{\omega}]_\times$ | — | $\boldsymbol{\omega}$ 的反对称矩阵，满足 $[\boldsymbol{\omega}]_\times\mathbf{p}=\boldsymbol{\omega}\times\mathbf{p}$ |
| $\delta R$ | — | 旋转误差对应的旋转增量；一阶 $\delta R\approx I+[\boldsymbol{\omega}]_\times$ |
| $I$ | — | $3\times 3$ 单位阵 |

#### 3D / 像素一阶变化量

| 符号 | 单位 | 含义 |
|------|------|------|
| $\delta\mathbf{p}=(\delta x,\delta y,\delta Z)^\top$ | m | 位姿误差引起的 3D 点一阶位移（源相机系） |
| $\delta x,\delta y,\delta Z$ | m | $\delta\mathbf{p}$ 的三个分量 |
| $(\delta u,\delta v)$ | px | 几何光流相对真值的一阶变化（像素） |
| $\delta\mathbf{f}=(\delta u,\delta v)$ | px | 同上，向量写法 |
| $\delta u_{\mathrm{t}},\delta v_{\mathrm{t}}$ | px | **仅平移误差**引起的光流变化 |
| $\delta u_{\mathrm{r}},\delta v_{\mathrm{r}}$ | px | **仅旋转误差**引起的光流变化 |
| $\mathrm{EPE}$ | px | 端点误差 $\sqrt{(\delta u)^2+(\delta v)^2}$ |

#### 深度误差与不确定度（§4–§5）

| 符号 | 单位 | 含义 |
|------|------|------|
| $\delta Z$（深度误差） | m | 深度测量误差（与上面位移分量同形异义，见 §4 语境） |
| $\varepsilon=\delta Z/Z$ | 无量纲 | 相对深度误差 |
| $\mathbf{p}_0'$ | m | 深度有偏后的源点 $(1+\varepsilon)\mathbf{p}_0$ |
| $\alpha$ | rad | 半视场角（量级：边缘像素对应的 $|x_n|$ 或 $|y_n|$） |
| $\sigma_\omega$ | rad | 旋转不确定度（标准差量级） |
| $\sigma_t$ | m | 平移不确定度 |
| $\sigma_f$ | px | 映射到像素光流的不确定度 |

---

## 2. 一阶推导

### 2.1 3D 位移场

旋转增量与 3D 点位移（$\delta R$：旋转误差矩阵；$[\boldsymbol{\omega}]_\times$：$\boldsymbol{\omega}$ 的反对称阵；$\delta\mathbf{p}$：点的一阶位移）：

$$
\delta R \approx I + [\boldsymbol{\omega}]_\times,\qquad
\delta\mathbf{p} \approx \boldsymbol{\omega}\times\mathbf{p}_0 + \delta\mathbf{t}.
$$

分量（$\delta x,\delta y,\delta Z$：源相机系下点的位移分量）：

$$
\begin{aligned}
\delta x &= \omega_y Z - \omega_z y + \delta t_x, \\
\delta y &= \omega_z x - \omega_x Z + \delta t_y, \\
\delta Z &= \omega_x y - \omega_y x + \delta t_z.
\end{aligned}
$$

本推导将 $(\boldsymbol{\omega},\delta\mathbf{t})$ 作用在**源相机系**点 $\mathbf{p}_0$ 上，并用源深度 $Z$ 做投影微分（相对位姿增量的一阶近似）。

### 2.2 投影微分

像素变化由 3D 位移投影而来（$\delta u,\delta v$：光流误差；$x_n=x/Z$，$y_n=y/Z$）：

$$
\delta u = \frac{f_x}{Z}\delta x - f_x\frac{x}{Z^2}\delta Z
= f_x\left(\frac{\delta x}{Z} - x_n\frac{\delta Z}{Z}\right),
$$

$$
\delta v = \frac{f_y}{Z}\delta y - f_y\frac{y}{Z^2}\delta Z
= f_y\left(\frac{\delta y}{Z} - y_n\frac{\delta Z}{Z}\right).
$$

含义：横向位移 $\delta x,\delta y$ 按 $f/Z$ 放大成像素；前向位移 $\delta Z$ 经射线方向 $(x_n,y_n)$ 耦合进 $(\delta u,\delta v)$（光心 $x_n=y_n=0$ 时前向项消失）。

### 2.3 仅平移

令 $\boldsymbol{\omega}=\mathbf{0}$，则 $\delta x=\delta t_x$，$\delta y=\delta t_y$，$\delta Z=\delta t_z$：

$$
\begin{aligned}
\delta u_{\mathrm{t}} &= \frac{f_x}{Z}(\delta t_x - x_n\,\delta t_z), \\
\delta v_{\mathrm{t}} &= \frac{f_y}{Z}(\delta t_y - y_n\,\delta t_z).
\end{aligned}
$$

| 误差分量 | 流场 | 深度依赖 | 光心处 |
|----------|------|----------|--------|
| $\delta t_x$ | 水平平移场 | $\propto 1/Z$ | $\delta u=f_x\delta t_x/Z$ |
| $\delta t_y$ | 竖直平移场 | $\propto 1/Z$ | $\delta v=f_y\delta t_y/Z$ |
| $\delta t_z$ | 径向缩放场 | $\propto 1/Z$ | **为 0**；边缘 $\propto |x_n|$、$|y_n|$ |

### 2.4 仅旋转

令 $\delta\mathbf{t}=\mathbf{0}$。$\delta u_{\mathrm{r}},\delta v_{\mathrm{r}}$ 为旋转误差单独引起的光流变化：

$$
\begin{aligned}
\delta u_{\mathrm{r}} &= -f_x\left(
  \frac{xy}{Z^2}\omega_x
  - \left(1+\frac{x^2}{Z^2}\right)\omega_y
  + \frac{y}{Z}\omega_z
\right), \\
\delta v_{\mathrm{r}} &= -f_y\left(
  \left(1+\frac{y^2}{Z^2}\right)\omega_x
  - \frac{xy}{Z^2}\omega_y
  - \frac{x}{Z}\omega_z
\right).
\end{aligned}
$$

用归一化坐标 $x_n=x/Z$、$y_n=y/Z$ 重写（二次项 $x_n^2$ 等为边缘修正）：

$$
\begin{aligned}
\delta u_{\mathrm{r}} &= f_x\big(
  -x_n y_n\,\omega_x + (1+x_n^2)\,\omega_y - y_n\,\omega_z
\big), \\
\delta v_{\mathrm{r}} &= f_y\big(
  -(1+y_n^2)\,\omega_x + x_n y_n\,\omega_y + x_n\,\omega_z
\big).
\end{aligned}
$$

近光心（$x_n,y_n\approx 0$，二次项可忽略）主项：

| 误差分量 | 含义 | 主导光流 | 深度依赖 |
|----------|------|----------|----------|
| $\omega_x$ | 绕相机右轴（抬头/低头） | $\delta v \approx -f_y\omega_x$ | **无关** |
| $\omega_y$ | 绕相机下轴（左右看） | $\delta u \approx +f_x\omega_y$ | **无关** |
| $\omega_z$ | 绕光轴（画面滚转） | $\delta u \approx -f_x y_n\omega_z$，$\delta v \approx +f_y x_n\omega_z$ | 中心弱；边缘随视场角增大 |

### 2.5 合成（经典刚体光流一阶式）

平移贡献 + 旋转贡献线性叠加（$\delta u,\delta v$：总光流误差）：

$$
\begin{aligned}
\delta u &\approx \frac{f_x}{Z}(\delta t_x - x_n\delta t_z)
  + f_x\big(-x_n y_n\,\omega_x + (1+x_n^2)\,\omega_y - y_n\,\omega_z\big), \\
\delta v &\approx \frac{f_y}{Z}(\delta t_y - y_n\delta t_z)
  + f_y\big(-(1+y_n^2)\,\omega_x + x_n y_n\,\omega_y + x_n\,\omega_z\big).
\end{aligned}
$$

小误差下旋转与平移贡献**线性叠加**。

---

## 3. 量级与等价换算（与具体帧无关）

取近光心、焦距 $f\approx f_x\approx f_y$、特征深度 $Z$（场景代表深度，可用深度中位数）。

### 3.1 灵敏度

下表中 $\theta$：某轴旋转误差（rad）；$\delta t_\perp$：横向平移误差（m）；$\alpha$：半视场角（rad，约等于边缘处 $|x_n|$ 或 $|y_n|$）。

| 误差类型 | 特征 EPE（量级） | 单位换算 |
|----------|------------------|----------|
| 绕 $X/Y$ 旋转 $\theta$（rad） | $\sim f\,\theta$ | $\sim f\cdot(\pi/180)$ **px/°** |
| 绕 $Z$ 旋转 | 中心 $\sim 0$；半视场角 $\alpha$ 处 $\sim f\alpha\,\theta$ | 通常弱于绕 $X/Y$ |
| 横向平移 $\delta t_\perp$ | $\sim f\,\delta t_\perp / Z$ | $\sim f/(100Z)$ **px/cm** |
| 前向平移 $\delta t_z$ | 中心 $\sim 0$；边缘 $\sim f\,\|x_n\|\,\delta t_z/Z$ | 弱于同量级横向平移 |

因此灵敏度排序（近光心）：

$$
\omega_y,\ \omega_x \;\gg\; \delta t_x,\ \delta t_y \;\gg\; \omega_z,\ \delta t_z
\quad\text{（同“看起来不大”的物理单位下）}.
$$

### 3.2 旋转–平移等价

使绕 $Y$（或 $X$）旋转 $\theta$（rad）与横向平移 $\delta t_\perp$ 产生同量级光流误差：

$$
f\,\theta \approx \frac{f\,\delta t_\perp}{Z}
\quad\Rightarrow\quad
\delta t_\perp \approx Z\,\theta
\quad\Rightarrow\quad
\theta_{\mathrm{deg}} \approx \frac{180}{\pi}\cdot\frac{\delta t_\perp}{Z}.
$$

或：

$$
1^\circ\ \text{旋转}
\;\Leftrightarrow\;
\delta t_\perp \approx Z\cdot\frac{\pi}{180} \approx 0.0175\,Z.
$$

| 特征深度 $Z$ | $1^\circ$ 等价横向平移 |
|--------------|----------------------|
| 1 m | ≈ 1.7 cm |
| 3 m | ≈ 5.3 cm |
| 5 m | ≈ 8.7 cm |
| 10 m | ≈ 17.5 cm |

**结论**：同样“一度 vs 一厘米”，旋转误差对光流的破坏通常远大于平移；深度越大，要匹配 $1^\circ$ 所需的平移越大，即远景下平移误差更“钝”。

### 3.3 随深度的定性行为

| 误差源 | 随 $Z$ 增大 |
|--------|-------------|
| 绕 $X/Y$ 旋转 | EPE **几乎不变** |
| 横向平移 | EPE $\propto 1/Z$，**远处钝感** |
| 前向平移 / 绕 $Z$ | 中心弱；边缘依赖视场，整体仍偏弱 |

同一套位姿误差：近处平移主导相对更明显；远处更接近“旋转误差决定光流误差”。

---

## 4. 深度测量误差的耦合（补充）

即使相对位姿 $(R,\mathbf{t})$ 正确，深度测量误差也会改几何光流。记 $\delta Z$ 为深度误差（m），$\varepsilon=\delta Z/Z$ 为相对深度误差（无量纲），$\mathbf{p}_0'$ 为有偏深度下的源点：

$$
\mathbf{p}_0' = \mathbf{p}_0 + \frac{\delta Z}{Z}\mathbf{p}_0
= (1+\varepsilon)\mathbf{p}_0,\quad \varepsilon=\delta Z/Z.
$$

经 $R,t$ 后再投影，光流相对真值的差在一阶上与**视差 / 平移诱导流**同阶耦合：深度误差主要污染**平移相关**的几何光流，对纯旋转场影响较弱。量级上，相对深度误差 $\varepsilon$ 引起的光流扰动大致与 $\varepsilon$ 乘以“平移诱导光流幅度”同阶，且近处更大。

（数值扫参见脚本输出的 `flow_sensitivity_depth_error.csv`。）

---

## 5. 对估计与融合的含义

1. **用几何光流 / 光流差分诊断位姿时**：优先怀疑旋转误差（尤其绕相机 $Y/X$）；平移误差在近处才显眼。
2. **IMU vs ICP**：若旋转已准、平移漂，光流差分可仍呈“平移型”残差（$\propto 1/Z$）；若旋转差 $0.5^\circ\sim 1^\circ$，常压过厘米级平移差。
3. **融合策略（与灵敏度一致）**：光流灵敏度高 = 视觉对该轴信息量大 → **旋转（尤其 $\omega_y,\omega_x$）更信 ICP/点云**；**平移（尤其 $t_z$）更信 IMU**。详见 [`icp_axis_precision.md`](icp_axis_precision.md) §4。勿把「pitch 估错时 EPE 大」理解成「融合时少信视觉 pitch」。
4. **加权 / 不确定度映射**：旋转 $\sigma_\omega$ → 像素约 $\sigma_f\sim f\sigma_\omega$（近光心）；平移 $\sigma_t$ → $\sigma_f\sim f\sigma_t/Z$。近处应对平移噪声更大权惩罚；远处光流对旋转误差更“显眼”，恰说明旋转应靠视觉约束收紧。

---

## 6. 适用条件与局限

| 假设 | 说明 |
|------|------|
| 小扰动 | $\|\boldsymbol{\omega}\|\ll 1$，$\|\delta\mathbf{t}\|$ 远小于场景尺度；大误差需全量重投影 |
| 针孔、无畸变 | 与仓库 `compute_flow_from_depth` 一致；强畸变需先矫正 |
| 源系一阶模型 | 增量作用在 $\mathbf{p}_0$；基准相对旋转很大时，与“在目标系扰动”有高阶差 |
| 静态场景 / 真深度 | 动态物体与错误深度会额外贡献，不在本式内 |

小扰动下，一阶 EPE 与全量重投影 EPE 相对误差通常：旋转 $\lesssim 5\%$，平移 $\lesssim 10\%$（见脚本对照）。

---

## 7. 相关代码与文档

| 内容 | 路径 |
|------|------|
| 一阶实现 | `flow_pose_sensitivity_model.py` → `linear_flow_delta` |
| 全量几何光流 | `imu_pointcloud_optical_flow.py` → `compute_flow_from_depth` |
| 坐标系总览 | `docs/coordinate_frames.md` §9.4 |
| 旋转误差 → 光流权重管线 | `docs/rot_ferr_flow_pipeline.md` |
| 由灵敏度反推 ICP 各轴精度 | [`icp_axis_precision.md`](icp_axis_precision.md) |
| 动态观测滤波中的量级与门控策略 | [`SCI_动态观测滤波.md`](SCI_动态观测滤波.md) §3.4–§3.6 |

数值验证（可选，需指定数据与帧）：

```bash
python flow_pose_sensitivity_model.py data --source-frame <src> --target-frame <tgt>
```
