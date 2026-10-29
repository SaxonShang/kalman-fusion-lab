# SIO_SLAM 五篇 SCI 论文索引

五篇相互区分、形成闭环：前端三篇各自独立，后端一篇，系统集成一篇。理论与雅可比推导均写在各篇第 3 节。

| # | 文件 | 主题 |
|---|------|------|
| 1 | [SCI_动态观测滤波.md](SCI_动态观测滤波.md) | 位姿不确定度驱动的几何–神经光流动态滤波（雅可比 §3.4） |
| 2 | [SCI_投影StereoGICP.md](SCI_投影StereoGICP.md) | 立体点协方差与投影关联 Stereo-GICP（雅可比 §3.5） |
| 3 | [SCI_ESKF与ESIKF.md](SCI_ESKF与ESIKF.md) | 松耦合 ESKF 与紧耦合几何-光度 ESIKF（雅可比 §3.2–3.5） |
| 4 | [SCI_3DGS_IMU后端.md](SCI_3DGS_IMU后端.md) | 以 IMU 运动先验稳定的 3DGS 位姿优化（预积分先验 + 优化中 IMU 测量项） |
| 5 | [SCI_系统集成.md](SCI_系统集成.md) | 统一双目-惯性-3DGS 系统与协同收益 |

## 拆分原则

1. **动态观测滤波**：只做观测净化与静态索引。
2. **投影 Stereo-GICP**：只做立体协方差、投影关联、马氏点到点与左 SE(3)。
3. **ESKF 与 ESIKF**：只做混合误差、松/紧耦合、$J_{g\to e}$、$J_{\mathrm{geo}}$ 与虚拟观测；几何细节引用第 2 篇。
4. **3DGS-IMU 后端**：以 IMU 预积分为运动先验，在 3DGS 光度优化中加入 IMU 一致性测量项以稳定位姿；不重复前端动态/滤波细节。
5. **系统集成**：接口、异步调度、组合消融与端到端收益。

## 建议投稿顺序

Stereo-GICP → ESKF/ESIKF → 动态滤波 → 3DGS-IMU → 系统集成

## 模块关系

```text
双目+IMU → 动态滤波(S) → Stereo-GICP → ESKF/ESIKF → 3DGS-IMU后端
                                              └──────────→ 系统集成验证
```
