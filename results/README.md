# 归档结果与输入

只需理解和评议实验时，先读 [实验审阅说明](../docs/EXPERIMENT_REVIEW_BRIEF.md)，再看下表中的汇总 CSV 与图；不需要配置环境或重跑。

共 50 个执行目录、2350 条指标记录，包含历史阶段、回放和调试；重复记录不代表独立样本。

| 路径 | 内容 |
|---|---|
| diagnostics/20261029_105246/ | 当前三组协方差对照；从 report.html、comparison.png 或 summary.csv 开始 |
| fusion/ | 全部原始逐次结果、配置、完整 P/R/K 和迭代矩阵 |
| RUNS.csv | 实验阶段、视差、配准版本、完成状态和结果清单 |
| controls.json | 可跨机器定位的六条对照索引 |
| inputs/ | 六条序列的 GT、IMU、帧索引、标定和转换元数据 |
| disparity/ | 54 帧实际模型输出及缓存校验；可在 CPU 重跑点云和配准 |
| recording/ | 原始标定、时间/真值检查报告和使用说明 |
| models/ | 原始模型身份、接口和图转换验证 |
| logs/ | 环境、构建、测试及实验执行日志，含失败尝试 |
| provenance/ | 原工程基点、补丁、两阶段旧实现源码 |
| SHA256SUMS | 导出文件完整性校验，包含代码与结果 |

原始观测与结果按字节保存，保留历史绝对路径；请用 [复现指南](../docs/REPRODUCING.md) 中的入口。新执行写入 artifacts/，不覆盖这里。

发布前独立编译、93 项测试、跨目录回放与单组全模块验证见 [publication-validation.json](provenance/publication-validation.json)；单组验证 CSV 与配置在 `validation/cached_disparity_pipeline/`，不计入上述 50 个研究执行目录。
