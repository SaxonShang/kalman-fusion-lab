# FR5 高速实验数据接入

本机原始数据位于 `data/recordings/recordings_highspeed/`。格式与时间约定已对照随数据提供的 `使用说明.md`、标定文件，以及[录制仓库 high-speed 分支](https://github.com/SaxonShang/Fr5_recording/tree/high-speed)。

## 准备与选择序列

在项目根目录执行：

```bash
python3 scripts/prepare_fr5_dataset.py --activate up
```

脚本需要 Python 3.10+ 和 PyYAML（Ubuntu 的 `python3-yaml`），不需要 ZED SDK 或 TensorRT 模型。默认生成全部六条序列的兼容索引，以 `up` 为当前运行数据集。原始 CSV、标定、真值与图像均保留；生成目录的 `left/`、`right/` 是相对符号链接，不复制约 4.5 GB 图像。

```text
data/datasets/fr5_highspeed/<序列名>/
├── left/ -> 原始 save_data/left/
├── right/ -> 原始 save_data/right/
├── frames.csv              DatasetLoader 已支持的分帧索引
├── imu.csv                 按帧分组并补偿时间的 IMU
├── groundtruth_tum.txt      保留帧对应的已修正左目真值
├── params.yaml             该相机的算法配置
└── metadata.json           原始路径、时间补偿、标定哈希与检查结果
```

`--activate` 只更新被 Git 忽略的 `project_config.local.yaml` 中的 `dataset_dir`、`imu_file`、`params_file`，保留模型及 TensorRT 路径。可选 `up`、`forward`、`right`、`yaw_left`、`pitch_up`、`roll_right` 或完整目录名，例如：

```bash
python3 scripts/prepare_fr5_dataset.py --activate yaw_left
source env.sh
bash scripts/check_env.sh gpu
```

省略 `--activate` 只生成索引。`--source` 和 `--dest` 可指定其他原始数据根目录和输出根目录。脚本仅覆盖带有自身生成标记的目录，检查全部输入后才写入输出。原始目录必须与生成目录分开；重复执行会重新生成索引。

默认跳过前 60 帧（约 2 s）的曝光、亮度和白平衡过渡段；`--skip-frames` 可调整。索引中的帧号从 0 开始，额外的 `source_frame_id` 保存原始帧号。第一个区间从原始第 59 帧开始，因此不会把启动时缓存的早期 IMU 堆进第一帧。之后约 1 s 即可收集初始化所需的 200 个样本，仍处于第一次同步摆动前的静止段。

| 序列 | 原始图像对 | 保留图像对 | 保留 IMU | 图像到 IMU 的 td（ms） |
|---|---:|---:|---:|---:|
| `20261015_112553_up` | 697 | 637 | 3980 | +4.891 |
| `20261015_112659_forward` | 683 | 623 | 3884 | +4.711 |
| `20261015_112757_right` | 697 | 637 | 3983 | +4.188 |
| `20261015_112845_yaw_left` | 680 | 620 | 3884 | +4.872 |
| `20261015_112940_pitch_up` | 687 | 627 | 3900 | +4.587 |
| `20261015_113028_roll_right` | 742 | 682 | 4253 | +4.859 |
| 合计 | 4186 | 3826 | 23884 | |

## 时间和单位

原始图像文件名与 `img_timeline.csv` 的图像时间是整数毫秒，IMU 的真实采样时间是 `zed_imu_timestamp_ns` 纳秒列。原始 `zed_imu.csv` 的第二列仅表示图像分组，不能作为 IMU 采样时间。

录制检查程序以 `gyro(t_image - lag)` 匹配图像运动，报告中的 `lag_ms` 为负值。图像内容实际对应 IMU 时间轴上晚约 4.7 ms 的时刻，即 `t_image + td = t_imu`，其中 `td = -lag_ms / 1000`。

为了保持图像、轨迹输出和现成真值的时间标签一致，转换采用等价方式：

```text
t_image_aligned = t_image_raw
t_imu_aligned   = t_imu_raw + lag_ms / 1000
```

补偿值按各序列报告计算并四舍五入到纳秒，再用整数纳秒划分 `(上一帧时间, 当前帧时间]`。每个保留 IMU 样本只出现一次。转换后不要再添加 `td` 或其他曝光时间补偿。

加速度保留 m/s²，角速度保留 °/s，由现有 C++ 加载器统一转换为 rad/s；测量值保留在原始 IMU 坐标系。录制最后一帧之后未继续采 IMU，补偿后的末样本比末帧早 4.36–8.80 ms；当前积分器用末样本恒值外推到帧末。转换器要求末尾差值和相邻 IMU 间隔都不超过 10 ms；这些末帧均在录制的最后静止段。

## 相机和 IMU 标定

生成的 `params.yaml` 从共享 `config/params.yaml` 复制算法设置，再替换该相机相关参数：

- 使用 `camera_s003_refit.yaml` 推荐的重标内参，原始 1920×1080：`fx=fy=736.23`、`cx=987.72`、`cy=555.14`。
- 保留项目工作分辨率 1280×720，内参等比例缩放为 `fx=fy=490.82`、`cx=658.48`、`cy=370.0933333`。加载器随后中心裁为 1280×640，模型输入为 896×448；推理内参为 `fx=fy=343.574`、`cx=460.936`、`cy=231.0653333`。原始图像文件不做预处理。
- 从 `SN49953548.conf` 读取 120.094 mm 基线，转换为 0.120094 m。图像已由 ZED SDK 校正，畸变设为零。
- `camera_imu.yaml` 提供 **IMU → 左目**，项目 `T_imu_cam` 需要 **左目 → IMU**，因此取逆。IMU 测量不再重复旋转。机械臂手眼标定 `flange_T_left_camera` 用于生成真值，不作为相机–IMU 外参。
- 采样率设为 200 Hz；SDK 标称陀螺仪噪声密度转换为弧度单位。当前核心 IMU 积分器未读取 YAML 中的这两个噪声字段，因此不能把填入这些值视作已经完成滤波器噪声标定。

标定输入文件与共享参数模板的 SHA256 记录在 `metadata.json`。共享 `config/params.yaml` 不因本机实验数据而改写。

## 真值和评估

生成的 `groundtruth_tum.txt` 逐行保留原始已修正真值，仅筛选保留图像对应的行。真值是左目光学相机在 FR5 基座坐标系中的位姿，位置单位米、四元数顺序 `qx qy qz qw`。原始真值已经包含曝光中心、机械臂时间、行程/转角及端点修正，直接按图像原时间配对。

如果导出的估计是 IMU 位姿 `T_world_imu`，比较前应转换为 `T_world_camera = T_world_imu * T_imu_cam`，再对齐世界坐标原点和朝向。单轴平移/转动不适合靠位置拟合确定完整对齐；可使用首个有效相机位姿对齐并评估相对运动，保留双目的米制尺度。

`forward` 的时钟拟合残差超过 1 ms、`pitch_up` 的前后互相关时间差约 3.7 ms；原报告警告保留在元数据中，解释见原始使用说明。IMU 轴不正交与约 0.2–1.3% 的陀螺仪尺度误差、残余双目视差偏置等尚需单独评估，转换器不对传感器测量和视差添加额外经验修正。

## 当前运行条件

本机已使用项目实际的 `DatasetLoader` 和 `ImuIntegrator::ImuInit` 检查全部六条生成序列：3826 对图像均可读取并得到 1280×640 输出，23884 个 IMU 样本均位于对应帧区间，时间连续且无重复，六次静止初始化全部成功；C++ 配置读取后的推理内参与外参逆变换也已核对。另已逐样本核对原始测量、纳秒补偿和分组，并确认重复生成结果一致，非法输入不改写已有结果。

本机验证程序保存在被 Git 忽略的 `build/fr5-loader-probe/`，运行日志为 `build/fr5-loader-probe.log`。这项检查覆盖数据读取和初始化，不依赖推理模型。

本机编译环境、数据与专用标定已接入。用户提供的视差和 RAFT ONNX 已生成本机 `.plan`，模型视差已用于六条序列的独立融合实验；完整 SLAM 与后端尚未运行。模型入口见 [LOCAL_SETUP.md](LOCAL_SETUP.md)，P/R 与精度限制见 [融合协方差诊断](FUSION_COVARIANCE_DIAGNOSIS.md)。
