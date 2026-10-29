#pragma once

#include "data.h"
#include "gaussian_map_manager.h"
#include "sio_backend_modules.h"
#include "sio_stereo_geometry.h"

#include <Eigen/Dense>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

// ────────────────────────────────────────────────────────────────────────────
// 后端 → 前端状态回传（WindowOptimize 后验）
// ────────────────────────────────────────────────────────────────────────────
struct BackendStateUpdate {
  int backend_keyframe_id = -1;
  int frontend_keyframe_id = -1; // 与 KeyFrameData::frontend_keyframe_id 对齐
  double timestamp = 0.0;        // 主匹配键（秒）
  ImuStatesGroup state;          // 优化后的 IMU 状态（位姿 / 速度 / 偏置 / cov）
};

// ────────────────────────────────────────────────────────────────────────────
// 后端位姿图节点（对应一个后端关键帧）
// ────────────────────────────────────────────────────────────────────────────
struct BackendNode {
  int id;                   // 后端关键帧 ID
  int frontend_keyframe_id = -1; // 对应前端稳定 ID（回传用）
  double timestamp;         // 时间戳（秒）
  ImuStatesGroup state;     // 优化变量：位姿 / 速度 / 偏置
  ImuPreintegration preint; // 与上一节点之间的 IMU 预积分结果
  cv::Mat obs_image;        // 该关键帧观测图像（用于渲染光度误差）
  cv::Mat obs_depth;        // 可选深度监督（米）
  cv::Mat depth_conf;       // 可选深度置信度（0~1）
  bool fixed = false;       // 仅位姿固定（不限制高斯地图点优化）
};

// ────────────────────────────────────────────────────────────────────────────
// 后端优化参数
// ────────────────────────────────────────────────────────────────────────────
struct BackendParams {
  // 滑动窗口
  int window_size = 10; // 滑动窗口关键帧数量上限

  // 3DGS 地图管理
  float gs_dim = 0.05f;                // 高斯球尺寸（m）
  float gs_opacity_prune_thr = 0.05f;  // 不透明度低于此值时剪枝
  int gs_miss_prune_thr = 20;          // 连续未命中超过此帧数时剪枝
  int gs_densify_interval = 5;         // 每隔多少帧致密化一次
  float gs_densify_grad_thr = 0.0002f; // 致密化梯度阈值
  int gs_densify_min_obs = 3;          // 触发稠密化所需的最少观测次数

  // IMU 预积分噪声（σ 值，单位与传感器一致）
  double imu_acc_noise = 0.1;
  double imu_gyro_noise = 0.01;
  double imu_acc_bias_noise = 0.001;
  double imu_gyro_bias_noise = 0.0001;

  // 优化收敛判断
  int max_iters = 10;         // 后端 GN/LM 最大迭代次数
  double converge_thr = 1e-6; // Δx 范数收敛阈值

  // 联合优化中 IMU 一致性约束权重
  double imu_rot_consistency_weight = 0.1;
  double imu_pos_consistency_weight = 0.1;

  // 联合优化中 IMU 零偏更新步长
  double imu_bias_gyr_lr = 0.15;
  double imu_bias_acc_lr = 0.10;

  // 联合优化中高斯球位姿/参数更新步长
  double gs_pose_lr = 0.08;
  double gs_opacity_lr = 0.20;
  float gs_cov_min = 1e-5f;
  float gs_cov_max = 0.25f;

  // 可选深度损失权重（<=0 关闭）
  double depth_loss_weight = 0.0;

  // 学习率调度配置：根据迭代轮次动态调整学习率
  // 支持的策略：
  //   "constant"（恒定）
  //   "linear_decay:max_iterations=X"（线性衰减）
  //   "exponential_decay:decay_rate=X:decay_steps=Y"（指数衰减）
  //   "step_decay:milestones=100,200:decay_rates=0.1,0.01"（分段衰减）
  //   "cosine_annealing:max_iterations=X:min_lr_ratio=Y"（余弦退火）
  //   "polynomial_decay:max_iterations=X:power=Y"（多项式衰减）
  //   "multistage:stages=100,200:ratios=1.0,0.5"（多阶段）
  std::string pose_lr_scheduler = "constant";     // 位姿学习率调度策略
  std::string bias_lr_scheduler = "constant";     // IMU偏置学习率调度策略
  std::string gs_pose_lr_scheduler = "constant";  // 高斯位姿学习率调度策略
  std::string gs_param_lr_scheduler = "constant"; // 高斯参数学习率调度策略
};

// ────────────────────────────────────────────────────────────────────────────
// SIOBackend：3DGS + IMU 后端优化线程
// ────────────────────────────────────────────────────────────────────────────
/**
 * @brief 后端：滑动窗口 3DGS 光度 + IMU 预积分联合优化（独立线程）。
 */
class SIOBackend {
public:
  SIOBackend();
  ~SIOBackend();

  /** @brief 启动后端工作线程。 */
  void Start();
  /** @brief 停止工作线程并 join。 */
  void Stop();

  /**
   * @brief 唤醒后端处理一个新的后端关键帧。
   * @param backend_kf 含点云/图像/IMU 边的关键帧
   */
  void Wake(const std::shared_ptr<KeyFrameData> &backend_kf);

  /**
   * @brief 非阻塞弹出一条 WindowOptimize 后验回传。
   * @param out 输出 BackendStateUpdate
   * @return 队列非空且弹出成功为 true
   */
  bool TryPopStateUpdate(BackendStateUpdate &out);

  /** @brief 可写参数（须在 Start() 前配置）。 */
  BackendParams &MutableParams() {
    return params_;
  }
  /** @brief 只读参数。 */
  const BackendParams &GetParams() const {
    return params_;
  }

  /**
   * @brief 设置 3DGS 渲染内参（裁减未缩放分辨率，如 1280×640）。
   * @param cam 渲染用 CameraIntrinsics
   */
  void SetCameraIntrinsics(const CameraIntrinsics &cam) {
    render_cam_ = cam;
  }

  /**
   * @brief 设置推理分辨率内参（视差→深度后再上采样）。
   * @param cam 如 896×448
   */
  void SetInferenceCameraIntrinsics(const CameraIntrinsics &cam) {
    inference_cam_ = cam;
  }

  /** @brief 设置立体图像裁减/缩放几何。 */
  void SetStereoImageGeometry(const sio_utils::StereoImageGeometry &geom) {
    stereo_geom_ = geom;
  }

  /**
   * @brief 设置外参：点_body = R·点_cam + t。
   * @param R_bc,t_bc 相机→机体
   */
  void SetBodyFromCameraExtrinsic(const Eigen::Matrix3d &R_bc, const Eigen::Vector3d &t_bc) {
    R_bc_ = R_bc;
    t_bc_ = t_bc;
  }

private:
  /** @brief 工作线程主循环：取 KF → 预积分 → 窗口优化。 */
  void Run();
  /**
   * @brief 用首个后端关键帧初始化窗口与高斯地图。
   * @param kf 首帧关键帧
   */
  void InitializeFromKeyframe(const std::shared_ptr<KeyFrameData> &kf);

  /**
   * @brief 对 node 做相对上一节点的 IMU 预积分。
   * @param imu_buf IMU 序列
   * @param t_start,t_end 积分区间 (s)
   */
  void IntegrateImu(BackendNode &node, const std::deque<sensor_msgs::msg::Imu::SharedPtr> &imu_buf,
                    double t_start, double t_end);

  /** @brief 滑动窗口联合优化（3DGS 光度 + IMU）。 */
  void WindowOptimize();
  /** @brief 窗口超限时边缘化最旧节点。 */
  void Marginalize();
  /** @brief 将最新节点后验推入 state_feedback_queue_。 */
  void PushLatestStateUpdate();

  // ── 线程同步 ───────────────────────────────────────────────────────────
  std::atomic<bool> running_{false};
  std::mutex mtx_backend_;
  std::condition_variable cv_backend_;
  std::thread worker_;

  // ── 关键帧输入队列 ─────────────────────────────────────────────────────
  std::deque<std::shared_ptr<KeyFrameData>> backend_keyframe_queue_;

  // ── 后端 → 前端状态回传队列 ───────────────────────────────────────────
  std::deque<BackendStateUpdate> state_feedback_queue_;
  std::mutex mtx_feedback_;

  // ── 3DGS 地图管理 ──────────────────────────────────────────────────────
  GaussianMapManager gaussian_map_manager_;
  Pose3DGSOptimizer pose_3dgs_optimizer_;

  CameraIntrinsics render_cam_{525.0f, 525.0f, 319.5f, 239.5f, 0.0f};
  CameraIntrinsics inference_cam_{525.0f, 525.0f, 319.5f, 239.5f, 0.0f};
  sio_utils::StereoImageGeometry stereo_geom_{};

  // ── 滑动窗口位姿图 ─────────────────────────────────────────────────────
  std::deque<BackendNode> window_; // 滑动窗口节点
  std::mutex mtx_window_;          // 窗口读写锁
  // ── LibTorch 联合优化器（跨迭代持久：优化变量 + Adam 动量）──────────
  TorchPoseOptimizer pose_optimizer_;

  // ── 学习率调度器（根据迭代轮次动态调整学习率）────────────────────
  std::unique_ptr<LearningRateScheduler> pose_lr_scheduler_;
  std::unique_ptr<LearningRateScheduler> bias_lr_scheduler_;
  std::unique_ptr<LearningRateScheduler> gs_pose_lr_scheduler_;
  std::unique_ptr<LearningRateScheduler> gs_param_lr_scheduler_;

  // ── 状态与配置 ─────────────────────────────────────────────────────────
  bool initialized_ = false;
  BackendParams params_;
  int global_iteration_ = 0; // 全局迭代计数器（用于学习率调度）

  Eigen::Matrix3d R_bc_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_bc_ = Eigen::Vector3d::Zero();
};
