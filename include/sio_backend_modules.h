#pragma once

#include "data.h"
#include "gaussian_map_manager.h"

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#ifdef SIO_USE_LIBTORCH
#  include <torch/optim.h>
#  include <torch/torch.h>
#endif

// ────────────────────────────────────────────────────────────────────────────
// 学习率调度器：根据迭代轮次动态计算学习率
// 支持多种策略：分段衰减、指数衰减、余弦退火、线性衰减等
// ────────────────────────────────────────────────────────────────────────────

class LearningRateScheduler {
public:
  virtual ~LearningRateScheduler() = default;

  // 根据当前迭代轮次计算学习率
  // iteration: 当前迭代轮次（从 0 开始）
  // base_lr: 基础学习率
  virtual double GetLearningRate(int iteration, double base_lr) const = 0;

  // 获取调度器类型名称
  virtual std::string GetName() const = 0;
};

// ────────────────────────────────────────────────────────────────────────────
// 恒定学习率（无调度）
// ────────────────────────────────────────────────────────────────────────────
class ConstantLRScheduler : public LearningRateScheduler {
public:
  double GetLearningRate(int iteration, double base_lr) const override;
  std::string GetName() const override;
};

// ────────────────────────────────────────────────────────────────────────────
// 线性衰减：lr = base_lr * (1 - iteration / max_iterations)
// ────────────────────────────────────────────────────────────────────────────
class LinearDecayScheduler : public LearningRateScheduler {
public:
  explicit LinearDecayScheduler(int max_iterations);
  double GetLearningRate(int iteration, double base_lr) const override;
  std::string GetName() const override;

private:
  int max_iterations_;
};

// ────────────────────────────────────────────────────────────────────────────
// 指数衰减：lr = base_lr * decay_rate^iteration
// ────────────────────────────────────────────────────────────────────────────
class ExponentialDecayScheduler : public LearningRateScheduler {
public:
  explicit ExponentialDecayScheduler(double decay_rate, int decay_steps = 100);
  double GetLearningRate(int iteration, double base_lr) const override;
  std::string GetName() const override;

private:
  double decay_rate_; // 每 decay_steps 迭代时的衰减倍率
  int decay_steps_;   // 衰减步长
};

// ────────────────────────────────────────────────────────────────────────────
// 分段学习率衰减：在指定的迭代轮次处降低学习率
// 例：milestones=[100, 200], decay_rates=[0.1, 0.01]
//     0-99: base_lr
//     100-199: base_lr * 0.1
//     200+: base_lr * 0.1 * 0.01
// ────────────────────────────────────────────────────────────────────────────
class StepDecayScheduler : public LearningRateScheduler {
public:
  explicit StepDecayScheduler(const std::vector<int> &milestones,
                              const std::vector<double> &decay_rates);
  double GetLearningRate(int iteration, double base_lr) const override;
  std::string GetName() const override;

private:
  std::vector<int> milestones_;     // 触发衰减的迭代轮次
  std::vector<double> decay_rates_; // 对应的衰减倍率
};

// ────────────────────────────────────────────────────────────────────────────
// 余弦退火：lr = lr_min + (lr_max - lr_min) * (1 + cos(π * t / T)) / 2
// ────────────────────────────────────────────────────────────────────────────
class CosineAnnealingScheduler : public LearningRateScheduler {
public:
  explicit CosineAnnealingScheduler(int max_iterations, double min_lr_ratio = 0.0);
  double GetLearningRate(int iteration, double base_lr) const override;
  std::string GetName() const override;

private:
  int max_iterations_;
  double min_lr_ratio_; // 最小学习率与基础学习率的比例
};

// ────────────────────────────────────────────────────────────────────────────
// 多阶段学习率：不同阶段使用不同的基础学习率
// 例：stages=[100, 200, 300], ratios=[1.0, 0.5, 0.1]
//     0-99: base_lr * 1.0
//     100-199: base_lr * 0.5
//     200-299: base_lr * 0.1
// ────────────────────────────────────────────────────────────────────────────
class MultiStageScheduler : public LearningRateScheduler {
public:
  explicit MultiStageScheduler(const std::vector<int> &stages,
                               const std::vector<double> &lr_ratios);
  double GetLearningRate(int iteration, double base_lr) const override;
  std::string GetName() const override;

private:
  std::vector<int> stages_;       // 每个阶段的起始迭代轮次
  std::vector<double> lr_ratios_; // 对应的学习率倍率
};

// ────────────────────────────────────────────────────────────────────────────
// 多项式衰减：lr = base_lr * (1 - iteration / max_iterations)^power
// ────────────────────────────────────────────────────────────────────────────
class PolynomialDecayScheduler : public LearningRateScheduler {
public:
  explicit PolynomialDecayScheduler(int max_iterations, double power = 1.0);
  double GetLearningRate(int iteration, double base_lr) const override;
  std::string GetName() const override;

private:
  int max_iterations_;
  double power_;
};

// ────────────────────────────────────────────────────────────────────────────
// 学习率调度器工厂：根据配置字符串创建调度器
// ────────────────────────────────────────────────────────────────────────────
class LearningRateSchedulerFactory {
public:
  // 从配置字符串创建调度器
  static std::unique_ptr<LearningRateScheduler> Create(const std::string &config,
                                                       int max_iterations = 1000);
};

struct JointLossWeights {
  double photometric_weight = 1.0;
  double ssim_weight = 0.15;
  double depth_weight = 0.0;
  double imu_linear_weight = 1.0;
  double imu_rot_prior_weight = 0.1;
  double imu_pos_prior_weight = 0.1;
  double preint_pose_weight = 1.0;
  double bias_weight = 1.0;
};

#ifdef SIO_USE_LIBTORCH
struct JointLossTerms {
  torch::Tensor photometric_loss;
  torch::Tensor ssim_loss;
  torch::Tensor depth_loss;
  torch::Tensor imu_linear_loss;
  torch::Tensor preint_pose_loss;
  torch::Tensor bias_loss;
  torch::Tensor total_loss;
};
#endif

class TorchPoseOptimizer {
public:
  static bool Available() {
#ifdef SIO_USE_LIBTORCH
    return true;
#else
    return false;
#endif
  }

  void Init(double pose_lr, double gyr_bias_lr, double acc_bias_lr);

  void Reset();

  bool ComputeJointLoss(ImuStatesGroup &state, const Eigen::Matrix<double, 6, 1> &photo_grad_pose,
                        const Eigen::Matrix<double, 6, 1> &imu_grad_pose,
                        double photometric_error_value, double ssim_error_value,
                        double depth_error_value, const ImuStatesGroup &imu_ref,
                        const ImuPreintegration &preint, double imu_rot_weight,
                        double imu_pos_weight, double depth_weight = 0.0);

  // 反向传播并优化步长（支持动态学习率）
  bool BackwardAndStep(ImuStatesGroup &state, const ImuStatesGroup &imu_ref);

  // 反向传播并优化步长（支持动态学习率参数）
  // pose_lr, bias_gyr_lr, bias_acc_lr: 动态学习率（<=0 表示使用默认值）
  bool BackwardAndStep(ImuStatesGroup &state, const ImuStatesGroup &imu_ref, double pose_lr,
                       double bias_gyr_lr, double bias_acc_lr);

private:
  static void ApplyPoseDeltaLocal(ImuStatesGroup &state, const Eigen::Matrix<double, 6, 1> &delta);

#ifdef SIO_USE_LIBTORCH
  static JointLossTerms BuildJointLoss(const torch::Tensor &pose_delta, const torch::Tensor &d_bg,
                                       const torch::Tensor &d_ba, const torch::Tensor &photo_grad,
                                       const torch::Tensor &imu_grad, const torch::Tensor &J_R_bg,
                                       const torch::Tensor &J_p_ba, const torch::Tensor &J_v_ba,
                                       double photometric_error_value, double ssim_error_value,
                                       double depth_error_value, const JointLossWeights &weights);

  bool initialized_ = false;
  torch::Device device_{torch::kCPU};

  torch::Tensor pose_;
  torch::Tensor d_bg_;
  torch::Tensor d_ba_;

  std::unique_ptr<torch::optim::Adam> pose_optim_;
  std::unique_ptr<torch::optim::Adam> bg_optim_;
  std::unique_ptr<torch::optim::Adam> ba_optim_;

  JointLossTerms staged_loss_terms_;
  bool has_staged_loss_ = false;
#endif
};

struct Pose3DGSOptimizeOptions {
  int max_iterations = 8;
  double learning_rate = 0.15;
  double finite_diff_eps = 1e-3;
  double grad_stop_thr = 1e-5;
  float min_depth = 0.2f;
  float max_depth = 80.0f;
  int max_splat_radius_px = 3;
};

struct Pose3DGSOptimizeStats {
  double initial_cost = 0.0;
  double final_cost = 0.0;
  int iterations = 0;
  bool converged = false;
};

struct Pose3DGSRenderGradResult {
  cv::Mat rendered_bgr;
  cv::Mat rendered_depth;
  Eigen::Matrix<double, 6, 1> grad_pose = Eigen::Matrix<double, 6, 1>::Zero();
  std::vector<float> gaussian_mean3d_grad_norm;
  double photometric_loss = 0.0;
  double depth_loss = 0.0;
  int rendered_count = 0;
  bool valid = false;
};

class Pose3DGSOptimizer {
public:
  Pose3DGSOptimizer() = default;

  void SetMapSnapshot(const std::vector<Gaussian3D> &gaussians);

  cv::Mat RenderBGR(const ImuStatesGroup &pose, const CameraIntrinsics &cam, int width, int height,
                    const Pose3DGSOptimizeOptions &options) const;

  bool RenderAndPoseGradients(const cv::Mat &observed_bgr, const ImuStatesGroup &pose,
                              const CameraIntrinsics &cam, const Pose3DGSOptimizeOptions &options,
                              const cv::Mat *observed_depth, const cv::Mat *depth_confidence,
                              double depth_loss_weight, Pose3DGSRenderGradResult *result) const;

  double ComputePhotometricCost(const cv::Mat &observed_bgr, const ImuStatesGroup &pose,
                                const CameraIntrinsics &cam,
                                const Pose3DGSOptimizeOptions &options) const;

  bool OptimizePose(const cv::Mat &observed_bgr, ImuStatesGroup &pose, const CameraIntrinsics &cam,
                    const Pose3DGSOptimizeOptions &options,
                    Pose3DGSOptimizeStats *stats = nullptr) const;

private:
  static Eigen::Matrix3d ExpSO3(const Eigen::Vector3d &w);
  static void UpdatePose(ImuStatesGroup &pose, const Eigen::Matrix<double, 6, 1> &delta);

  std::vector<Gaussian3D> gaussians_;
};
