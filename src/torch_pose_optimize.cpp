#include "sio_backend_modules.h"

#ifdef SIO_USE_LIBTORCH
JointLossTerms TorchPoseOptimizer::BuildJointLoss(
  const torch::Tensor &pose_delta, const torch::Tensor &d_bg, const torch::Tensor &d_ba,
  const torch::Tensor &photo_grad, const torch::Tensor &imu_grad, const torch::Tensor &J_R_bg,
  const torch::Tensor &J_p_ba, const torch::Tensor &J_v_ba, double photometric_error_value,
  double ssim_error_value, double depth_error_value, const JointLossWeights &weights) {
  // Loss construction pipeline (all scalar terms):
  // 1) photometric linearized term
  // 2) SSIM-coupled term
  // 3) IMU linear consistency term
  // 4) preintegration pose prior term
  // 5) bias-coupled residual term
  // Total:
  //   L = L_photo + L_ssim + L_imu + L_preint + L_bias
  const auto opts = pose_delta.options();

  // pose_delta = [theta(3), rho(3)] where
  // theta: rotation increment, rho: translation increment.
  auto rot_res = pose_delta.slice(0, 0, 3);
  auto pos_res = pose_delta.slice(0, 3, 6);

  const auto photo_err = torch::full({}, photometric_error_value, opts);
  const auto ssim_err = torch::full({}, ssim_error_value, opts);
  const auto depth_err = torch::full({}, depth_error_value, opts);

  // L_photo = w_photo * (E_photo + <delta_pose, g_photo>)
  auto photo_linear = (pose_delta * photo_grad).sum();
  auto loss_photo = std::max(0.0, weights.photometric_weight) * (photo_err + photo_linear);

  // L_ssim = w_ssim * (E_ssim * <delta_pose, g_photo>)
  auto loss_ssim = std::max(0.0, weights.ssim_weight) * (ssim_err * photo_linear);

  // L_depth = w_depth * E_depth (depth gradient is injected from renderer)
  auto loss_depth = std::max(0.0, weights.depth_weight) * depth_err;

  // L_imu = w_imu * <delta_pose, g_imu>
  auto loss_imu_linear = std::max(0.0, weights.imu_linear_weight) * (pose_delta * imu_grad).sum();

  // L_preint = w_preint * (w_rot * ||theta||^2 + w_pos * ||rho||^2)
  auto loss_preint_pose = std::max(0.0, weights.preint_pose_weight)
                          * (std::max(0.0, weights.imu_rot_prior_weight) * rot_res.pow(2).sum()
                             + std::max(0.0, weights.imu_pos_prior_weight) * pos_res.pow(2).sum());

  // Bias-coupled residuals using preintegration Jacobians:
  // r_bg = theta + J_R_bg * d_bg
  // r_ba = rho + (J_p_ba + 0.3 * J_v_ba) * d_ba
  // L_bias = w_bias * (||r_bg||^2 + 0.3 * ||r_ba||^2)
  auto bias_rot_res = rot_res + torch::matmul(J_R_bg, d_bg);
  auto bias_pos_res = pos_res + torch::matmul(J_p_ba + 0.3 * J_v_ba, d_ba);
  auto loss_bias = std::max(0.0, weights.bias_weight)
                   * (bias_rot_res.pow(2).sum() + 0.3 * bias_pos_res.pow(2).sum());

  JointLossTerms terms;
  terms.photometric_loss = loss_photo;
  terms.ssim_loss = loss_ssim;
  terms.depth_loss = loss_depth;
  terms.imu_linear_loss = loss_imu_linear;
  terms.preint_pose_loss = loss_preint_pose;
  terms.bias_loss = loss_bias;
  terms.total_loss =
    loss_photo + loss_ssim + loss_depth + loss_imu_linear + loss_preint_pose + loss_bias;
  return terms;
}
#endif


void TorchPoseOptimizer::Init(double pose_lr, double gyr_bias_lr, double acc_bias_lr) {
#ifdef SIO_USE_LIBTORCH
  device_ = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  const auto f64 = torch::TensorOptions().dtype(torch::kFloat64).device(device_);


  // 初始化优化变量和Adam优化器
  pose_ = torch::zeros({6}, f64).requires_grad_(true);
  d_bg_ = torch::zeros({3}, f64).requires_grad_(true);
  d_ba_ = torch::zeros({3}, f64).requires_grad_(true);

  const double plr = std::clamp(pose_lr, 1e-8, 1.0);
  const double glr = std::clamp(gyr_bias_lr, 1e-8, 1.0);
  const double alr = std::clamp(acc_bias_lr, 1e-8, 1.0);

  pose_optim_ = std::make_unique<torch::optim::Adam>(
    std::vector<torch::Tensor>{pose_},
    torch::optim::AdamOptions(plr).betas({0.9, 0.999}).eps(1e-8));
  bg_optim_ = std::make_unique<torch::optim::Adam>(
    std::vector<torch::Tensor>{d_bg_},
    torch::optim::AdamOptions(glr).betas({0.9, 0.999}).eps(1e-8));
  ba_optim_ = std::make_unique<torch::optim::Adam>(
    std::vector<torch::Tensor>{d_ba_},
    torch::optim::AdamOptions(alr).betas({0.9, 0.999}).eps(1e-8));

  initialized_ = true;
  has_staged_loss_ = false;
#endif
}

void TorchPoseOptimizer::Reset() {
#ifdef SIO_USE_LIBTORCH
  if (!initialized_)
    return;
  {
    torch::NoGradGuard no_grad;
    pose_.zero_();
    d_bg_.zero_();
    d_ba_.zero_();
  }
  pose_optim_->zero_grad();
  bg_optim_->zero_grad();
  ba_optim_->zero_grad();
  has_staged_loss_ = false;
#endif
}

bool TorchPoseOptimizer::ComputeJointLoss(ImuStatesGroup &state,
                                          const Eigen::Matrix<double, 6, 1> &photo_grad_pose,
                                          const Eigen::Matrix<double, 6, 1> &imu_grad_pose,
                                          double photometric_error_value, double ssim_error_value,
                                          double depth_error_value, const ImuStatesGroup &imu_ref,
                                          const ImuPreintegration &preint, double imu_rot_weight,
                                          double imu_pos_weight, double depth_weight) {
#ifdef SIO_USE_LIBTORCH
  if (!initialized_)
    return false;

  const auto cpu_f64 = torch::TensorOptions().dtype(torch::kFloat64);
  // 将输入数据转换为LibTorch张量
  auto photo_grad = torch::from_blob(const_cast<double *>(photo_grad_pose.data()), {6}, cpu_f64)
                      .clone()
                      .to(device_);
  auto imu_grad =
    torch::from_blob(const_cast<double *>(imu_grad_pose.data()), {6}, cpu_f64).clone().to(device_);
  auto J_R_bg =
    torch::from_blob(const_cast<double *>(preint.J_R_bg), {3, 3}, cpu_f64).clone().to(device_);
  auto J_p_ba =
    torch::from_blob(const_cast<double *>(preint.J_p_ba), {3, 3}, cpu_f64).clone().to(device_);
  auto J_v_ba =
    torch::from_blob(const_cast<double *>(preint.J_v_ba), {3, 3}, cpu_f64).clone().to(device_);

  // 赋予损失权重
  JointLossWeights weights;
  weights.imu_rot_prior_weight = imu_rot_weight;
  weights.imu_pos_prior_weight = imu_pos_weight;
  weights.depth_weight = depth_weight;
  // 优化器梯度计算前先清零之前的梯度积累
  pose_optim_->zero_grad();
  bg_optim_->zero_grad();
  ba_optim_->zero_grad();
  // 构建由光度损失、IMU一致性损失、SSIM损失和深度损失组成的联合损失，并计算其相对于优化变量的梯度。

  staged_loss_terms_ =
    BuildJointLoss(pose_, d_bg_, d_ba_, photo_grad, imu_grad, J_R_bg, J_p_ba, J_v_ba,
                   photometric_error_value, ssim_error_value, depth_error_value, weights);
  has_staged_loss_ = true;

  (void)state;
  (void)imu_ref;
  return true;
#else
  (void)state;
  (void)photo_grad_pose;
  (void)imu_grad_pose;
  (void)photometric_error_value;
  (void)ssim_error_value;
  (void)depth_error_value;
  (void)imu_ref;
  (void)preint;
  (void)imu_rot_weight;
  (void)imu_pos_weight;
  (void)depth_weight;
  return false;
#endif
}

bool TorchPoseOptimizer::BackwardAndStep(ImuStatesGroup &state, const ImuStatesGroup &imu_ref) {
  return BackwardAndStep(state, imu_ref, -1.0, -1.0, -1.0);
}

bool TorchPoseOptimizer::BackwardAndStep(ImuStatesGroup &state, const ImuStatesGroup &imu_ref,
                                         double pose_lr, double bias_gyr_lr, double bias_acc_lr) {
#ifdef SIO_USE_LIBTORCH
  if (!initialized_ || !has_staged_loss_)
    return false;

  if (pose_lr > 0 && pose_optim_) {
    for (auto &group : pose_optim_->param_groups()) {
      group.options().set_lr(pose_lr);
    }
  }
  if (bias_gyr_lr > 0 && bg_optim_) {
    for (auto &group : bg_optim_->param_groups()) {
      group.options().set_lr(bias_gyr_lr);
    }
  }
  if (bias_acc_lr > 0 && ba_optim_) {
    for (auto &group : ba_optim_->param_groups()) {
      group.options().set_lr(bias_acc_lr);
    }
  }

  staged_loss_terms_.total_loss.backward();

  pose_optim_->step();
  bg_optim_->step();
  ba_optim_->step();

  torch::clamp_(d_bg_, -0.5, 0.5);
  torch::clamp_(d_ba_, -2.0, 2.0);

  const auto pose_cpu = pose_.detach().to(torch::kCPU).contiguous();
  const auto bg_cpu = d_bg_.detach().to(torch::kCPU).contiguous();
  const auto ba_cpu = d_ba_.detach().to(torch::kCPU).contiguous();

  Eigen::Matrix<double, 6, 1> delta = Eigen::Matrix<double, 6, 1>::Zero();
  const double *pose_ptr = pose_cpu.data_ptr<double>();
  for (int i = 0; i < 6; ++i)
    delta[i] = pose_ptr[i];

  ApplyPoseDeltaLocal(state, delta);

  const double *bg_ptr = bg_cpu.data_ptr<double>();
  const double *ba_ptr = ba_cpu.data_ptr<double>();
  state.bias_g += Eigen::Vector3d(bg_ptr[0], bg_ptr[1], bg_ptr[2]);
  state.bias_a += Eigen::Vector3d(ba_ptr[0], ba_ptr[1], ba_ptr[2]);

  state.bias_g = state.bias_g.cwiseMax(-1.0).cwiseMin(1.0);
  state.bias_a = state.bias_a.cwiseMax(-5.0).cwiseMin(5.0);

  (void)imu_ref;
  has_staged_loss_ = false;
  return true;
#else
  (void)state;
  (void)imu_ref;
  (void)pose_lr;
  (void)bias_gyr_lr;
  (void)bias_acc_lr;
  return false;
#endif
}

void TorchPoseOptimizer::ApplyPoseDeltaLocal(ImuStatesGroup &state,
                                             const Eigen::Matrix<double, 6, 1> &delta) {
  const Eigen::Vector3d d_rot = delta.head<3>();
  const Eigen::Vector3d d_pos = delta.tail<3>();
  const double theta = d_rot.norm();
  if (theta > 1e-12) {
    state.rot_end = state.rot_end * Eigen::AngleAxisd(theta, d_rot / theta).toRotationMatrix();
  }
  state.pos_end += d_pos;
}
