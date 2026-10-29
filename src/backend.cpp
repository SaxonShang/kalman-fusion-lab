#include "backend.h"

#include "sio_frontend_modules.h"
#include "utils.h"

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <rclcpp/rclcpp.hpp>

// 计算SO(3)群中的下指数映射（已移到 utils.h 中的 svio_utils::ExpSO3）
namespace {
  // 根据IMU预积分增量预测新的IMU状态
  ImuStatesGroup PredictStateFromImuPreintegration(const ImuStatesGroup &prev,
                                                   const ImuPreintegration &preint,
                                                   double timestamp) {
    ImuStatesGroup predicted = prev;
    predicted.timestamp = timestamp;

    Eigen::Matrix3d dR = Eigen::Matrix3d::Identity();
    dR << preint.delta_R[0], preint.delta_R[1], preint.delta_R[2], preint.delta_R[3],
      preint.delta_R[4], preint.delta_R[5], preint.delta_R[6], preint.delta_R[7], preint.delta_R[8];

    const Eigen::Vector3d dp(preint.delta_p[0], preint.delta_p[1], preint.delta_p[2]);
    const Eigen::Vector3d dv(preint.delta_v[0], preint.delta_v[1], preint.delta_v[2]);

    predicted.rot_end = prev.rot_end * dR;
    predicted.pos_end = prev.pos_end + prev.rot_end * dp;
    predicted.vel_end = prev.vel_end + prev.rot_end * dv;
    return predicted;
  }

  // 对优化状态应用IMU一致性正则化约束
  void ApplyImuConsistencyRegularization(ImuStatesGroup &state, const ImuStatesGroup &imu_ref,
                                         double rot_weight, double pos_weight) {
    const double wr = std::clamp(rot_weight, 0.0, 1.0);
    const double wp = std::clamp(pos_weight, 0.0, 1.0);

    const Eigen::Matrix3d R_err = state.rot_end.transpose() * imu_ref.rot_end;
    Eigen::AngleAxisd aa(R_err);
    if (std::isfinite(aa.angle()) && aa.angle() > 1e-12) {
      state.rot_end = state.rot_end * svio_utils::ExpSO3(wr * aa.angle() * aa.axis());
    }
    state.pos_end = (1.0 - wp) * state.pos_end + wp * imu_ref.pos_end;
  }

  // 计算IMU一致性约束相对于状态的梯度
  Eigen::Matrix<double, 6, 1> ComputeImuConsistencyGradient(const ImuStatesGroup &state,
                                                            const ImuStatesGroup &imu_ref,
                                                            double rot_weight, double pos_weight) {
    Eigen::Matrix<double, 6, 1> grad = Eigen::Matrix<double, 6, 1>::Zero();

    const double wr = std::max(0.0, rot_weight);
    const double wp = std::max(0.0, pos_weight);

    const Eigen::Matrix3d R_err = state.rot_end.transpose() * imu_ref.rot_end;
    Eigen::AngleAxisd aa(R_err);
    if (std::isfinite(aa.angle()) && aa.angle() > 1e-12) {
      // L_rot = wr * ||Log(R_state^T * R_imu)||^2
      grad.head<3>() = -2.0 * wr * aa.angle() * aa.axis();
    }

    // L_pos = wp * ||p_state - p_imu||^2
    grad.tail<3>() = 2.0 * wp * (state.pos_end - imu_ref.pos_end);
    return grad;
  }

  // 计算两幅图像的SSIM结构相似度损失
  double ComputeSSIMLoss(const cv::Mat &img_a_bgr, const cv::Mat &img_b_bgr) {
    if (img_a_bgr.empty() || img_b_bgr.empty() || img_a_bgr.size() != img_b_bgr.size()) {
      return 1.0;
    }

    cv::Mat a_gray, b_gray;
    if (img_a_bgr.channels() == 3) {
      cv::cvtColor(img_a_bgr, a_gray, cv::COLOR_BGR2GRAY);
    } else {
      a_gray = img_a_bgr;
    }
    if (img_b_bgr.channels() == 3) {
      cv::cvtColor(img_b_bgr, b_gray, cv::COLOR_BGR2GRAY);
    } else {
      b_gray = img_b_bgr;
    }

    cv::Mat I1, I2;
    a_gray.convertTo(I1, CV_32F, 1.0 / 255.0);
    b_gray.convertTo(I2, CV_32F, 1.0 / 255.0);

    cv::Mat mu1, mu2;
    cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_sq = mu1.mul(mu1);
    cv::Mat mu2_sq = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_sq, sigma2_sq, sigma12;
    cv::GaussianBlur(I1.mul(I1), sigma1_sq, cv::Size(11, 11), 1.5);
    sigma1_sq -= mu1_sq;
    cv::GaussianBlur(I2.mul(I2), sigma2_sq, cv::Size(11, 11), 1.5);
    sigma2_sq -= mu2_sq;
    cv::GaussianBlur(I1.mul(I2), sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    constexpr float C1 = 0.01f * 0.01f;
    constexpr float C2 = 0.03f * 0.03f;

    cv::Mat ssim_map_num = (2.0f * mu1_mu2 + C1).mul(2.0f * sigma12 + C2);
    cv::Mat ssim_map_den = (mu1_sq + mu2_sq + C1).mul(sigma1_sq + sigma2_sq + C2);

    cv::Mat ssim_map;
    cv::divide(ssim_map_num, ssim_map_den + 1e-12f, ssim_map);
    const cv::Scalar mssim = cv::mean(ssim_map);
    const double ssim = std::clamp(static_cast<double>(mssim[0]), -1.0, 1.0);
    return 1.0 - ssim;
  }

  // 构建与裁减未缩放 obs_image 对齐的深度/置信度监督
  void FillBackendObservations(BackendNode &node, const KeyFrameData &kf,
                               const CameraIntrinsics &inference_cam,
                               const sio_utils::StereoImageGeometry &geom) {
    node.obs_image = kf.image_left.clone();
    node.obs_depth = sio_utils::DisparityToDepthAtCropResolution(kf.disp_filtered, inference_cam,
                                                                 geom, 0.2f, 80.0f);
    if (kf.confidence_map.empty()) {
      node.depth_conf.release();
      return;
    }
    if (kf.confidence_map.size() == node.obs_image.size()) {
      node.depth_conf = kf.confidence_map.clone();
      return;
    }
    cv::resize(kf.confidence_map, node.depth_conf, node.obs_image.size(), 0, 0, cv::INTER_LINEAR);
  }

} // namespace
// 初始化后端并从配置文件加载参数
SIOBackend::SIOBackend() {
  const std::filesystem::path cfg_path = sio_utils::LocateParamsFile();
  if (cfg_path.empty()) {
    return;
  }

  double win_size = static_cast<double>(params_.window_size);
  if (sio_utils::ParseYamlScalarDouble(cfg_path, "sliding_window_size", win_size)) {
    params_.window_size = static_cast<int>(std::max(2.0, std::round(win_size)));
    auto logger = rclcpp::get_logger("SIOBackend");
    RCLCPP_INFO(logger,
                "[Backend] "
                "sliding_window_size=%d from %s",
                params_.window_size, cfg_path.c_str());
  }
}

// 清理后端资源
SIOBackend::~SIOBackend() {
  Stop();
}

void SIOBackend::Start() {
  bool expected = false;
  if (!running_.compare_exchange_strong(expected, true)) {
    return;
  }
  // 主流程：声明优化变量并构建 Adam
  // 优化器，仅调用一次
  pose_optimizer_.Init(params_.gs_pose_lr, params_.imu_bias_gyr_lr, params_.imu_bias_acc_lr);

  // 初始化学习率调度器
  pose_lr_scheduler_ =
    LearningRateSchedulerFactory::Create(params_.pose_lr_scheduler, params_.max_iters);
  bias_lr_scheduler_ =
    LearningRateSchedulerFactory::Create(params_.bias_lr_scheduler, params_.max_iters);
  gs_pose_lr_scheduler_ =
    LearningRateSchedulerFactory::Create(params_.gs_pose_lr_scheduler, params_.max_iters);
  gs_param_lr_scheduler_ =
    LearningRateSchedulerFactory::Create(params_.gs_param_lr_scheduler, params_.max_iters);

  worker_ = std::thread(&SIOBackend::Run, this);
}

void SIOBackend::Stop() {
  const bool was_running = running_.exchange(false);
  if (!was_running && !worker_.joinable()) {
    return;
  }

  cv_backend_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
}

void SIOBackend::Wake(const std::shared_ptr<KeyFrameData> &backend_kf) {
  if (!backend_kf) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(mtx_backend_);
    backend_keyframe_queue_.push_back(backend_kf);
  }
  cv_backend_.notify_one();
}
// 尝试弹出后端状态更新
bool SIOBackend::TryPopStateUpdate(BackendStateUpdate &out) {
  std::lock_guard<std::mutex> lock(mtx_feedback_);
  if (state_feedback_queue_.empty()) {
    return false;
  }
  out = std::move(state_feedback_queue_.front());
  state_feedback_queue_.pop_front();
  return true;
}
// 推送最新后端状态更新
void SIOBackend::PushLatestStateUpdate() {
  BackendStateUpdate upd;
  {
    std::lock_guard<std::mutex> lk(mtx_window_);
    if (window_.empty()) {
      return;
    }
    const BackendNode &latest = window_.back();
    upd.backend_keyframe_id = latest.id;
    upd.frontend_keyframe_id = latest.frontend_keyframe_id;
    upd.timestamp = latest.timestamp;
    upd.state = latest.state;
  }
  {
    std::lock_guard<std::mutex> lock(mtx_feedback_);
    state_feedback_queue_.push_back(std::move(upd));
  }
}
// 后端主循环：处理关键帧的初始化或优化
void SIOBackend::Run() {
  auto logger = rclcpp::get_logger("SIOBackend");

  while (running_.load()) {
    std::shared_ptr<KeyFrameData> backend_kf;

    {
      std::unique_lock<std::mutex> lock(mtx_backend_);
      cv_backend_.wait(lock,
                       [this] { return !backend_keyframe_queue_.empty() || !running_.load(); });

      if (!running_.load()) {
        break;
      }

      backend_kf = backend_keyframe_queue_.front();
      backend_keyframe_queue_.pop_front();
    }

    if (!backend_kf) {
      continue;
    }

    if (!initialized_ || backend_kf->keyframe_id == 0) {
      InitializeFromKeyframe(backend_kf);
      initialized_ = true;
    }

    else {
      RCLCPP_INFO(logger,
                  "[Backend] Optimize KF id=%d "
                  "ts=%.6f imu=%zu pts=%zu",
                  backend_kf->keyframe_id, backend_kf->timestamp, backend_kf->imu_buffer.size(),
                  backend_kf->pointcloud ? backend_kf->pointcloud->size() : 0);

      // 1. 构建新的滑动窗口节点
      BackendNode node;
      node.id = backend_kf->keyframe_id;
      node.frontend_keyframe_id = backend_kf->frontend_keyframe_id;
      node.timestamp = backend_kf->timestamp;
      node.state = backend_kf->imu_states;
      FillBackendObservations(node, *backend_kf, inference_cam_, stereo_geom_);
      node.fixed = false;

      // 预积分 IMU 增量，并将节点添加到滑动窗口
      {
        std::lock_guard<std::mutex> lk(mtx_window_);
        if (!window_.empty()) {
          IntegrateImu(node, backend_kf->imu_buffer, window_.back().timestamp,
                       backend_kf->timestamp);
        }
        window_.push_back(std::move(node));
      }

      // 2.按照窗口大小限制，边界外节点进行边缘化
      while (true) {
        std::lock_guard<std::mutex> lk(mtx_window_);
        if (window_.size() <= static_cast<size_t>(params_.window_size)) {
          break;
        }
        Marginalize();
      }

      // 3. 用当前帧点云增量更新高斯地图（与节点 fixed 无关）
      gaussian_map_manager_.AddFromKeyframe(backend_kf, params_.gs_dim);

      // 4. 滑动窗口联合优化（位姿图 + 3DGS 外观）
      //    致密化 / 剪枝已移至优化循环中，基于迭代进度动态执行
      WindowOptimize();
      // 5. 将最新后端关键帧后验回传前端（绝对状态 + 新偏置）
      PushLatestStateUpdate();
    }
  }

  RCLCPP_INFO(logger, "Backend worker stopped.");
}

void SIOBackend::InitializeFromKeyframe(const std::shared_ptr<KeyFrameData> &kf) {
  auto logger = rclcpp::get_logger("SIOBackend");
  RCLCPP_INFO(logger,
              "[Backend] Init from KF id=%d "
              "ts=%.6f pts=%zu",
              kf->keyframe_id, kf->timestamp, kf->pointcloud ? kf->pointcloud->size() : 0);

  // 1.
  // 构建第一个滑动窗口节点（固定，作为全局锚点）
  BackendNode node;
  node.id = kf->keyframe_id;
  node.frontend_keyframe_id = kf->frontend_keyframe_id;
  node.timestamp = kf->timestamp;
  node.state = kf->imu_states;
  FillBackendObservations(node, *kf, inference_cam_, stereo_geom_);
  node.fixed = true; // 第一帧固定，不参与优化
  {
    std::lock_guard<std::mutex> lk(mtx_window_);
    window_.push_back(std::move(node));
  }

  // 2. 用关键帧点云初始化高斯地图
  gaussian_map_manager_.InitializeFromKeyframe(kf, params_.gs_dim);
}

// ── IMU 预积分
// ────────────────────────────────────────────────────────────

void SIOBackend::IntegrateImu(BackendNode &node,
                              const std::deque<sensor_msgs::msg::Imu::SharedPtr> &imu_buf,
                              double t_start, double t_end) {
  // 调用前端 IMU
  // 积分模块，回填后端图优化所需的预积分增量。
  node.preint.timestamp = t_start;
  node.preint.dt = t_end - t_start;

  if (t_end <= t_start || imu_buf.size() < 2 || window_.empty()) {
    return;
  }

  KeyFrameData prev_keyframe;
  prev_keyframe.timestamp = t_start;
  prev_keyframe.imu_states = window_.back().state;

  ImuIntegrator imu_integrator;
  const Pose6D pose_preint = imu_integrator.ImuPreintegrationPose(prev_keyframe, imu_buf, t_end);

  node.preint.dt = pose_preint.dt;
  std::copy(pose_preint.delta_p, pose_preint.delta_p + 3, node.preint.delta_p);
  std::copy(pose_preint.delta_v, pose_preint.delta_v + 3, node.preint.delta_v);
  std::copy(pose_preint.delta_R, pose_preint.delta_R + 9, node.preint.delta_R);

  node.preint.bias_gyr[0] = prev_keyframe.imu_states.bias_g.x();
  node.preint.bias_gyr[1] = prev_keyframe.imu_states.bias_g.y();
  node.preint.bias_gyr[2] = prev_keyframe.imu_states.bias_g.z();
  node.preint.bias_acc[0] = prev_keyframe.imu_states.bias_a.x();
  node.preint.bias_acc[1] = prev_keyframe.imu_states.bias_a.y();
  node.preint.bias_acc[2] = prev_keyframe.imu_states.bias_a.z();
}

// ── 位姿图优化
// ────────────────────────────────────────────────────────────
// 执行滑动窗口内所有关键帧的联合优化

void SIOBackend::WindowOptimize() {
  std::deque<BackendNode> local_window;
  {
    std::lock_guard<std::mutex> lk(mtx_window_);
    if (window_.size() < 2) {
      return;
    }
    local_window = window_;
  }

  const auto gaussians = gaussian_map_manager_.GetMapSnapshot();
  if (gaussians.empty()) {
    return;
  }

  pose_3dgs_optimizer_.SetMapSnapshot(gaussians);
  Pose3DGSOptimizeOptions opt;
  opt.max_iterations = params_.max_iters;

  // 每个优化窗口开始前：将 delta 归零，清除 Adam
  // 动量状态
  pose_optimizer_.Reset();

  if (!TorchPoseOptimizer::Available()) {
    auto logger = rclcpp::get_logger("SIOBackend");
    RCLCPP_ERROR(logger,
                 "[Backend] LibTorch optimizer is "
                 "required but unavailable. "
                 "Rebuild with SIO_ENABLE_LIBTORCH=ON.");
    return;
  }

  Eigen::Vector3d sum_rot_res = Eigen::Vector3d::Zero();
  Eigen::Vector3d sum_pos_res = Eigen::Vector3d::Zero();
  double sum_photo_loss = 0.0;
  int residual_count = 0;


  // “迭代外层、窗口内层”的联合优化结构：每轮先扫描完整个窗口更新位姿/偏置，再按轮次执行高斯更新与densify/prune。
  constexpr int kGaussianUpdateStride = 3;
  for (int iter = 0; iter < opt.max_iterations; ++iter) {
    Eigen::Vector3d iter_sum_rot_res = Eigen::Vector3d::Zero();
    Eigen::Vector3d iter_sum_pos_res = Eigen::Vector3d::Zero();
    double iter_sum_photo_loss = 0.0;
    int iter_residual_count = 0;

    for (size_t i = 1; i < local_window.size(); ++i) {
      BackendNode &prev = local_window[i - 1];
      BackendNode &curr = local_window[i];

      if (curr.fixed || curr.obs_image.empty()) {
        continue;
      }

      // 1. 预测当前节点的IMU初值，并转换到相机坐标系
      const ImuStatesGroup imu_init =
        PredictStateFromImuPreintegration(prev.state, curr.preint, curr.timestamp);
      ImuStatesGroup optimized_state = curr.state;

      // 先将位姿从本体坐标系(IMU)转换到相机坐标系以用于高斯渲染与求导
      ImuStatesGroup camera_state = optimized_state;
      camera_state.rot_end = optimized_state.rot_end * R_bc_;
      camera_state.pos_end = optimized_state.pos_end + optimized_state.rot_end * t_bc_;

      // 调用3DGS渲染与求导接口，在相机坐标系，获取当前位姿的渲染结果和光度梯度。
      Pose3DGSRenderGradResult render_grad;
      const bool ok = pose_3dgs_optimizer_.RenderAndPoseGradients(
        curr.obs_image, camera_state, render_cam_, opt,
        /*observed_depth=*/
        curr.obs_depth.empty() ? nullptr : &curr.obs_depth,
        /*depth_confidence=*/
        curr.depth_conf.empty() ? nullptr : &curr.depth_conf, params_.depth_loss_weight,
        &render_grad);
      if (!ok || !render_grad.valid) {
        continue;
      }

      // 将每高斯梯度统计回写给地图管理器，用于后续densify选择。
      gaussian_map_manager_.AccumulateDensificationStats(render_grad.gaussian_mean3d_grad_norm);

      // 获取基于相机坐标系的光度梯度 [d_theta_c, d_p_c]
      const Eigen::Matrix<double, 6, 1> grad_photo_cam = render_grad.grad_pose;
      const Eigen::Vector3d grad_theta_c = grad_photo_cam.head<3>();
      const Eigen::Vector3d grad_p_c = grad_photo_cam.tail<3>();

      // 利用伴随矩阵的转置将相机的光度梯度转换回本体坐标系(IMU)
      const Eigen::Vector3d grad_p_b = R_bc_ * grad_p_c;
      const Eigen::Vector3d grad_theta_b = R_bc_ * grad_theta_c + t_bc_.cross(grad_p_b);

      Eigen::Matrix<double, 6, 1> grad_photo;
      grad_photo.head<3>() = grad_theta_b;
      grad_photo.tail<3>() = grad_p_b;

      const Eigen::Matrix<double, 6, 1> grad_imu =
        ComputeImuConsistencyGradient(optimized_state, imu_init, params_.imu_rot_consistency_weight,
                                      params_.imu_pos_consistency_weight);
      const Eigen::Matrix<double, 6, 1> grad_joint = grad_photo + grad_imu;

      if (grad_joint.norm() < params_.converge_thr) {
        continue;
      }

      // 根据当前全局迭代次数从学习率调度器获取动态学习率，用于当前步的位姿和IMU偏置更新。
      const double curr_pose_lr =
        pose_lr_scheduler_->GetLearningRate(global_iteration_, params_.gs_pose_lr);
      const double curr_bias_gyr_lr =
        bias_lr_scheduler_->GetLearningRate(global_iteration_, params_.imu_bias_gyr_lr);
      const double curr_bias_acc_lr =
        bias_lr_scheduler_->GetLearningRate(global_iteration_, params_.imu_bias_acc_lr);
      const double ssim_loss = ComputeSSIMLoss(curr.obs_image, render_grad.rendered_bgr);

      // 构建由光度损失、IMU一致性损失、SSIM损失和深度损失组成的联合损失，并计算其相对于优化变量的梯度。
      const bool loss_ok = pose_optimizer_.ComputeJointLoss(
        optimized_state, grad_photo, grad_imu, render_grad.photometric_loss, ssim_loss,
        render_grad.depth_loss, imu_init, curr.preint, params_.imu_rot_consistency_weight,
        params_.imu_pos_consistency_weight, params_.depth_loss_weight);
      if (!loss_ok) {
        auto logger = rclcpp::get_logger("SIOBackend");
        RCLCPP_ERROR(logger,
                     "[Backend] LibTorch "
                     "ComputeJointLoss failed; "
                     "manual optimizer "
                     "fallback is disabled.");
        return;
      }

      // 功能：基于当前联合损失执行一次反向传播并更新状态变量。
      // 其中三路学习率分别作用于：位姿增量、陀螺仪偏置、加速度计偏置。
      const bool stepped = pose_optimizer_.BackwardAndStep(optimized_state, imu_init, curr_pose_lr,
                                                           curr_bias_gyr_lr, curr_bias_acc_lr);
      if (!stepped) {
        auto logger = rclcpp::get_logger("SIOBackend");
        RCLCPP_ERROR(logger,
                     "[Backend] LibTorch BackwardAndStep "
                     "failed; manual optimizer "
                     "fallback is disabled.");
        return;
      }

      // 递增全局迭代计数器
      ++global_iteration_;

      // 兜底：若优化异常导致数值不稳定，则回退到IMU初值。
      if (!optimized_state.pos_end.allFinite() || !optimized_state.rot_end.allFinite()) {
        optimized_state = imu_init;
        ApplyImuConsistencyRegularization(optimized_state, imu_init,
                                          params_.imu_rot_consistency_weight,
                                          params_.imu_pos_consistency_weight);
      }

      curr.state = optimized_state;

      // 统计当前残差以用于后续高斯更新：
      const Eigen::Matrix3d R_err = imu_init.rot_end.transpose() * optimized_state.rot_end;

      // 仅当旋转残差有效且足够大时才统计，以避免数值噪声干扰。
      Eigen::AngleAxisd aa(R_err);
      if (std::isfinite(aa.angle()) && aa.angle() > 1e-12) {
        const Eigen::Vector3d rot_res = aa.angle() * aa.axis();
        iter_sum_rot_res += rot_res;
        sum_rot_res += rot_res;
      }
      // 统计平移残差，条件同上，避免噪声干扰。
      const Eigen::Vector3d pos_res = optimized_state.pos_end - imu_init.pos_end;
      if (pos_res.allFinite() && pos_res.norm() > 1e-12) {
        iter_sum_pos_res += pos_res;
        sum_pos_res += pos_res;
      }

      iter_sum_photo_loss += render_grad.photometric_loss;
      sum_photo_loss += render_grad.photometric_loss;
      ++iter_residual_count;
      ++residual_count;
    }

    bool gaussian_map_changed = false;

    // 在迭代中周期性更新高斯参数，使下一轮渲染使用最新高斯状态。
    // 仅在每kGaussianUpdateStride次迭代触发，避免每步全量更新导致震荡。
    if ((iter + 1) % kGaussianUpdateStride == 0 && iter_residual_count > 0) {
      const double curr_gs_pose_lr =
        gs_pose_lr_scheduler_->GetLearningRate(global_iteration_, params_.gs_pose_lr);
      const double curr_gs_param_lr =
        gs_param_lr_scheduler_->GetLearningRate(global_iteration_, params_.gs_opacity_lr);

      // 用当前迭代的平均残差更新高斯地图：
      const double inv_count = 1.0 / static_cast<double>(iter_residual_count);
      const Eigen::Matrix3d dR_iter = svio_utils::ExpSO3(-iter_sum_rot_res * inv_count);
      const Eigen::Vector3d dt_iter = -(iter_sum_pos_res * inv_count);
      gaussian_map_manager_.OptimizeGaussianPose(dR_iter.cast<float>(), dt_iter.cast<float>(),
                                                 static_cast<float>(curr_gs_pose_lr));
      gaussian_map_manager_.OptimizeGaussianParams(iter_sum_photo_loss * inv_count,
                                                   params_.gs_cov_min, params_.gs_cov_max,
                                                   static_cast<float>(curr_gs_param_lr));
      gaussian_map_changed = true;
    }

    // 功能：在优化中后期动态调整高斯数量。
    // 触发策略：迭代过半后，每5次迭代检查一次，避免前期频繁改图导致收敛不稳。
    if (iter > opt.max_iterations / 2 && iter % 5 == 0) {
      if (gaussian_map_manager_.IsDensifyAndPrune(1)) {
        gaussian_map_manager_.Densify(params_.gs_densify_grad_thr, params_.gs_densify_min_obs);
        gaussian_map_manager_.Prune(params_.gs_opacity_prune_thr, params_.gs_miss_prune_thr);
        gaussian_map_changed = true;
      }
    }

    // 高斯发生变化后，刷新渲染器快照，确保下一次迭代看到最新地图。
    if (gaussian_map_changed) {
      const auto updated_gaussians = gaussian_map_manager_.GetMapSnapshot();
      if (updated_gaussians.empty()) {
        break;
      }
      pose_3dgs_optimizer_.SetMapSnapshot(updated_gaussians);
    }
  }

  // 功能：基于整个滑窗的平均残差执行一次“窗口级”高斯全局校正。
  // 触发条件：当前轮至少有一个有效残差样本（residual_count > 0）。
  // 执行动作：
  // 1) 用平均旋转/平移残差修正高斯整体位姿，降低系统性漂移；
  // 2) 用平均光度损失更新高斯外观参数（协方差/不透明度相关）。
  if (residual_count > 0) {
    const Eigen::Vector3d mean_rot_res = sum_rot_res / residual_count;
    const Eigen::Vector3d mean_pos_res = sum_pos_res / residual_count;
    const double mean_photo_loss = sum_photo_loss / residual_count;

    // 计算高斯球参数优化的动态学习率
    const double curr_gs_pose_lr =
      gs_pose_lr_scheduler_->GetLearningRate(global_iteration_, params_.gs_pose_lr);
    const double curr_gs_param_lr =
      gs_param_lr_scheduler_->GetLearningRate(global_iteration_, params_.gs_opacity_lr);

    // 反向修正高斯球整体位姿，减少系统性残差。
    const Eigen::Matrix3d dR_w = svio_utils::ExpSO3(-mean_rot_res);
    const Eigen::Vector3d dt_w = -mean_pos_res;
    gaussian_map_manager_.OptimizeGaussianPose(dR_w.cast<float>(), dt_w.cast<float>(),
                                               static_cast<float>(curr_gs_pose_lr));
    gaussian_map_manager_.OptimizeGaussianParams(mean_photo_loss, params_.gs_cov_min,
                                                 params_.gs_cov_max,
                                                 static_cast<float>(curr_gs_param_lr));
  }

  // 将局部优化结果安全回写到共享窗口：
  // 先校验窗口长度与关键帧 ID 未变化，再逐项更新状态，避免并发期间错配写入。
  {
    std::lock_guard<std::mutex> lk(mtx_window_);
    if (window_.size() != local_window.size()) {
      return;
    }
    for (size_t i = 0; i < window_.size(); ++i) {
      if (window_[i].id != local_window[i].id) {
        return;
      }
    }
    for (size_t i = 0; i < window_.size(); ++i) {
      window_[i].state = local_window[i].state;
    }
  }
}
// 删除滑动窗口中最旧的节点进行边缘化
void SIOBackend::Marginalize() {
  if (window_.size() <= 1) {
    return;
  }

  // 仅按时间顺序边缘化：删除最旧节点，不在中间查找删除。
  // 若首节点是固定锚点，则删除其后的最旧活动节点。
  if (window_.front().fixed) {
    if (window_.size() > 1) {
      window_.erase(window_.begin() + 1);
    }
  } else {
    window_.pop_front();
  }
}
