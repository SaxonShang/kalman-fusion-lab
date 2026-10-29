#include "gaussian_map_manager.h"

#include <cmath>
#include <cstdint>

#ifdef SIO_USE_LIBTORCH
#  include <torch/torch.h>
#endif

namespace {
  inline Eigen::Matrix3f CovFromScaleRotation(const Eigen::Vector3f &scale,
                                              const Eigen::Quaternionf &q_in) {
    Eigen::Quaternionf q = q_in;
    if (!std::isfinite(q.norm()) || q.norm() < 1e-8f) {
      q = Eigen::Quaternionf::Identity();
    }
    q.normalize();

    const Eigen::Vector3f s = scale.cwiseMax(1e-4f);
    const Eigen::Matrix3f S = s.array().square().matrix().asDiagonal();
    const Eigen::Matrix3f R = q.toRotationMatrix();
    return R * S * R.transpose();
  }

  inline float MaxScaleComponent(const Eigen::Vector3f &s) {
    return std::max(s.x(), std::max(s.y(), s.z()));
  }

  inline float HashToUnitFloat(uint32_t x) {
    // xorshift32 -> [0, 1)
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return static_cast<float>(x & 0x00FFFFFFu) / static_cast<float>(0x01000000u);
  }

  inline Eigen::Vector3f DeterministicUnitDir(int id, int salt) {
    const uint32_t base =
      static_cast<uint32_t>(id * 2654435761u + 1013904223u + static_cast<uint32_t>(salt));
    float x = 2.0f * HashToUnitFloat(base + 0u) - 1.0f;
    float y = 2.0f * HashToUnitFloat(base + 1u) - 1.0f;
    float z = 2.0f * HashToUnitFloat(base + 2u) - 1.0f;
    Eigen::Vector3f v(x, y, z);
    if (!v.allFinite() || v.norm() < 1e-6f) {
      return Eigen::Vector3f::UnitX();
    }
    return v.normalized();
  }
} // namespace

// 将局部坐标系中的点变换到世界坐标系
Eigen::Vector3f GaussianMapManager::ToWorldPoint(const ImuStatesGroup &state,
                                                 const Eigen::Vector3f &p_local) {
  return (state.rot_end.cast<float>() * p_local) + state.pos_end.cast<float>();
}

// 缓存前一关键帧的状态信息用于後续重叠检验
void GaussianMapManager::UpdatePreviousKeyframeCache(const std::shared_ptr<KeyFrameData> &kf) {
  if (!kf) {
    has_prev_kf_cache_ = false;
    return;
  }

  prev_kf_state_ = kf->imu_states;
  has_prev_kf_cache_ = true;
}

// 释放LibTorch优化器的所有稠量和状态
#ifdef SIO_USE_LIBTORCH
void GaussianMapManager::ResetTorchOptimizerState() {
  torch_params_ready_ = false;
  torch_param_count_ = 0;
  t_log_scale_ = torch::Tensor();
  t_color_feature_ = torch::Tensor();
  t_opacity_ = torch::Tensor();
  torch_param_optim_.reset();
}

// 从当前高斯地图构建一个LibTorch Adam优化器
void GaussianMapManager::BuildTorchOptimizerFromMap(double lr) {
  torch_device_ = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

  const int n = static_cast<int>(gaussian_map_.size());
  const auto cpu = torch::TensorOptions().dtype(torch::kFloat32);
  std::vector<float> host_log_scale(3 * n, 0.0f);
  std::vector<float> host_color(3 * n, 0.0f);
  std::vector<float> host_opacity(n, 1.0f);
  for (int i = 0; i < n; ++i) {
    const auto &g = gaussian_map_[i];
    const Eigen::Vector3f s = g.scale.cwiseMax(1e-4f);
    host_log_scale[3 * i + 0] = std::log(s.x());
    host_log_scale[3 * i + 1] = std::log(s.y());
    host_log_scale[3 * i + 2] = std::log(s.z());
    host_color[3 * i + 0] = std::clamp(g.color_feature.x(), 0.0f, 1.0f);
    host_color[3 * i + 1] = std::clamp(g.color_feature.y(), 0.0f, 1.0f);
    host_color[3 * i + 2] = std::clamp(g.color_feature.z(), 0.0f, 1.0f);
    host_opacity[i] = std::clamp(g.opacity, 0.02f, 1.0f);
  }

  t_log_scale_ = torch::from_blob(host_log_scale.data(), {n, 3}, cpu).clone().to(torch_device_);
  t_color_feature_ = torch::from_blob(host_color.data(), {n, 3}, cpu).clone().to(torch_device_);
  t_opacity_ = torch::from_blob(host_opacity.data(), {n, 1}, cpu).clone().to(torch_device_);

  // 设置优化参数，使参数可求导以供优化器更新
  t_log_scale_.set_requires_grad(true);
  t_color_feature_.set_requires_grad(true);
  t_opacity_.set_requires_grad(true);

  torch_param_lr_ = std::clamp(lr, 1e-5, 0.2);
  torch_param_optim_ = std::make_unique<torch::optim::Adam>(
    std::vector<torch::Tensor>{t_log_scale_, t_color_feature_, t_opacity_},
    torch::optim::AdamOptions(torch_param_lr_).betas({0.9, 0.999}).eps(1e-8));
  torch_param_count_ = n;
  torch_params_ready_ = true;
}

// 根据保留掩码删除与LibTorch优化器中的高斯参数
bool GaussianMapManager::PruneTorchOptimizer(const std::vector<uint8_t> &keep_mask) {
  if (!torch_params_ready_ || !torch_param_optim_
      || torch_param_count_ != static_cast<int>(keep_mask.size())) {
    return false;
  }

  const auto cpu_bool = torch::TensorOptions().dtype(torch::kBool);
  auto keep = torch::from_blob(const_cast<uint8_t *>(keep_mask.data()),
                               {static_cast<long>(keep_mask.size())}, cpu_bool)
                .clone()
                .to(torch_device_);

  auto new_log_scale = t_log_scale_.index({keep}).clone().to(torch_device_);
  auto new_color = t_color_feature_.index({keep}).clone().to(torch_device_);
  auto new_opacity = t_opacity_.index({keep}).clone().to(torch_device_);
  new_log_scale.set_requires_grad(true);
  new_color.set_requires_grad(true);
  new_opacity.set_requires_grad(true);

  auto new_optim = std::make_unique<torch::optim::Adam>(
    std::vector<torch::Tensor>{new_log_scale, new_color, new_opacity},
    torch::optim::AdamOptions(torch_param_lr_).betas({0.9, 0.999}).eps(1e-8));

  auto transfer_pruned_state = [&](const torch::Tensor &old_param, const torch::Tensor &new_param) {
    auto &old_state_map = torch_param_optim_->state();
    auto old_it = old_state_map.find(old_param.unsafeGetTensorImpl());
    if (old_it == old_state_map.end() || !old_it->second) {
      return;
    }

    auto &old_state = static_cast<torch::optim::AdamParamState &>(*old_it->second);
    auto new_state = std::make_unique<torch::optim::AdamParamState>();
    new_state->step(old_state.step());
    new_state->exp_avg(old_state.exp_avg().index({keep}).clone());
    new_state->exp_avg_sq(old_state.exp_avg_sq().index({keep}).clone());
    new_optim->state()[new_param.unsafeGetTensorImpl()] = std::move(new_state);
  };

  transfer_pruned_state(t_log_scale_, new_log_scale);
  transfer_pruned_state(t_color_feature_, new_color);
  transfer_pruned_state(t_opacity_, new_opacity);

  t_log_scale_ = new_log_scale;
  t_color_feature_ = new_color;
  t_opacity_ = new_opacity;
  torch_param_count_ = static_cast<int>(new_log_scale.size(0));
  torch_param_optim_ = std::move(new_optim);
  torch_params_ready_ = true;
  return true;
}

// 向LibTorch优化器追加新的高斯疆数
bool GaussianMapManager::AppendTorchOptimizer(const std::vector<Gaussian3D> &additions) {
  if (additions.empty()) {
    return true;
  }
  if (!torch_params_ready_ || !torch_param_optim_
      || torch_param_count_ != static_cast<int>(gaussian_map_.size())) {
    return false;
  }

  const int n = static_cast<int>(additions.size());
  const auto cpu = torch::TensorOptions().dtype(torch::kFloat32);
  std::vector<float> host_log_scale(3 * n, 0.0f);
  std::vector<float> host_color(3 * n, 0.0f);
  std::vector<float> host_opacity(n, 1.0f);
  for (int i = 0; i < n; ++i) {
    const auto &g = additions[static_cast<size_t>(i)];
    const Eigen::Vector3f s = g.scale.cwiseMax(1e-4f);
    host_log_scale[3 * i + 0] = std::log(s.x());
    host_log_scale[3 * i + 1] = std::log(s.y());
    host_log_scale[3 * i + 2] = std::log(s.z());
    host_color[3 * i + 0] = std::clamp(g.color_feature.x(), 0.0f, 1.0f);
    host_color[3 * i + 1] = std::clamp(g.color_feature.y(), 0.0f, 1.0f);
    host_color[3 * i + 2] = std::clamp(g.color_feature.z(), 0.0f, 1.0f);
    host_opacity[i] = std::clamp(g.opacity, 0.02f, 1.0f);
  }

  auto ext_log_scale =
    torch::from_blob(host_log_scale.data(), {n, 3}, cpu).clone().to(torch_device_);
  auto ext_color = torch::from_blob(host_color.data(), {n, 3}, cpu).clone().to(torch_device_);
  auto ext_opacity = torch::from_blob(host_opacity.data(), {n, 1}, cpu).clone().to(torch_device_);

  auto new_log_scale = torch::cat({t_log_scale_, ext_log_scale}, 0).clone().to(torch_device_);
  auto new_color = torch::cat({t_color_feature_, ext_color}, 0).clone().to(torch_device_);
  auto new_opacity = torch::cat({t_opacity_, ext_opacity}, 0).clone().to(torch_device_);
  new_log_scale.set_requires_grad(true);
  new_color.set_requires_grad(true);
  new_opacity.set_requires_grad(true);

  auto new_optim = std::make_unique<torch::optim::Adam>(
    std::vector<torch::Tensor>{new_log_scale, new_color, new_opacity},
    torch::optim::AdamOptions(torch_param_lr_).betas({0.9, 0.999}).eps(1e-8));

  auto transfer_cat_state = [&](const torch::Tensor &old_param, const torch::Tensor &ext_param,
                                const torch::Tensor &new_param) {
    auto &old_state_map = torch_param_optim_->state();
    auto old_it = old_state_map.find(old_param.unsafeGetTensorImpl());
    if (old_it == old_state_map.end() || !old_it->second) {
      return;
    }

    auto &old_state = static_cast<torch::optim::AdamParamState &>(*old_it->second);
    auto new_state = std::make_unique<torch::optim::AdamParamState>();
    new_state->step(old_state.step());
    new_state->exp_avg(torch::cat({old_state.exp_avg(), torch::zeros_like(ext_param)}, 0).clone());
    new_state->exp_avg_sq(
      torch::cat({old_state.exp_avg_sq(), torch::zeros_like(ext_param)}, 0).clone());
    new_optim->state()[new_param.unsafeGetTensorImpl()] = std::move(new_state);
  };

  transfer_cat_state(t_log_scale_, ext_log_scale, new_log_scale);
  transfer_cat_state(t_color_feature_, ext_color, new_color);
  transfer_cat_state(t_opacity_, ext_opacity, new_opacity);

  t_log_scale_ = new_log_scale;
  t_color_feature_ = new_color;
  t_opacity_ = new_opacity;
  torch_param_count_ = static_cast<int>(new_log_scale.size(0));
  torch_param_optim_ = std::move(new_optim);
  torch_params_ready_ = true;
  return true;
}
#endif

// 用关键帧点云初始化高斯地图
void GaussianMapManager::InitializeFromKeyframe(const std::shared_ptr<KeyFrameData> &kf,
                                                float init_size) {
  if (!kf || !kf->pointcloud || kf->pointcloud->empty()) {
    return;
  }

  std::lock_guard<std::mutex> lk(mtx_map_);
  gaussian_map_.clear();
  next_gaussian_id_ = 0;
  densify_counter_ = 0;
#ifdef SIO_USE_LIBTORCH
  ResetTorchOptimizerState();
#endif

  const float s = init_size;
  for (const auto &pt : kf->pointcloud->points) {
    const Eigen::Vector3f p_local(pt.x, pt.y, pt.z);
    Gaussian3D g;
    g.mean = ToWorldPoint(kf->imu_states, p_local);
    g.scale = Eigen::Vector3f::Constant(s);
    g.rotation = Eigen::Quaternionf::Identity();
    g.cov3d = CovFromScaleRotation(g.scale, g.rotation);
    g.color = Eigen::Vector3f(pt.r / 255.f, pt.g / 255.f, pt.b / 255.f);
    g.color_feature = g.color;
    g.opacity = 1.0f;
    g.id = next_gaussian_id_++;
    g.source_kf = kf->keyframe_id;
    gaussian_map_.push_back(std::move(g));
  }

  UpdatePreviousKeyframeCache(kf);
}

// 去重检查后增量添加新的高斯点到地图
void GaussianMapManager::AddFromKeyframe(const std::shared_ptr<KeyFrameData> &kf, float init_size) {
  if (!kf || !kf->pointcloud || kf->pointcloud->empty()) {
    return;
  }

  std::lock_guard<std::mutex> lk(mtx_map_);
  const float s = init_size;
  const float min_prev_depth = 0.10f;
  const float tan_half_hfov = std::tan(62.0f * static_cast<float>(M_PI) / 180.0f / 2.0f);
  const float tan_half_vfov = std::tan(49.0f * static_cast<float>(M_PI) / 180.0f / 2.0f);
  const float min_view_dir_cos = std::cos(20.0f * static_cast<float>(M_PI) / 180.0f);

  const Eigen::Matrix3f R_w_prev = prev_kf_state_.rot_end.cast<float>();
  const Eigen::Matrix3f R_prev_w = R_w_prev.transpose();
  const Eigen::Vector3f t_w_prev = prev_kf_state_.pos_end.cast<float>();
  const Eigen::Vector3f t_w_curr = kf->imu_states.pos_end.cast<float>();

  for (const auto &pt : kf->pointcloud->points) {
    const Eigen::Vector3f p_local(pt.x, pt.y, pt.z);
    const Eigen::Vector3f p_world = ToWorldPoint(kf->imu_states, p_local);

    bool is_overlapped = false;
    if (has_prev_kf_cache_) {
      const Eigen::Vector3f p_prev = R_prev_w * (p_world - t_w_prev);
      const bool in_prev_frustum = (p_prev.z() > min_prev_depth)
                                   && (std::abs(p_prev.x() / p_prev.z()) <= tan_half_hfov)
                                   && (std::abs(p_prev.y() / p_prev.z()) <= tan_half_vfov);

      if (in_prev_frustum) {
        const Eigen::Vector3f v_prev = (p_world - t_w_prev).normalized();
        const Eigen::Vector3f v_curr = (p_world - t_w_curr).normalized();
        const float view_dir_cos = v_prev.dot(v_curr);
        is_overlapped = view_dir_cos >= min_view_dir_cos;
      }
    }

    // 与上一帧重合的视角区域不重复初始化，仅新增视角初始化新高斯点。
    if (is_overlapped) {
      continue;
    }

    Gaussian3D g;
    g.mean = p_world;
    g.scale = Eigen::Vector3f::Constant(s);
    g.rotation = Eigen::Quaternionf::Identity();
    g.cov3d = CovFromScaleRotation(g.scale, g.rotation);
    g.color = Eigen::Vector3f(pt.r / 255.f, pt.g / 255.f, pt.b / 255.f);
    g.color_feature = g.color;
    g.opacity = 1.0f;
    g.id = next_gaussian_id_++;
    g.source_kf = kf->keyframe_id;
    gaussian_map_.push_back(std::move(g));
  }

  UpdatePreviousKeyframeCache(kf);
}

// 根据梯度对大尺度高斯执行分裂摊克隨操作
void GaussianMapManager::Densify(float densify_grad_thr, int min_obs_count) {
  std::lock_guard<std::mutex> lk(mtx_map_);
  if (gaussian_map_.empty()) {
    return;
  }

  const int n = static_cast<int>(gaussian_map_.size());
  const int max_new_points = std::clamp(n / 6, 8, 512);
  const float thr = std::max(1e-8f, densify_grad_thr);
  const int obs_min = std::max(1, min_obs_count);

  // 梯度驱动 + 尺度约束：
  // - 大尺度且梯度高：split（并删除父点）
  // - 小尺度且梯度高：clone（保留父点）
  const float split_scale_gate = std::max(0.02f, 20.0f * thr);
  const float clone_scale_gate = std::max(0.003f, 4.0f * thr);
  const float min_opacity_gate = 0.20f;
  const float min_child_scale = std::max(1e-4f, 0.25f * thr);

  std::vector<Gaussian3D> additions;
  additions.reserve(static_cast<size_t>(max_new_points));
  std::vector<uint8_t> remove_parent(static_cast<size_t>(n), 0u);

  // 第一阶段：split 大尺度点。与 Python 参考一致，保留两个子点并删除父点。
  int new_count = 0;
  for (int i = 0; i < n && new_count + 2 <= max_new_points; ++i) {
    const auto &g = gaussian_map_[i];
    if (!g.is_active || g.opacity < min_opacity_gate) {
      continue;
    }

    if (g.densify_grad_denom < obs_min) {
      continue;
    }
    const float avg_grad = g.densify_grad_accum / static_cast<float>(g.densify_grad_denom);
    if (!std::isfinite(avg_grad) || avg_grad < thr) {
      continue;
    }

    const float smax = MaxScaleComponent(g.scale);
    if (!std::isfinite(smax) || smax < split_scale_gate) {
      continue;
    }

    const Eigen::Vector3f dir = DeterministicUnitDir(g.id, /*salt=*/17);
    const Eigen::Vector3f offset = dir.cwiseProduct(0.5f * g.scale);

    Gaussian3D c1 = g;
    Gaussian3D c2 = g;
    c1.id = next_gaussian_id_++;
    c2.id = next_gaussian_id_++;
    c1.mean = g.mean + offset;
    c2.mean = g.mean - offset;
    c1.scale = (g.scale / 1.6f).cwiseMax(min_child_scale);
    c2.scale = (g.scale / 1.6f).cwiseMax(min_child_scale);
    c1.opacity = std::clamp(g.opacity * 0.95f, 0.02f, 1.0f);
    c2.opacity = std::clamp(g.opacity * 0.95f, 0.02f, 1.0f);
    c1.hit_count = 0;
    c1.miss_count = 0;
    c2.hit_count = 0;
    c2.miss_count = 0;
    c1.cov3d = CovFromScaleRotation(c1.scale, c1.rotation);
    c2.cov3d = CovFromScaleRotation(c2.scale, c2.rotation);

    additions.push_back(std::move(c1));
    additions.push_back(std::move(c2));
    remove_parent[static_cast<size_t>(i)] = 1u;
    new_count += 2;
  }

  // 第二阶段：clone 中等尺度点（不删除父点）。
  for (int i = 0; i < n && new_count < max_new_points; ++i) {
    if (remove_parent[static_cast<size_t>(i)] != 0u) {
      continue;
    }
    const auto &g = gaussian_map_[i];
    if (!g.is_active || g.opacity < min_opacity_gate) {
      continue;
    }

    if (g.densify_grad_denom < obs_min) {
      continue;
    }
    const float avg_grad = g.densify_grad_accum / static_cast<float>(g.densify_grad_denom);
    if (!std::isfinite(avg_grad) || avg_grad < thr) {
      continue;
    }

    const float smax = MaxScaleComponent(g.scale);
    if (!std::isfinite(smax) || smax > split_scale_gate || smax < clone_scale_gate) {
      continue;
    }

    Gaussian3D c = g;
    c.id = next_gaussian_id_++;
    const Eigen::Vector3f dir = DeterministicUnitDir(g.id, /*salt=*/73);
    c.mean = g.mean + dir.cwiseProduct(0.25f * g.scale);
    c.scale = (g.scale * 0.9f).cwiseMax(min_child_scale);
    c.opacity = std::clamp(g.opacity * 0.98f, 0.02f, 1.0f);
    c.hit_count = 0;
    c.miss_count = 0;
    c.cov3d = CovFromScaleRotation(c.scale, c.rotation);
    additions.push_back(std::move(c));
    ++new_count;
  }

  if (additions.empty()) {
    return;
  }

  // 删除 split 的父点。
  std::vector<Gaussian3D> survivors;
  survivors.reserve(gaussian_map_.size());
  std::vector<uint8_t> keep_mask;
  keep_mask.reserve(gaussian_map_.size());
  for (size_t i = 0; i < gaussian_map_.size(); ++i) {
    if (i < remove_parent.size() && remove_parent[i] != 0u) {
      keep_mask.push_back(0u);
      continue;
    }
    keep_mask.push_back(1u);
    survivors.push_back(std::move(gaussian_map_[i]));
  }

#ifdef SIO_USE_LIBTORCH
  bool updated_optimizer = false;
  if (torch_params_ready_ && torch_param_optim_
      && torch_param_count_ == static_cast<int>(gaussian_map_.size())) {
    updated_optimizer = PruneTorchOptimizer(keep_mask);
  }
#endif

  gaussian_map_ = std::move(survivors);

#ifdef SIO_USE_LIBTORCH
  if (!updated_optimizer) {
    ResetTorchOptimizerState();
  }
#endif

#ifdef SIO_USE_LIBTORCH
  if (torch_params_ready_ && torch_param_optim_
      && torch_param_count_ == static_cast<int>(gaussian_map_.size())) {
    if (!AppendTorchOptimizer(additions)) {
      ResetTorchOptimizerState();
    }
  }
#endif

  gaussian_map_.insert(gaussian_map_.end(), additions.begin(), additions.end());

  // 与 Python densification_postfix 一致：每次稠密后重置统计窗口。
  for (auto &g : gaussian_map_) {
    g.densify_grad_accum = 0.0f;
    g.densify_grad_denom = 0;
  }
}

// 累积每个高斯点的梯度统计信息用于后续稠密化
void GaussianMapManager::AccumulateDensificationStats(const std::vector<float> &xyz_grad_norm) {
  if (xyz_grad_norm.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lk(mtx_map_);
  if (gaussian_map_.empty()) {
    return;
  }

  const size_t n = std::min(gaussian_map_.size(), xyz_grad_norm.size());
  for (size_t i = 0; i < n; ++i) {
    const float g = xyz_grad_norm[i];
    if (!std::isfinite(g) || g <= 0.0f) {
      continue;
    }
    gaussian_map_[i].densify_grad_accum += g;
    gaussian_map_[i].densify_grad_denom += 1;
  }
}

// 删除不透明度低或质量殭的高斯点
void GaussianMapManager::Prune(float opacity_prune_thr, int miss_prune_thr) {
  std::lock_guard<std::mutex> lk(mtx_map_);
  if (gaussian_map_.empty()) {
    return;
  }

  const float opacity_thr = std::clamp(opacity_prune_thr, 0.0f, 1.0f);
  const int miss_thr = std::max(0, miss_prune_thr);
  constexpr float kMaxWorldScale = 8.0f;

  std::vector<uint8_t> keep_mask;
  keep_mask.reserve(gaussian_map_.size());
  std::vector<Gaussian3D> survivors;
  survivors.reserve(gaussian_map_.size());
  for (auto &g : gaussian_map_) {
    const float smax = MaxScaleComponent(g.scale);
    const bool invalid_geom = !g.mean.allFinite() || !g.scale.allFinite()
                              || !std::isfinite(g.opacity) || smax <= 1e-8f
                              || smax > kMaxWorldScale;
    const bool should_prune = invalid_geom || g.opacity < opacity_thr || g.miss_count > miss_thr;
    keep_mask.push_back(should_prune ? 0u : 1u);
    if (!should_prune) {
      survivors.push_back(std::move(g));
    }
  }

  if (survivors.size() == gaussian_map_.size()) {
    return;
  }

#ifdef SIO_USE_LIBTORCH
  if (!PruneTorchOptimizer(keep_mask)) {
    ResetTorchOptimizerState();
  }
#endif

  gaussian_map_ = std::move(survivors);
}

// 对全体高斯进行旋转和平移变换
void GaussianMapManager::OptimizeGaussianPose(const Eigen::Matrix3f &dR, const Eigen::Vector3f &dt,
                                              float step_size) {
  if (gaussian_map_.empty()) {
    return;
  }

  const float step = std::clamp(step_size, 0.0f, 1.0f);
  if (step <= 0.0f) {
    return;
  }

  std::lock_guard<std::mutex> lk(mtx_map_);
  for (auto &g : gaussian_map_) {
    const Eigen::Vector3f target_mean = dR * g.mean + dt;
    g.mean = g.mean + step * (target_mean - g.mean);

    // 地图整体旋转时同步更新每个高斯朝向。
    Eigen::Quaternionf dq(dR);
    if (!std::isfinite(dq.norm()) || dq.norm() < 1e-8f) {
      dq = Eigen::Quaternionf::Identity();
    }
    dq.normalize();
    g.rotation = dq * g.rotation;
    if (!std::isfinite(g.rotation.norm()) || g.rotation.norm() < 1e-8f) {
      g.rotation = Eigen::Quaternionf::Identity();
    } else {
      g.rotation.normalize();
    }
    g.cov3d = CovFromScaleRotation(g.scale, g.rotation);
  }
}

// 根据光度损失自適应优化高斯尺度和不透明度
void GaussianMapManager::OptimizeGaussianParams(double photometric_loss, float min_cov,
                                                float max_cov, float opacity_lr) {
  std::lock_guard<std::mutex> lk(mtx_map_);
  if (gaussian_map_.empty()) {
    return;
  }

#ifndef SIO_USE_LIBTORCH
  (void)photometric_loss;
  (void)min_cov;
  (void)max_cov;
  (void)opacity_lr;
  return;
#else
  const float loss = static_cast<float>(std::max(0.0, photometric_loss));
  const float norm_loss = std::clamp(loss / 30.0f, 0.0f, 1.0f);
  const float cov_min = std::max(1e-8f, min_cov);
  const float cov_max = std::max(cov_min, max_cov);
  const float op_lr = std::clamp(opacity_lr, 0.0f, 1.0f);
  const double lr = std::clamp(0.01 + 0.05 * static_cast<double>(op_lr), 1e-5, 0.2);

  const int n = static_cast<int>(gaussian_map_.size());
  if (!torch_params_ready_ || !torch_param_optim_ || torch_param_count_ != n) {
    BuildTorchOptimizerFromMap(lr);
  } else if (std::abs(lr - torch_param_lr_) > 1e-12) {
    torch_param_lr_ = lr;
    for (auto &group : torch_param_optim_->param_groups()) {
      static_cast<torch::optim::AdamOptions &>(group.options()).lr(lr);
    }
  }

  const float residual = norm_loss - 0.35f;
  const float s_mid = std::sqrt(0.5f * (cov_min + cov_max));
  const auto c = torch::TensorOptions().dtype(torch::kFloat32).device(torch_device_);

  torch_param_optim_->zero_grad();
  //
  auto scale = torch::exp(t_log_scale_);
  auto color = t_color_feature_;
  auto opacity = t_opacity_;

  auto residual_t = torch::full({}, residual, c);
  auto s_mid_t = torch::full({}, s_mid, c);
  // 损失项设计参考 Python 版本，包含趋势项和先验项：
  auto loss_trend = residual_t * opacity.mean() - residual_t * t_log_scale_.mean()
                    + 0.5f * residual_t * (color - 0.5f).pow(2).mean();
  auto loss_prior = 0.01f * (opacity - 0.60f).pow(2).mean()
                    + 0.01f * (scale - s_mid_t).pow(2).mean()
                    + 0.02f * (color - 0.5f).pow(2).mean();
  auto total = loss_trend + loss_prior;

  total.backward();
  torch_param_optim_->step();

  {
    torch::NoGradGuard no_grad;
    const float s_min = std::sqrt(cov_min);
    const float s_max = std::sqrt(cov_max);
    auto smin_t = torch::full({}, s_min, c);
    auto smax_t = torch::full({}, s_max, c);
    auto clamped_scale = torch::exp(t_log_scale_).clamp_min(smin_t).clamp_max(smax_t);
    t_log_scale_.copy_(torch::log(clamped_scale));
    t_color_feature_.clamp_(0.0, 1.0);
    t_opacity_.clamp_(0.02, 1.0);
  }

  const auto scale_cpu = torch::exp(t_log_scale_).detach().to(torch::kCPU).contiguous();
  const auto color_cpu = t_color_feature_.detach().to(torch::kCPU).contiguous();
  const auto op_cpu = t_opacity_.detach().to(torch::kCPU).contiguous();

  const float *s_ptr = scale_cpu.data_ptr<float>();
  const float *c_ptr = color_cpu.data_ptr<float>();
  const float *o_ptr = op_cpu.data_ptr<float>();
  for (int i = 0; i < n; ++i) {
    auto &g = gaussian_map_[i];
    g.scale = Eigen::Vector3f(s_ptr[3 * i + 0], s_ptr[3 * i + 1], s_ptr[3 * i + 2]);
    g.color_feature = Eigen::Vector3f(c_ptr[3 * i + 0], c_ptr[3 * i + 1], c_ptr[3 * i + 2]);
    g.color = g.color_feature;
    g.opacity = o_ptr[i];
    g.cov3d = CovFromScaleRotation(g.scale, g.rotation);
  }
#endif
}

// 判断当前迭代是否应执行统密化和剔毶
bool GaussianMapManager::IsDensifyAndPrune(int densify_interval) {
  if (densify_interval <= 0) {
    return false;
  }
  ++densify_counter_;
  return densify_counter_ % densify_interval == 0;
}

// 返回高斯地图中的点数
size_t GaussianMapManager::Size() const {
  std::lock_guard<std::mutex> lk(mtx_map_);
  return gaussian_map_.size();
}

// 返回高斯地图的线程安全副本
std::vector<Gaussian3D> GaussianMapManager::GetMapSnapshot() const {
  std::lock_guard<std::mutex> lk(mtx_map_);
  return gaussian_map_;
}
