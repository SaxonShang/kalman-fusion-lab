#pragma once

#include "data.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#ifdef SIO_USE_LIBTORCH
#  include <torch/optim.h>
#  include <torch/torch.h>
#endif

struct Gaussian3D {
  Eigen::Vector3f mean;
  Eigen::Matrix3f cov3d;

  Eigen::Vector3f color;         // 兼容字段：RGB 颜色
  Eigen::Vector3f color_feature; // 3DGS 颜色特征（当前采用 RGB）
  float opacity = 1.0f;
  Eigen::Vector3f scale = Eigen::Vector3f::Constant(0.01f);
  Eigen::Quaternionf rotation = Eigen::Quaternionf::Identity();

  int id = -1;
  int source_kf = -1;
  bool is_active = true;
  int hit_count = 0;
  int miss_count = 0;

  // 稠密化统计：由渲染反传梯度累计得到。
  float densify_grad_accum = 0.0f;
  int densify_grad_denom = 0;
};

class GaussianMapManager {
public:
  GaussianMapManager() = default;

  // 初始化地图：清空旧地图并用当前关键帧种子化高斯点。
  void InitializeFromKeyframe(const std::shared_ptr<KeyFrameData> &kf, float init_size);
  // 增量更新：在现有地图上追加当前关键帧对应的高斯点。
  void AddFromKeyframe(const std::shared_ptr<KeyFrameData> &kf, float init_size);
  void Densify(float densify_grad_thr, int min_obs_count = 1);
  void Prune(float opacity_prune_thr, int miss_prune_thr);
  void AccumulateDensificationStats(const std::vector<float> &xyz_grad_norm);

  // 基于窗口优化结果对高斯整体位姿做小步更新。
  void OptimizeGaussianPose(const Eigen::Matrix3f &dR, const Eigen::Vector3f &dt, float step_size);
  // 基于光度损失对高斯参数（协方差/不透明度）做自适应更新。
  void OptimizeGaussianParams(double photometric_loss, float min_cov, float max_cov,
                              float opacity_lr);

  bool IsDensifyAndPrune(int densify_interval);
  size_t Size() const;
  std::vector<Gaussian3D> GetMapSnapshot() const;

private:
  static Eigen::Vector3f ToWorldPoint(const ImuStatesGroup &state, const Eigen::Vector3f &p_local);
  void UpdatePreviousKeyframeCache(const std::shared_ptr<KeyFrameData> &kf);

  mutable std::mutex mtx_map_;
  std::vector<Gaussian3D> gaussian_map_;
  ImuStatesGroup prev_kf_state_;
  bool has_prev_kf_cache_ = false;
  int next_gaussian_id_ = 0;
  int densify_counter_ = 0;

#ifdef SIO_USE_LIBTORCH
  // LibTorch 内部优化器状态（参数数量变化时自动重建）。
  void ResetTorchOptimizerState();
  void BuildTorchOptimizerFromMap(double lr);
  bool PruneTorchOptimizer(const std::vector<uint8_t> &keep_mask);
  bool AppendTorchOptimizer(const std::vector<Gaussian3D> &additions);

  bool torch_params_ready_ = false;
  int torch_param_count_ = 0;
  torch::Device torch_device_{torch::kCPU};
  double torch_param_lr_ = 0.02;
  torch::Tensor t_log_scale_;
  torch::Tensor t_color_feature_;
  torch::Tensor t_opacity_;
  std::unique_ptr<torch::optim::Adam> torch_param_optim_;
#endif
};
