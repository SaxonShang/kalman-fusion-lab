#include "sio_frontend_modules.h"
#include "fusion_debug.h"
#include "sio_pipeline_helpers.h"
#include "stereo_gicp.h"
#include "utils.h"
#include "profiler/simple_profiler.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <pcl/filters/voxel_grid.h>
#include <rclcpp/rclcpp.hpp>

// =============================================================================
// BuildObservationSubclouds
// -----------------------------------------------------------------------------
// 功能：在已变换到目标系的源点云上做 KNN 匹配，筛出法向有效的对应点对，
//       并装配供 ColorPointObserver / ESIKF 使用的对应子点云及附属属性。
//
// 输入：
//   source                    - 源点云（已变换到目标坐标系，如 prevkf body）
//   target_kdtree             - 目标点云的 KD 树（输入云为 target）
//   target_color_gradient     - 目标点颜色梯度（与 target 点一一对应）
//   target_normals            - 目标点法向（与 target 点一一对应；零法向点被丢弃）
//   source_point_confidences  - 源点置信度（可空；不足时按 1.0）
//   target_point_confidences  - 目标点置信度（可空；不足时按 1.0）
//   max_dist_sq               - 最大对应距离的平方 (m²)
//   source_point_covariances  - 源点 3×3 位置协方差（可选）
//   target_point_covariances  - 目标点 3×3 位置协方差（可选）
//
// 输出：
//   source_corr / target_corr - 一一对应的源/目标子点云
//   target_corr_color_gradient / target_corr_normals - 对应点的梯度与法向
//   source_corr_confidences / target_corr_confidences - 对应点置信度
//   source_corr_covariances / target_corr_covariances - 对应点协方差（非空时填充）
//   funnel_out                - 匹配漏斗统计（KNN/距离/法向通过数，可选）
// =============================================================================
void StateEstimator::BuildObservationSubclouds(
    const pcl::PointCloud<PointTypeRGB>::Ptr &source,
    const pcl::KdTreeFLANN<PointTypeRGB>::Ptr &target_kdtree,
    const std::vector<Eigen::Vector3d> &target_color_gradient,
    const std::vector<Eigen::Vector3d> &target_normals,
    const std::vector<float> &source_point_confidences,
    const std::vector<float> &target_point_confidences,
    pcl::PointCloud<PointTypeRGB>::Ptr &source_corr,
    pcl::PointCloud<PointTypeRGB>::Ptr &target_corr,
    std::vector<Eigen::Vector3d> &target_corr_color_gradient,
    std::vector<Eigen::Vector3d> &target_corr_normals,
    std::vector<float> &source_corr_confidences,
    std::vector<float> &target_corr_confidences, float max_dist_sq,
    const std::vector<Eigen::Matrix3d> &source_point_covariances,
    const std::vector<Eigen::Matrix3d> &target_point_covariances,
    std::vector<Eigen::Matrix3d> *source_corr_covariances,
    std::vector<Eigen::Matrix3d> *target_corr_covariances,
    CorrespondenceFunnelStats *funnel_out) {

  if (!source_corr) {
    source_corr.reset(new pcl::PointCloud<PointTypeRGB>());
  }
  if (!target_corr) {
    target_corr.reset(new pcl::PointCloud<PointTypeRGB>());
  }

  source_corr->clear();
  target_corr->clear();
  target_corr_color_gradient.clear();
  target_corr_normals.clear();
  source_corr_confidences.clear();
  target_corr_confidences.clear();
  if (source_corr_covariances) {
    source_corr_covariances->clear();
  }
  if (target_corr_covariances) {
    target_corr_covariances->clear();
  }

  if (!source || source->empty() || !target_kdtree) {
    return;
  }

  const auto &target_cloud = target_kdtree->getInputCloud();
  if (!target_cloud || target_cloud->empty() ||
      target_color_gradient.size() < target_cloud->size() ||
      target_normals.size() < target_cloud->size()) {
    return;
  }

  // ── KNN 搜索与距离过滤：委托 FeatureProcessor::FindCorrespondences ──
  // 消除与 BuildObservationSubclouds 的重复搜索逻辑，统一由 FeatureProcessor
  // 管理。
  FeatureProcessor feature_processor;
  std::vector<int> match_indices;
  CorrespondenceFunnelStats knn_funnel;
  feature_processor.FindCorrespondences(source, target_kdtree, match_indices, max_dist_sq,
                                        &knn_funnel);

  const size_t N = source->size();
  source_corr->reserve(N);
  target_corr->reserve(N);
  target_corr_color_gradient.reserve(N);
  target_corr_normals.reserve(N);
  source_corr_confidences.reserve(N);
  target_corr_confidences.reserve(N);
  if (source_corr_covariances) {
    source_corr_covariances->reserve(N);
  }
  if (target_corr_covariances) {
    target_corr_covariances->reserve(N);
  }

  // ── 子点云装配：从匹配索引中筛选法向有效的对应对 ──
  for (size_t i = 0; i < N; ++i) {
    const int target_index = match_indices[i];
    if (target_index < 0) {
      continue;
    }

    const Eigen::Vector3d &normal = target_normals[target_index];
    if (normal.squaredNorm() < 1e-12) {
      continue;
    }

    source_corr->push_back(source->points[i]);
    target_corr->push_back(target_cloud->points[target_index]);
    target_corr_color_gradient.push_back(target_color_gradient[target_index]);
    target_corr_normals.push_back(normal);

    const float src_conf = (i < source_point_confidences.size())
                               ? source_point_confidences[i]
                               : 1.0f;
    const float tgt_conf =
        (target_index < static_cast<int>(target_point_confidences.size()))
            ? target_point_confidences[target_index]
            : 1.0f;
    source_corr_confidences.push_back(src_conf);
    target_corr_confidences.push_back(tgt_conf);
    if (source_corr_covariances) {
      source_corr_covariances->push_back(
          (i < source_point_covariances.size())
              ? source_point_covariances[i]
              : (Eigen::Matrix3d::Identity() * 1e-4));
    }
    if (target_corr_covariances) {
      target_corr_covariances->push_back(
          (target_index < static_cast<int>(target_point_covariances.size()))
              ? target_point_covariances[target_index]
              : (Eigen::Matrix3d::Identity() * 1e-4));
    }
  }

  source_corr->width = source_corr->size();
  source_corr->height = 1;
  source_corr->is_dense = false;
  target_corr->width = target_corr->size();
  target_corr->height = 1;
  target_corr->is_dense = false;

  if (funnel_out) {
    funnel_out->source_N = knn_funnel.source_N;
    funnel_out->target_N = knn_funnel.target_N;
    funnel_out->knn_hit = knn_funnel.knn_hit;
    funnel_out->dist_pass = knn_funnel.dist_pass;
    funnel_out->normal_pass = source_corr->size();
  }
}

// =============================================================================
// BuildProjectiveObservationSubclouds（Stereo-GICP 投影关联）
// -----------------------------------------------------------------------------
// 功能：将已变换到 prevkf body 的源点转到相机系，按深度自适应半径投影到
//       关键帧深度图，目标点取命中像素中心光线；3σ_Z / 近邻边缘门控。
//       颜色梯度与法向仍从 target KD-tree 最近邻读取（光度项需要）。
// =============================================================================
void StateEstimator::BuildProjectiveObservationSubclouds(
  const pcl::PointCloud<PointTypeRGB>::Ptr &source_body_in_target,
  const pcl::KdTreeFLANN<PointTypeRGB>::Ptr &target_kdtree,
  const std::vector<Eigen::Vector3d> &target_color_gradient,
  const std::vector<Eigen::Vector3d> &target_normals,
  const std::vector<float> &source_point_confidences,
  const std::vector<float> &target_point_confidences, const cv::Mat &target_depth_cam,
  const CameraIntrinsics &cam, const Eigen::Matrix4d &T_body_cam,
  pcl::PointCloud<PointTypeRGB>::Ptr &source_corr,
  pcl::PointCloud<PointTypeRGB>::Ptr &target_corr,
  std::vector<Eigen::Vector3d> &target_corr_color_gradient,
  std::vector<Eigen::Vector3d> &target_corr_normals,
  std::vector<float> &source_corr_confidences, std::vector<float> &target_corr_confidences,
  const std::vector<Eigen::Matrix3d> &source_point_covariances,
  std::vector<Eigen::Matrix3d> *source_corr_covariances,
  std::vector<Eigen::Matrix3d> *target_corr_covariances, CorrespondenceFunnelStats *funnel_out,
  int search_radius, float zmin_m, float zmax_m, float sigma_d_px, float z_gate_sigma,
  float edge_sigma) {

  if (!source_corr) {
    source_corr.reset(new pcl::PointCloud<PointTypeRGB>());
  }
  if (!target_corr) {
    target_corr.reset(new pcl::PointCloud<PointTypeRGB>());
  }
  source_corr->clear();
  target_corr->clear();
  target_corr_color_gradient.clear();
  target_corr_normals.clear();
  source_corr_confidences.clear();
  target_corr_confidences.clear();
  if (source_corr_covariances) {
    source_corr_covariances->clear();
  }
  if (target_corr_covariances) {
    target_corr_covariances->clear();
  }

  if (!source_body_in_target || source_body_in_target->empty() || target_depth_cam.empty()
      || target_depth_cam.type() != CV_32FC1) {
    return;
  }

  const Eigen::Matrix3d R_body_cam = T_body_cam.block<3, 3>(0, 0);
  const Eigen::Vector3d t_body_cam = T_body_cam.block<3, 1>(0, 3);
  const Eigen::Matrix3d R_cam_body = R_body_cam.transpose();
  const Eigen::Vector3d t_cam_body = -R_cam_body * t_body_cam;

  const int width = target_depth_cam.cols;
  const int height = target_depth_cam.rows;
  const size_t N = source_body_in_target->size();

  source_corr->reserve(N);
  target_corr->reserve(N);
  target_corr_color_gradient.reserve(N);
  target_corr_normals.reserve(N);
  source_corr_confidences.reserve(N);
  target_corr_confidences.reserve(N);
  if (source_corr_covariances) {
    source_corr_covariances->reserve(N);
  }
  if (target_corr_covariances) {
    target_corr_covariances->reserve(N);
  }

  pcl::PointCloud<PointTypeRGB> source_cam;
  source_cam.reserve(N);
  std::vector<size_t> cam_to_body;
  cam_to_body.reserve(N);
  for (size_t i = 0; i < N; ++i) {
    const auto &src = source_body_in_target->points[i];
    if (!std::isfinite(src.x) || !std::isfinite(src.y) || !std::isfinite(src.z)) {
      continue;
    }
    const Eigen::Vector3d p_cam =
      R_cam_body * Eigen::Vector3d(src.x, src.y, src.z) + t_cam_body;
    if (!p_cam.allFinite() || p_cam.z() <= 1e-6) {
      continue;
    }
    PointTypeRGB pt = src;
    pt.x = static_cast<float>(p_cam.x());
    pt.y = static_cast<float>(p_cam.y());
    pt.z = static_cast<float>(p_cam.z());
    source_cam.push_back(pt);
    cam_to_body.push_back(i);
  }

  const std::vector<sio_stereo_gicp::ProjectiveMatch> matches =
    sio_stereo_gicp::ProjectiveAssociate(
      source_cam, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), target_depth_cam, cam,
      cv::Mat(), search_radius, z_gate_sigma, edge_sigma, zmin_m, zmax_m);

  const auto &target_cloud = target_kdtree ? target_kdtree->getInputCloud() : nullptr;
  for (const auto &match : matches) {
    if (match.src_index < 0 || match.src_index >= static_cast<int>(cam_to_body.size())) {
      continue;
    }
    const size_t i = cam_to_body[static_cast<size_t>(match.src_index)];
    const auto &src = source_body_in_target->points[i];
    const Eigen::Vector3d q_body = R_body_cam * match.q + t_body_cam;
    if (!q_body.allFinite()) {
      continue;
    }

    PointTypeRGB tgt_pt = src;
    tgt_pt.x = static_cast<float>(q_body.x());
    tgt_pt.y = static_cast<float>(q_body.y());
    tgt_pt.z = static_cast<float>(q_body.z());

    Eigen::Vector3d grad = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
    float tgt_conf = 1.0f;
    if (target_kdtree && target_cloud && !target_cloud->empty()) {
      std::vector<int> nn_idx(1);
      std::vector<float> nn_dist(1);
      if (target_kdtree->nearestKSearch(tgt_pt, 1, nn_idx, nn_dist) > 0) {
        const int ti = nn_idx[0];
        // 属性来自最近的稀疏目标点；离几何目标过远时不借用其颜色、梯度和法向
        const double gate = esikf_options_.attribute_max_distance_m;
        const bool attribute_near = gate <= 0.0 || nn_dist[0] <= gate * gate;
        if (!attribute_near) {
          normal = Eigen::Vector3d::Zero();
        } else if (ti >= 0 && ti < static_cast<int>(target_cloud->size())) {
          tgt_pt.r = target_cloud->points[ti].r;
          tgt_pt.g = target_cloud->points[ti].g;
          tgt_pt.b = target_cloud->points[ti].b;
          if (ti < static_cast<int>(target_color_gradient.size())) {
            grad = target_color_gradient[ti];
          }
          if (ti < static_cast<int>(target_normals.size())
              && target_normals[ti].squaredNorm() > 1e-12) {
            normal = target_normals[ti];
          }
          if (ti < static_cast<int>(target_point_confidences.size())) {
            tgt_conf = target_point_confidences[ti];
          }
        }
      }
    }

    source_corr->push_back(src);
    target_corr->push_back(tgt_pt);
    target_corr_color_gradient.push_back(grad);
    target_corr_normals.push_back(normal);
    const float src_conf =
      (i < source_point_confidences.size()) ? source_point_confidences[i] : 1.0f;
    source_corr_confidences.push_back(src_conf);
    target_corr_confidences.push_back(tgt_conf);

    if (source_corr_covariances) {
      source_corr_covariances->push_back(
        (i < source_point_covariances.size()) ? source_point_covariances[i]
                                              : (Eigen::Matrix3d::Identity() * 1e-4));
    }
    if (target_corr_covariances) {
      const Eigen::Matrix3d Sigma_cam = sio_stereo_gicp::PointCovarianceFromXyz(
        match.q, cam, sio_stereo_gicp::kMatchSigmaUPx, sio_stereo_gicp::kMatchSigmaVPx,
        sigma_d_px);
      Eigen::Matrix3d Sigma_body = R_body_cam * Sigma_cam * R_body_cam.transpose();
      target_corr_covariances->push_back(0.5 * (Sigma_body + Sigma_body.transpose()));
    }
  }

  source_corr->width = source_corr->size();
  source_corr->height = 1;
  source_corr->is_dense = false;
  target_corr->width = target_corr->size();
  target_corr->height = 1;
  target_corr->is_dense = false;

  if (funnel_out) {
    funnel_out->source_N = N;
    funnel_out->target_N =
      (target_cloud && !target_cloud->empty()) ? target_cloud->size() : static_cast<size_t>(width * height);
    funnel_out->knn_hit = source_cam.size();
    funnel_out->dist_pass = matches.size();
    funnel_out->normal_pass = source_corr->size();
  }
}

// =============================================================================
// BuildPointObservationFactor
// -----------------------------------------------------------------------------
// 功能：单点观测因子。几何项按 Stereo-GICP：点对点马氏残差 e=p'-q，
//       C=R Σ_src Rᵀ+Σ_tgt，J 对 ESIKF 误差状态 [δθ_body, δp_world]；
//       光度项仍为点到切平面颜色残差（需有效法向）。
// =============================================================================
bool StateEstimator::BuildPointObservationFactor(
    const PointTypeRGB &source_pt, const PointTypeRGB &target_pt,
    const Eigen::Vector3d &target_color_gradient,
    const Eigen::Vector3d &target_normal,
    const Eigen::Vector3d &source_origin_in_target,
    const Eigen::Matrix3d &R_target_to_world, double source_confidence,
    double target_confidence, PointObservationFactor &factor,
    const Eigen::Matrix3d &R_state_to_target, 
    const Eigen::Vector3d &sensor_origin_in_state,
    const Eigen::Matrix3d *source_cov,
    const Eigen::Matrix3d *target_cov) const {

  // 颜色梯度（FeatureProcessor::PrepareColorGradient）按 RGB 均值计算亮度；
  // uniform_luminance 打开时残差也用均值，否则沿用原来的 BT.709 权重。
  static const Eigen::Vector3d kRec709Weights(0.2126, 0.7152, 0.0722);
  static const Eigen::Vector3d kUniformWeights = Eigen::Vector3d::Constant(1.0 / 3.0);
  const Eigen::Vector3d &rgb_w =
      esikf_options_.uniform_luminance ? kUniformWeights : kRec709Weights;

  factor = PointObservationFactor();

  if (!std::isfinite(source_pt.x) || !std::isfinite(source_pt.y) ||
      !std::isfinite(source_pt.z) || !std::isfinite(target_pt.x) ||
      !std::isfinite(target_pt.y) || !std::isfinite(target_pt.z)) {
    return false;
  }

  // source_pt 已在目标系（p'）；p_src（状态系）由相对原点反变换得到
  const Eigen::Vector3d p_prime(source_pt.x, source_pt.y, source_pt.z);
  const Eigen::Vector3d q_tgt(target_pt.x, target_pt.y, target_pt.z);
  const Eigen::Vector3d q_rel = p_prime - source_origin_in_target;
  const Eigen::Vector3d p_src =
      R_state_to_target.transpose() * q_rel + sensor_origin_in_state;

  // ── Stereo-GICP 几何：e = p' - q，C = R Σ_i Rᵀ + Σ_j ──
  factor.r_geo_vec = p_prime - q_tgt;
  factor.match_dist2 = factor.r_geo_vec.squaredNorm();
  factor.r_geo = std::sqrt(factor.match_dist2);

  // ESIKF：R_curr←R_curr Exp(δθ)，p←p+δp_world
  // de = R_st (δθ × p_src) + R_w_tgtᵀ δp = -R_st [p_src]_× δθ + R_tgt_worldᵀ δp
  factor.J_geo_mat.setZero();
  factor.J_geo_mat.block<3, 3>(0, 0) = -R_state_to_target * svio_utils::SkewSymmetric(p_src);
  factor.J_geo_mat.block<3, 3>(0, 3) = R_target_to_world.transpose();

  Eigen::Matrix3d Sigma_src_tgt = Eigen::Matrix3d::Identity() * 1e-4;
  Eigen::Matrix3d Sigma_tgt = Eigen::Matrix3d::Identity() * 1e-4;
  const bool use_camera_cov =
      source_cov != nullptr && target_cov != nullptr && source_cov->allFinite()
      && target_cov->allFinite();
  if (use_camera_cov) {
    Sigma_src_tgt = R_state_to_target * (*source_cov) * R_state_to_target.transpose();
    Sigma_tgt = *target_cov;
    factor.src_pos_var = source_cov->trace() / 3.0;
    factor.tgt_pos_var = target_cov->trace() / 3.0;
  } else {
    const double src_conf = std::clamp(source_confidence, 1e-3, 1.0);
    const double tgt_conf = std::clamp(target_confidence, 1e-3, 1.0);
    constexpr double kMinPointVar = 1e-4;
    constexpr double kConfVarScale = 4e-3;
    factor.src_pos_var =
        kMinPointVar + kConfVarScale * (1.0 - src_conf) / src_conf;
    factor.tgt_pos_var =
        kMinPointVar + kConfVarScale * (1.0 - tgt_conf) / tgt_conf;
    Sigma_src_tgt = Eigen::Matrix3d::Identity() * factor.src_pos_var;
    Sigma_tgt = Eigen::Matrix3d::Identity() * factor.tgt_pos_var;
  }
  factor.pos_var_sum = factor.src_pos_var + factor.tgt_pos_var;

  Eigen::Matrix3d C = Sigma_src_tgt + Sigma_tgt;
  C += Eigen::Matrix3d::Identity() * 1e-8;
  Eigen::LDLT<Eigen::Matrix3d> C_ldlt(C);
  if (C_ldlt.info() != Eigen::Success) {
    return false;
  }
  factor.W_geo = C_ldlt.solve(Eigen::Matrix3d::Identity());
  if (!factor.W_geo.allFinite()) {
    return false;
  }
  // 一维残差：只用 e 沿方向 u 的分量，信息为秩一 u uᵀ / (uᵀ C u)，与 u 垂直的方向不加约束。
  // Ray 取目标点所在的锚点相机视线；PointToPlane 取目标表面法向，缺少有效法向时不建立该对应。
  if (esikf_options_.geometry_residual != ESIKFOptions::GeometryResidual::PointToPoint) {
    Eigen::Vector3d u = Eigen::Vector3d::Zero();
    if (esikf_options_.geometry_residual == ESIKFOptions::GeometryResidual::Ray) {
      u = q_tgt - ray_origin_in_target_;
    } else if (target_normal.allFinite()) {
      u = target_normal;
    }
    const double u_norm = u.norm();
    if (!(u_norm > 1e-9)) {
      return false;
    }
    u /= u_norm;
    const double variance = u.dot(C * u);
    if (!std::isfinite(variance) || !(variance > 0.0)) {
      return false;
    }
    factor.W_geo = u * u.transpose() / variance;
  }
  factor.has_stereo_geo = true;
  // 标量噪声近似：用于日志兼容；实际累加用 W_geo
  factor.inv_geo_noise = std::max(factor.W_geo.trace() / 3.0, 1e-6);
  factor.J_geo = factor.J_geo_mat.row(2).transpose(); // 占位：沿 Z 分量

  // ── 光度项（点到切平面颜色）；法向无效则跳过颜色 ──
  const double normal_norm = target_normal.norm();
  if (normal_norm >= 1e-6 && target_color_gradient.allFinite() && target_normal.allFinite()) {
    const Eigen::Vector3d n_unit = target_normal / normal_norm;
    const double iq =
        Eigen::Vector3d(source_pt.r, source_pt.g, source_pt.b).dot(rgb_w) / 255.0;
    const double ip =
        Eigen::Vector3d(target_pt.r, target_pt.g, target_pt.b).dot(rgb_w) / 255.0;
    const Eigen::Vector3d dpM =
        target_color_gradient - target_color_gradient.dot(n_unit) * n_unit;

    factor.r_color = ip + dpM.dot(p_prime - q_tgt) - iq;
    factor.J_color.head<3>() =
        p_src.cross(R_state_to_target.transpose() * dpM);
    factor.J_color.tail<3>() = R_target_to_world * dpM;
    factor.grad_strength2 = dpM.squaredNorm();

    double pos_var_color = 0.0;
    if (use_camera_cov) {
      pos_var_color = dpM.dot((Sigma_src_tgt + Sigma_tgt) * dpM);
    } else {
      pos_var_color = factor.pos_var_sum * factor.grad_strength2;
    }
    const double color_texture = 1.0 + 1.0 / (factor.grad_strength2 + 1e-6);
    const double color_noise =
        color_texture * (1.0 + std::abs(factor.r_color)) + pos_var_color;
    factor.inv_color_noise = 1.0 / std::max(color_noise, 1e-6);
  }

  if (!factor.J_geo_mat.allFinite() || !factor.r_geo_vec.allFinite() ||
      !factor.W_geo.allFinite() ||
      (factor.inv_color_noise > 0.0
       && (!factor.J_color.allFinite() || !std::isfinite(factor.r_color)))) {
    return false;
  }

  return true;
}

// =============================================================================
// ColorPointObserver
// -----------------------------------------------------------------------------
// 功能：对全部对应点调用 BuildPointObservationFactor，按几何/光度权重累加
//       6-DOF 位姿观测信息矩阵与信息向量（可 OpenMP 并行）。
//
// 输入：
//   source_corr / target_corr - 一一对应的源/目标子点云
//   target_color_gradient     - 对应目标点颜色梯度（长度 ≥ 对应点数）
//   target_normals            - 对应目标点法向
//   source_origin_in_target   - 源原点在目标系下的平移
//   R_target_to_world         - 目标系 → 世界系旋转
//   source_corr_confidences   - 源对应点置信度（可空）
//   target_corr_confidences   - 目标对应点置信度（可空）
//   lambda                    - 几何权重系数 ∈[0,1]；光度权重为 (1-λ)
//   R_state_to_target         - 状态系 → 目标系旋转
//   sensor_origin_in_state    - 传感器原点在状态系中的位置
//   source_corr_covariances   - 源对应点协方差（可空）
//   target_corr_covariances   - 目标对应点协方差（可空）
//
// 输出（累加到传入引用，调用方应先清零）：
//   H_pose (6×6)              - Σ w·J·Jᵀ，位姿观测信息矩阵
//   b_pose (6×1)              - Σ w·J·r，位姿观测信息向量
// =============================================================================
void StateEstimator::ColorPointObserver(
    const pcl::PointCloud<PointTypeRGB>::Ptr &source_corr,
    const pcl::PointCloud<PointTypeRGB>::Ptr &target_corr,
    const std::vector<Eigen::Vector3d> &target_color_gradient,
    const std::vector<Eigen::Vector3d> &target_normals,
    const Eigen::Vector3d &source_origin_in_target,
    const Eigen::Matrix3d &R_target_to_world,
    Eigen::Matrix<double, 6, 6> &H_pose, Eigen::Matrix<double, 6, 1> &b_pose,
    const std::vector<float> &source_corr_confidences,
    const std::vector<float> &target_corr_confidences, double lambda,
    const Eigen::Matrix3d &R_state_to_target,
    const Eigen::Vector3d &sensor_origin_in_state,
    const std::vector<Eigen::Matrix3d> &source_corr_covariances,
    const std::vector<Eigen::Matrix3d> &target_corr_covariances) {

  if (!source_corr || !target_corr || source_corr->empty() ||
      target_corr->empty())
    return;

  if (source_corr->size() != target_corr->size() ||
      target_color_gradient.size() < source_corr->size() ||
      target_normals.size() < source_corr->size())
    return;

  // 几何/光度融合系数
  lambda = std::max(0.0, std::min(1.0, lambda));

  // 观测信息不再依赖先验标准差，而仅由相对可靠性自适应调节。

  const int cloud_size = static_cast<int>(source_corr->points.size());

  // OpenMP 并行化：每线程独立累加，避免 false sharing 和锁竞争。
  // 非 OpenMP 编译时退化为串行循环。
#ifdef _OPENMP
#pragma omp parallel
  {
    Eigen::Matrix<double, 6, 6> H_thread = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> b_thread = Eigen::Matrix<double, 6, 1>::Zero();

#pragma omp for nowait
    for (int i = 0; i < cloud_size; ++i) {
#else
  for (int i = 0; i < cloud_size; ++i) {
#endif
    const double src_conf =
        (!source_corr_confidences.empty() &&
         i < static_cast<int>(source_corr_confidences.size()))
            ? std::clamp(static_cast<double>(source_corr_confidences[i]), 1e-3,
                         1.0)
            : 1.0;
    const double tgt_conf =
        (!target_corr_confidences.empty() &&
         i < static_cast<int>(target_corr_confidences.size()))
            ? std::clamp(static_cast<double>(target_corr_confidences[i]), 1e-3,
                         1.0)
            : 1.0;

    PointObservationFactor factor;
    const Eigen::Matrix3d *src_cov_ptr = nullptr;
    const Eigen::Matrix3d *tgt_cov_ptr = nullptr;
    if (!source_corr_covariances.empty()
        && i < static_cast<int>(source_corr_covariances.size())) {
      src_cov_ptr = &source_corr_covariances[i];
    }
    if (!target_corr_covariances.empty()
        && i < static_cast<int>(target_corr_covariances.size())) {
      tgt_cov_ptr = &target_corr_covariances[i];
    }
    if (!BuildPointObservationFactor(
            source_corr->points[i], target_corr->points[i],
            target_color_gradient[i], target_normals[i],
            source_origin_in_target, R_target_to_world, src_conf, tgt_conf,
            factor, R_state_to_target, sensor_origin_in_state, src_cov_ptr,
            tgt_cov_ptr)) {
      continue;
    }

    const double mahalanobis_sq =
      factor.has_stereo_geo ? factor.r_geo_vec.dot(factor.W_geo * factor.r_geo_vec) : 0.0;
    const double robust_w = sio_stereo_gicp::HuberWeight(mahalanobis_sq);
    const double w_geo_scale =
      (lambda > 0.0 && factor.has_stereo_geo && robust_w > 0.0) ? lambda * robust_w : 0.0;
    const double w_color_i =
        (1.0 - lambda) > 0.0 ? (1.0 - lambda) * factor.inv_color_noise : 0.0;

#ifdef _OPENMP
    if (factor.has_stereo_geo && w_geo_scale > 0.0) {
      // Stereo-GICP: H += λ Jᵀ W J，b += λ Jᵀ W e
      H_thread.noalias() += w_geo_scale * factor.J_geo_mat.transpose() * factor.W_geo
                            * factor.J_geo_mat;
      b_thread.noalias() += w_geo_scale * factor.J_geo_mat.transpose() * factor.W_geo
                            * factor.r_geo_vec;
    }
    if (w_color_i > 0.0) {
      H_thread.noalias() += w_color_i * factor.J_color * factor.J_color.transpose();
      b_thread.noalias() += w_color_i * factor.J_color * factor.r_color;
    }
#else
    if (factor.has_stereo_geo && w_geo_scale > 0.0) {
      H_pose.noalias() += w_geo_scale * factor.J_geo_mat.transpose() * factor.W_geo
                          * factor.J_geo_mat;
      b_pose.noalias() += w_geo_scale * factor.J_geo_mat.transpose() * factor.W_geo
                          * factor.r_geo_vec;
    }
    if (w_color_i > 0.0) {
      H_pose.noalias() += w_color_i * factor.J_color * factor.J_color.transpose();
      b_pose.noalias() += w_color_i * factor.J_color * factor.r_color;
    }
#endif
  }

#ifdef _OPENMP
#pragma omp critical
  {
    H_pose += H_thread;
    b_pose += b_thread;
  }
  } // end omp parallel
  return;
#endif
}

namespace {

// 与主滤波同一次 K、H、R，Joseph 形式写到并行协方差上。P 用副本自己的先验。
void ApplyPoseJoseph(Eigen::Matrix<double, DIM_STATE, DIM_STATE> *parallel_cov,
                     const Eigen::Matrix<double, DIM_STATE, 6> &K,
                     const Eigen::Matrix<double, 6, DIM_STATE> &H,
                     const Eigen::Matrix<double, 6, 6> &R) {
  if (parallel_cov == nullptr) {
    return;
  }
  const Eigen::Matrix<double, DIM_STATE, DIM_STATE> I_KH =
      Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Identity() - K * H;
  const Eigen::Matrix<double, DIM_STATE, DIM_STATE> updated =
      I_KH * (*parallel_cov) * I_KH.transpose() + K * R * K.transpose();
  *parallel_cov = updated;
}

// 误差状态注入：旋转机体系右乘，其余加法。只改名义状态，不改协方差。
void InjectErrorState(ImuStatesGroup &state, const ImuStatesGroup &base,
                      const Eigen::Matrix<double, DIM_STATE, 1> &dx) {
  state.rot_end = base.rot_end * svio_utils::ExpSO3(dx.segment<3>(0));
  state.pos_end = base.pos_end + dx.segment<3>(3);
  state.vel_end = base.vel_end + dx.segment<3>(6);
  state.bias_g = base.bias_g + dx.segment<3>(9);
  state.bias_a = base.bias_a + dx.segment<3>(12);
  state.gravity = base.gravity + dx.segment<3>(15);
}

} // namespace

EsikfStepResult EsikfIterationStep(const ImuStatesGroup &prior, const ImuStatesGroup &current,
                                   const Eigen::Matrix<double, 6, 1> &virtual_residual,
                                   const Eigen::Matrix<double, 6, 6> &observation_covariance,
                                   bool prior_residual, bool observation_only,
                                   bool update_rotation) {
  EsikfStepResult out;
  out.state = current;

  // e_j = x_j ⊟ x_pred：旋转为机体系右扰动 Log(R_predᵀ R_j)，其余为加法差
  Eigen::Matrix<double, DIM_STATE, 1> offset = Eigen::Matrix<double, DIM_STATE, 1>::Zero();
  offset.segment<3>(0) = svio_utils::LogSO3(prior.rot_end.transpose() * current.rot_end);
  offset.segment<3>(3) = current.pos_end - prior.pos_end;
  offset.segment<3>(6) = current.vel_end - prior.vel_end;
  offset.segment<3>(9) = current.bias_g - prior.bias_g;
  offset.segment<3>(12) = current.bias_a - prior.bias_a;
  offset.segment<3>(15) = current.gravity - prior.gravity;
  out.prior_offset = offset.head<6>();
  if (!virtual_residual.allFinite() || !observation_covariance.allFinite() || !offset.allFinite()) {
    return out;
  }

  if (observation_only) {
    out.gain.topRows<6>().setIdentity();
  } else {
    // K = P Hᵀ (H P Hᵀ + R)⁻¹，H 选取 [δθ, δp]
    const Eigen::Matrix<double, 6, 6> S = prior.cov.block<6, 6>(0, 0) + observation_covariance;
    if (!S.allFinite()) {
      return out;
    }
    const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> S_ldlt(S);
    if (S_ldlt.info() != Eigen::Success) {
      return out;
    }
    out.gain = prior.cov.block<DIM_STATE, 6>(0, 0)
               * S_ldlt.solve(Eigen::Matrix<double, 6, 6>::Identity());
    if (!out.gain.allFinite()) {
      return out;
    }
  }
  if (!update_rotation) {
    // Schmidt：旋转不更新，其余行仍是最优增益
    out.gain.topRows<3>().setZero();
  }

  if (prior_residual && !observation_only) {
    // 线性化点 x_j 处的观测：H (x ⊟ x_j) ≈ r_v，即 H (x ⊟ x_pred) ≈ r_v + H e_j。
    // 对固定先验做一次卡尔曼更新，再从 x_pred 出发注入。
    const Eigen::Matrix<double, DIM_STATE, 1> d = out.gain * (virtual_residual + offset.head<6>());
    InjectErrorState(out.state, prior, d);
    out.pose_step.head<3>() = svio_utils::LogSO3(current.rot_end.transpose() * out.state.rot_end);
    out.pose_step.tail<3>() = out.state.pos_end - current.pos_end;
  } else {
    // 原实现：每轮在 x_j 上直接注入 K r_v
    const Eigen::Matrix<double, DIM_STATE, 1> dx = out.gain * virtual_residual;
    InjectErrorState(out.state, current, dx);
    out.pose_step = dx.head<6>();
  }

  out.ok = out.state.rot_end.allFinite() && out.state.pos_end.allFinite()
           && out.state.vel_end.allFinite() && out.state.bias_g.allFinite()
           && out.state.bias_a.allFinite() && out.state.gravity.allFinite()
           && out.pose_step.allFinite();
  return out;
}

// =============================================================================
// ESIKF（迭代误差状态卡尔曼滤波，紧耦合点云观测）
// -----------------------------------------------------------------------------
// 功能：在 IMU 传播先验上，用当前帧点云对上一关键帧 target（含法向/颜色梯度）
//       做多轮重线性化匹配与观测更新，修正位姿并经交叉协方差传播到速度/偏置等。
//       就地更新 imu_states（均值 + Joseph 形式后验协方差）。
//
// 输入：
//   pointcloud_cur            - 当前帧点云（body 系）
//   target_kdtree             - 关键帧目标点云 KD 树（body 系）
//   target_color_gradient     - 目标颜色梯度（与 KD 树输入云对齐）
//   target_normals            - 目标法向
//   R_anchor / t_anchor       - 锚点（上一关键帧）世界系姿态 R_w_kf、位置 p_w_kf
//   source_point_confidences  - 当前帧点置信度（可空）
//   target_point_confidences  - 目标点置信度（可空）
//   lambda                    - 几何/光度融合系数 ∈[0,1]
//   flow_gate_cov             - 可选。非空时用本步最终 K、H、R 做 Joseph，不改 imu_states.cov
//   T_imu_cam                 - camera→body 外参（投影关联：body↔cam）
//   source_voxel_size         - 源点云体素降采样边长 (m)，点数 >5000 时启用
//   source_point_covariances  - 当前帧点 3×3 协方差（可空）
//   target_point_covariances  - 目标点 3×3 协方差（欧氏关联时用；投影时目标 Σ 现算）
//   esikf_max_correspondence_distance - 欧氏关联最大对应距离 (m)
//   target_depth_cam          - 关键帧左目深度图 (CV_32FC1, m)；非空则投影关联
//   cam                       - 与深度图对齐的推理分辨率内参
//
// 输入/输出：
//   imu_states                - 入：IMU 传播先验；出：ESIKF 后验状态与 cov
//
// 输出：
//   debug_out                 - 可选调试摘要（是否跳过/收敛、对应数、位姿增量等）
// =============================================================================
void StateEstimator::ESIKF(
    const pcl::PointCloud<PointTypeRGB>::Ptr &pointcloud_cur,
    const pcl::KdTreeFLANN<PointTypeRGB>::Ptr &target_kdtree,
    const std::vector<Eigen::Vector3d> &target_color_gradient,
    const std::vector<Eigen::Vector3d> &target_normals,
    ImuStatesGroup &imu_states, const Eigen::Matrix3d &R_anchor,
    const Eigen::Vector3d &t_anchor,
    const std::vector<float> &source_point_confidences,
    const std::vector<float> &target_point_confidences,
    double lambda,
    Eigen::Matrix<double, DIM_STATE, DIM_STATE> *flow_gate_cov,
    const Eigen::Matrix4d &T_imu_cam,
    ESIKFDebugSummary *debug_out,
    double source_voxel_size,
    const std::vector<Eigen::Matrix3d> &source_point_covariances,
    const std::vector<Eigen::Matrix3d> &target_point_covariances,
    double esikf_max_correspondence_distance,
    const cv::Mat &target_depth_cam,
    const CameraIntrinsics &cam
    SIO_PROFILER_PARAMS_IMPL) {

  const auto logger = rclcpp::get_logger("StateEstimator");
  if (debug_out) {
    *debug_out = ESIKFDebugSummary{};
  }

  const ImuStatesGroup imu_before = imu_states;
  RCLCPP_INFO(logger, "Starting ESIKF fusion...");

  if (!pointcloud_cur) {
    RCLCPP_WARN(logger, "ESIKF skipped: pointcloud_cur is null.");
    if (debug_out) {
      debug_out->skipped = true;
    }
    return;
  }

  const int N_orig = static_cast<int>(pointcloud_cur->size());
  if (N_orig == 0 || !target_kdtree) {
    if (debug_out) {
      debug_out->skipped = true;
    }
    return;
  }

  // ── 源点云降采样：减少匹配点数和迭代计算量 ──
  pcl::PointCloud<PointTypeRGB>::Ptr cloud_downsampled(
      new pcl::PointCloud<PointTypeRGB>());
  std::vector<float> downsampled_confidences;
  std::vector<Eigen::Matrix3d> downsampled_covariances;
  const float kESIKFVoxelLeaf = static_cast<float>(std::max(0.01, source_voxel_size));
  if (N_orig > 5000) {
    std::vector<size_t> downsampled_indices;
    const std::vector<Eigen::Matrix3d> *source_cov_ptr =
        source_point_covariances.empty() ? nullptr : &source_point_covariances;
    if (sio_pipeline::BuildIndexAlignedTargetCloud(
            pointcloud_cur, source_point_confidences, kESIKFVoxelLeaf,
            cloud_downsampled, downsampled_confidences, downsampled_indices,
            source_cov_ptr, &downsampled_covariances)) {
      RCLCPP_INFO(logger,
                  "ESIKF voxel downsampling: %d -> %zu points (leaf=%.2fm)",
                  N_orig, cloud_downsampled->size(), kESIKFVoxelLeaf);
    } else {
      pcl::VoxelGrid<PointTypeRGB> voxel;
      voxel.setInputCloud(pointcloud_cur);
      voxel.setLeafSize(kESIKFVoxelLeaf, kESIKFVoxelLeaf, kESIKFVoxelLeaf);
      voxel.filter(*cloud_downsampled);
      downsampled_confidences.clear();
      downsampled_covariances.clear();
      RCLCPP_WARN(logger,
                  "ESIKF index-aligned downsampling failed, fallback pcl voxel: "
                  "%d -> %zu points",
                  N_orig, cloud_downsampled->size());
    }
  } else {
    *cloud_downsampled = *pointcloud_cur;
    downsampled_confidences = source_point_confidences;
    downsampled_covariances = source_point_covariances;
  }

  const pcl::PointCloud<PointTypeRGB>::Ptr &cloud_for_ikf = cloud_downsampled;
  const std::vector<float> &confidences_for_ikf = downsampled_confidences;
  const std::vector<Eigen::Matrix3d> &covariances_for_ikf = downsampled_covariances;
  const int N = static_cast<int>(cloud_for_ikf->size());
  const float max_dist_sq = static_cast<float>(
    esikf_max_correspondence_distance * esikf_max_correspondence_distance);
  CorrespondenceFunnelStats last_funnel;
  if (debug_out) {
    debug_out->source_N = static_cast<std::size_t>(N);
    debug_out->target_N =
      target_kdtree && target_kdtree->getInputCloud() ? target_kdtree->getInputCloud()->size() : 0;
    debug_out->max_corr_m = esikf_max_correspondence_distance;
  }

  // ══════════════════════════════════════════════════════════════════════════
  // LIVO2-style 迭代误差状态卡尔曼滤波器 (iESKF)
  //
  // 误差状态 δx ∈ R^{DIM_STATE}（21 维）:
  //   [δφ(0:3), δp(3:6), δv(6:9), δbg(9:12), δba(12:15), δg(15:18), ...]
  //
  // ColorPointObserver 输出观测信息（汇聚所有点）:
  //   H_obs = Σ Jᵢᵀ Rᵢ⁻¹ Jᵢ  (6×6, 观测信息矩阵)
  //   b_obs = Σ Jᵢᵀ Rᵢ⁻¹ rᵢ  (6×1, 观测信息向量)
  //
  // 等价为单次 6-DOF "虚拟测量":
  //   观测噪声协方差: R_obs = (H_obs + εI)⁻¹         (6×6)
  //   虚拟观测残差:   r_v   = -(H_obs + εI)⁻¹ b_obs  (6×1)
  //
  // 标准卡尔曼更新（由矩阵反演引理得到，每轮在当前线性化点 x̂_k 处展开）:
  //   K       = P_pred Hᵀ_sel (H_sel P_pred Hᵀ_sel + R_obs)⁻¹ ← 卡尔曼增益
  //   (21×1) delta_x = K r_v                                            ←
  //   当前重线性化点下的本轮增量
  //   P_post  = (I-KH) P_pred (I-KH)ᵀ + K R_obs Kᵀ                     ←
  //   Joseph 协方差
  //
  // H_sel (6×21) 从 21 维误差状态中选取 [δφ, δp]
  // K 通过 P 的非对角交叉协方差，隐式将位姿修正传播到速度/偏置/重力
  //
  // 默认行为保持原实现；ESIKFOptions 可逐项打开（单轮更新见 EsikfIterationStep）：
  //   prior_residual    x_{j+1} = x_pred ⊞ K (r_v + H e_j)，e_j = x_j ⊟ x_pred
  //   observation_scale R_obs ← D R_obs D
  //   stop_*            旋转、平移分开判断收敛
  // 迭代在状态副本上进行；没有成功的迭代或出现数值失败时保留先验均值与协方差。
  // ══════════════════════════════════════════════════════════════════════════

  // ── 原始先验（iEKF 迭代过程中固定不变）与迭代副本 ──
  const ESIKFOptions &options = esikf_options_;
  const ImuStatesGroup prior_state = imu_states;
  ImuStatesGroup state = imu_states;
  const Eigen::Matrix<double, DIM_STATE, DIM_STATE> &P_pred = prior_state.cov; // (21×21)
  // 视线残差的相机光心：锚点左目光心在锚点机体系中的位置
  SetRayOriginInTarget(T_imu_cam.block<3, 1>(0, 3));

  // ── 观测选择矩阵 H_sel (6×21): 前 6 列为 I，其余为 0 ──
  // 显式定义仅用于协方差更新中的 K·H_sel 运算
  Eigen::Matrix<double, 6, DIM_STATE> H_sel =
      Eigen::Matrix<double, 6, DIM_STATE>::Zero();
  H_sel.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity(); // δφ → 旋转
  H_sel.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity(); // δp → 平移

  // 预分配重线性化与对应关系缓冲区
  pcl::PointCloud<PointTypeRGB>::Ptr source_trans(
      new pcl::PointCloud<PointTypeRGB>());
  source_trans->points.resize(N);
  pcl::PointCloud<PointTypeRGB>::Ptr source_corr(
      new pcl::PointCloud<PointTypeRGB>());
  pcl::PointCloud<PointTypeRGB>::Ptr target_corr(
      new pcl::PointCloud<PointTypeRGB>());
  std::vector<Eigen::Vector3d> target_corr_color_gradient;
  std::vector<Eigen::Vector3d> target_corr_normals;
  std::vector<float> source_corr_confidences;
  std::vector<float> target_corr_confidences;
  std::vector<Eigen::Matrix3d> source_corr_covariances;
  std::vector<Eigen::Matrix3d> target_corr_covariances;

  // 最后一次成功迭代的 K 和 R_obs 成对保存，结束时用于 Joseph 形式更新 P_post
  Eigen::Matrix<double, DIM_STATE, 6> K_last =
      Eigen::Matrix<double, DIM_STATE, 6>::Zero(); // (21×6)
  Eigen::Matrix<double, 6, 6> R_last =
      Eigen::Matrix<double, 6, 6>::Identity(); // (6×6)
  bool is_converged = false;
  bool has_update = false;
  bool numerical_failure = false;
  std::string stop_reason = "max_iterations";
  int num_iters = 0;
  double last_delta_norm = std::numeric_limits<double>::infinity();
  double last_pose_delta_norm = std::numeric_limits<double>::infinity();
  double last_virtual_residual_norm = std::numeric_limits<double>::infinity();
  size_t last_corr_count = 0;
  constexpr int kMinCorrespondences = 20;

  const int max_iter = std::max(1, options.max_iterations);
  const bool component_stop = options.stop_rotation_rad > 0.0 && options.stop_translation_m > 0.0;
  Eigen::Matrix<double, 6, 1> observation_scale_sqrt;
  observation_scale_sqrt.head<3>().setConstant(
    std::sqrt(options.observation_scale_rotation > 0.0 ? options.observation_scale_rotation : 1.0));
  observation_scale_sqrt.tail<3>().setConstant(
    std::sqrt(options.observation_scale_translation > 0.0 ? options.observation_scale_translation
                                                          : 1.0));

  for (int iter = 0; iter < max_iter; ++iter) {
    // ── 1. 重线性化：源点在 curr body，目标在 prevkf body ──
    const Eigen::Matrix3d R_curr_to_prevkf =
        R_anchor.transpose() * state.rot_end;
    const Eigen::Vector3d t_curr_in_prevkf =
        R_anchor.transpose() * (state.pos_end - t_anchor);
    const Eigen::Matrix3d R_state_to_target = R_curr_to_prevkf;
    {
      SIO_PROFILE((*sio_prof), sio_frame_id, "esikf", "transform_source");
#ifdef _OPENMP
#pragma omp parallel for
      for (int i = 0; i < N; ++i) {
        const auto &src = cloud_for_ikf->points[i];
        const Eigen::Vector3d p_prevkf =
            R_curr_to_prevkf * Eigen::Vector3d(src.x, src.y, src.z) +
            t_curr_in_prevkf;
        source_trans->points[i] = src;
        source_trans->points[i].x = static_cast<float>(p_prevkf.x());
        source_trans->points[i].y = static_cast<float>(p_prevkf.y());
        source_trans->points[i].z = static_cast<float>(p_prevkf.z());
      }
#else
      for (int i = 0; i < N; ++i) {
        const auto &src = cloud_for_ikf->points[i];
        const Eigen::Vector3d p_prevkf =
            R_curr_to_prevkf * Eigen::Vector3d(src.x, src.y, src.z) +
            t_curr_in_prevkf;
        source_trans->points[i] = src;
        source_trans->points[i].x = static_cast<float>(p_prevkf.x());
        source_trans->points[i].y = static_cast<float>(p_prevkf.y());
        source_trans->points[i].z = static_cast<float>(p_prevkf.z());
      }
#endif
    }

    // ── 2. 建立对应关系：优先 Stereo-GICP 投影关联，否则欧氏 KD-tree ──
    {
      SIO_PROFILE((*sio_prof), sio_frame_id, "esikf", "correspondence");
      const bool use_projective =
        !target_depth_cam.empty() && target_depth_cam.type() == CV_32FC1 && cam.fx > 1e-3f
        && cam.baseline > 1e-6f;
      if (use_projective) {
        BuildProjectiveObservationSubclouds(
          source_trans, target_kdtree, target_color_gradient, target_normals,
          confidences_for_ikf, target_point_confidences, target_depth_cam, cam, T_imu_cam,
          source_corr, target_corr, target_corr_color_gradient, target_corr_normals,
          source_corr_confidences, target_corr_confidences, covariances_for_ikf,
          &source_corr_covariances, &target_corr_covariances, &last_funnel);
      } else {
        BuildObservationSubclouds(
          source_trans, target_kdtree, target_color_gradient, target_normals,
          confidences_for_ikf, target_point_confidences, source_corr, target_corr,
          target_corr_color_gradient, target_corr_normals, source_corr_confidences,
          target_corr_confidences, max_dist_sq, covariances_for_ikf, target_point_covariances,
          &source_corr_covariances, &target_corr_covariances, &last_funnel);
      }
    }

    if (!source_corr ||
        static_cast<int>(source_corr->size()) < kMinCorrespondences) {
      RCLCPP_WARN(
          logger, "ESIKF iter %d skipped: too few correspondences (%zu < %d).",
          iter, source_corr ? source_corr->size() : 0, kMinCorrespondences);
      stop_reason = has_update ? "correspondences_lost" : "too_few_correspondences";
      break;
    }
    last_corr_count = source_corr->size();

    // ── 3. 构建观测信息 (ColorPointObserver) ──
    Eigen::Matrix<double, 6, 6> H_obs =
        Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> b_obs =
        Eigen::Matrix<double, 6, 1>::Zero();
    {
      SIO_PROFILE((*sio_prof), sio_frame_id, "esikf", "factor_accumulate");
      ColorPointObserver(source_corr, target_corr, target_corr_color_gradient,
                         target_corr_normals, t_curr_in_prevkf, R_anchor,
                         H_obs, b_obs, source_corr_confidences,
                         target_corr_confidences, lambda, R_state_to_target,
                         Eigen::Vector3d::Zero(), source_corr_covariances,
                         target_corr_covariances);
    }

    // ── 4. 求解 6×6 虚拟观测：R_obs = H_obs⁻¹，r_v = −H_obs⁻¹ b_obs ──
    Eigen::Matrix<double, 6, 1> r_v;
    Eigen::Matrix<double, 6, 6> R_obs;
    {
      SIO_PROFILE((*sio_prof), sio_frame_id, "esikf", "solve_6x6");
      const double damping = 1e-6 * std::max(1.0, H_obs.trace() / 6.0);
      const Eigen::Matrix<double, 6, 6> H_obs_reg =
          H_obs + damping * Eigen::Matrix<double, 6, 6>::Identity();
      if (!H_obs_reg.allFinite()) {
        RCLCPP_WARN(logger, "H_obs contains invalid values at iter %d", iter);
        numerical_failure = true;
        break;
      }
      const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> H_ldlt(H_obs_reg);
      if (H_ldlt.info() != Eigen::Success) {
        RCLCPP_WARN(logger, "H_obs_reg decomposition failed at iter %d", iter);
        numerical_failure = true;
        break;
      }
      R_obs = H_ldlt.solve(Eigen::Matrix<double, 6, 6>::Identity());
      // 观测协方差标定：旋转、平移块分别放大，保留块间相关
      R_obs = observation_scale_sqrt.asDiagonal() * R_obs * observation_scale_sqrt.asDiagonal();
      if (!R_obs.allFinite()) {
        RCLCPP_WARN(logger, "R_obs contains invalid values at iter %d", iter);
        numerical_failure = true;
        break;
      }

      r_v = -H_ldlt.solve(b_obs);
      if (!r_v.allFinite()) {
        RCLCPP_WARN(logger, "Virtual residual contains invalid values at iter %d",
                    iter);
        numerical_failure = true;
        break;
      }
      last_virtual_residual_norm = r_v.norm();
    }

    // ── 5. iEKF 状态更新（K 用固定先验 P）──
    EsikfStepResult step;
    {
      SIO_PROFILE((*sio_prof), sio_frame_id, "esikf", "inject_state");
      step = EsikfIterationStep(prior_state, state, r_v, R_obs, options.prior_residual,
                                options.observation_only, options.update_rotation);
    }
    if (!step.ok) {
      RCLCPP_WARN(logger, "ESIKF state update invalid at iter %d", iter);
      numerical_failure = true;
      break;
    }
    state = step.state;
    K_last = step.gain;
    R_last = R_obs;
    has_update = true;
    last_delta_norm = (step.gain * r_v).norm();
    last_pose_delta_norm = step.pose_step.norm();

    if (debug_out) {
      ESIKFIterationDebug record;
      record.iteration = iter;
      record.correspondences = last_corr_count;
      record.information = H_obs;
      record.observation_covariance = R_obs;
      record.pose_gain = step.gain.topRows<6>();
      record.virtual_residual = r_v;
      record.pose_correction = step.pose_step;
      record.prior_offset = step.prior_offset;
      debug_out->iterations.push_back(std::move(record));
    }
    RCLCPP_INFO(logger,
                "ESIKF iter %d: corr=%zu r_v=%.6e delta_pose=%.6e "
                "delta_state=%.6e",
                iter, last_corr_count, last_virtual_residual_norm,
                last_pose_delta_norm, last_delta_norm);

    num_iters = iter + 1;
    const bool step_small =
      component_stop
        ? (step.pose_step.head<3>().norm() < options.stop_rotation_rad
           && step.pose_step.tail<3>().norm() < options.stop_translation_m)
        : last_pose_delta_norm < 1e-3;
    if (step_small) {
      is_converged = true;
      stop_reason = "converged";
      break;
    }
  }
  if (numerical_failure) {
    stop_reason = "numerical_failure";
  }

  if (is_converged)
    RCLCPP_INFO(logger, "ESIKF successfully converged.");
  else
    RCLCPP_WARN(logger,
                "ESIKF stopped without convergence: %s "
                "(last_corr=%zu last_r_v=%.6e last_delta_pose=%.6e "
                "last_delta_state=%.6e).",
                stop_reason.c_str(), last_corr_count, last_virtual_residual_norm,
                last_pose_delta_norm, last_delta_norm);

  // ── 7. 验收与后验协方差 P_post（Joseph 形式，K 与 R_obs 取同一轮）──
  // 数值失败或没有任何成功迭代：保留先验均值与协方差，不写入半成品状态。
  const bool accepted = has_update && !numerical_failure;
  if (accepted) {
    SIO_PROFILE((*sio_prof), sio_frame_id, "esikf", "covariance_update");
    const Eigen::Matrix<double, DIM_STATE, DIM_STATE> I_KH =
        Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Identity() -
        K_last * H_sel;
    state.cov =
        I_KH * P_pred * I_KH.transpose() +
        K_last * R_last * K_last.transpose();
    ApplyPoseJoseph(flow_gate_cov, K_last, H_sel, R_last);
    imu_states = state;
  } else {
    imu_states = prior_state;
  }

  if (debug_out) {
    debug_out->converged = is_converged;
    debug_out->updated = accepted;
    debug_out->status = stop_reason;
    debug_out->num_iters = num_iters;
    debug_out->final_corr = last_corr_count;
    debug_out->knn_hit = last_funnel.knn_hit;
    debug_out->dist_pass = last_funnel.dist_pass;
    debug_out->virtual_residual_norm = last_virtual_residual_norm;
    ComputeImuStatePoseDelta(imu_before, imu_states,
                             &debug_out->pose_delta_m,
                             &debug_out->pose_delta_deg);
  }

}

// =============================================================================
// ESKF（松耦合：IMU 预积分预测 + GICP 绝对位姿观测）
// -----------------------------------------------------------------------------
// 功能：用 GICP 给出的世界系 6-DOF 位姿观测，对 IMU 传播先验做一次误差状态
//       卡尔曼更新；通过交叉协方差将位姿修正传播到速度/偏置/重力。
//
// 误差状态（21 维）：
//   [δθ(3), δp(3), δv(3), δbg(3), δba(3), δg(3), ...]
// 观测创新：δθ = Log(R_predᵀ R_gicp)，δp = p_gicp - p_pred
//
// 输入：
//   imu_propagated - IMU 传播后的先验状态（含 21×21 协方差 P_pred）
//   T_gicp         - GICP 绝对位姿观测（世界系 4×4，float）
//   R_gicp         - 观测噪声协方差 6×6（已对齐到 ESKF 混合坐标 [δθ_body, δp_world]）
//   flow_gate_cov  - 可选。非空时用同一次 K、H、R 做 Joseph，不改 fused.cov
//
// 返回：
//   fused          - 融合后的后验 ImuStatesGroup（位姿/速度/偏置/重力 + Joseph cov）
// =============================================================================
ImuStatesGroup StateEstimator::ESKF(
    const ImuStatesGroup &imu_propagated, const Eigen::Matrix4f &T_gicp,
    const Eigen::Matrix<double, 6, 6> &R_gicp,
    Eigen::Matrix<double, DIM_STATE, DIM_STATE> *flow_gate_cov,
    ESKFDebugSummary *debug_out) {

  ImuStatesGroup fused = imu_propagated;

  const Eigen::Matrix3d R_gicp_rot = T_gicp.block<3, 3>(0, 0).cast<double>();
  const Eigen::Vector3d p_gicp_pos = T_gicp.block<3, 1>(0, 3).cast<double>();
  const Eigen::Matrix3d dR = imu_propagated.rot_end.transpose() * R_gicp_rot;
  const Eigen::Vector3d delta_theta = svio_utils::So3LogApprox(dR);
  const Eigen::Vector3d delta_p = p_gicp_pos - imu_propagated.pos_end;

  Eigen::Matrix<double, 6, 1> z_innov;
  z_innov.head<3>() = delta_theta;
  z_innov.tail<3>() = delta_p;

  Eigen::Matrix<double, 6, DIM_STATE> H_sel = Eigen::Matrix<double, 6, DIM_STATE>::Zero();
  H_sel.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  H_sel.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();

  const auto &P_pred = imu_propagated.cov;
  const Eigen::Matrix<double, 6, 6> &R_obs = R_gicp;
  const Eigen::Matrix<double, 6, 6> P_pose = H_sel * P_pred * H_sel.transpose();
  const Eigen::Matrix<double, 6, 6> S = 0.5 *
    (P_pose + R_obs + (P_pose + R_obs).transpose());
  if (!S.allFinite() || (S.diagonal().array() <= 0.0).any())
    throw std::invalid_argument("ESKF innovation covariance must have finite positive diagonal");
  // S contains both radians^2 and metres^2. Solve in dimensionless coordinates
  // so a large position variance cannot inject artificial angular noise.
  const Eigen::Matrix<double, 6, 1> scales = S.diagonal().cwiseSqrt();
  const Eigen::Matrix<double, 6, 1> inverse_scales = scales.cwiseInverse();
  const Eigen::Matrix<double, 6, 6> normalized =
    inverse_scales.asDiagonal() * S * inverse_scales.asDiagonal();
  Eigen::Matrix<double, 6, 6> normalized_regularized;
  Eigen::LDLT<Eigen::Matrix<double, 6, 6>> solver;
  double damping = 0.0;
  bool solved = false;
  for (int attempt = 0; attempt < 8; ++attempt) {
    normalized_regularized = normalized +
      damping * Eigen::Matrix<double, 6, 6>::Identity();
    solver.compute(normalized_regularized);
    if (solver.info() == Eigen::Success && solver.isPositive() &&
        solver.vectorD().minCoeff() > 64 * std::numeric_limits<double>::epsilon() *
                                       solver.vectorD().maxCoeff()) {
      solved = true;
      break;
    }
    damping = damping == 0.0 ? 1e-12 : damping * 10;
  }
  if (!solved) throw std::runtime_error("ESKF innovation covariance is not positive definite");
  const Eigen::Matrix<double, DIM_STATE, 6> scaled_cross =
    (P_pred * H_sel.transpose()) * inverse_scales.asDiagonal();
  const Eigen::Matrix<double, DIM_STATE, 6> K =
    solver.solve(scaled_cross.transpose()).transpose() * inverse_scales.asDiagonal();
  const Eigen::Matrix<double, 6, 6> S_reg =
    scales.asDiagonal() * normalized_regularized * scales.asDiagonal();

  const Eigen::Matrix<double, DIM_STATE, 1> delta_x = K * z_innov;
  if (debug_out) {
    debug_out->innovation = z_innov;
    debug_out->observation_covariance = R_obs;
    debug_out->innovation_covariance = S_reg;
    debug_out->innovation_scale = scales;
    debug_out->normalized_innovation_covariance = normalized_regularized;
    debug_out->gain = K;
    debug_out->correction = delta_x;
    debug_out->damping = damping;
    const Eigen::Matrix<double, 6, 1> standardized = inverse_scales.asDiagonal() * z_innov;
    debug_out->nis_regularized = standardized.dot(solver.solve(standardized));
  }
  fused.rot_end = imu_propagated.rot_end * svio_utils::ExpSO3(delta_x.segment<3>(0));
  fused.pos_end = imu_propagated.pos_end + delta_x.segment<3>(3);
  fused.vel_end = imu_propagated.vel_end + delta_x.segment<3>(6);
  fused.bias_g = imu_propagated.bias_g + delta_x.segment<3>(9);
  fused.bias_a = imu_propagated.bias_a + delta_x.segment<3>(12);
  fused.gravity = imu_propagated.gravity + delta_x.segment<3>(15);

  const Eigen::Matrix<double, DIM_STATE, DIM_STATE> I_KH =
    Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Identity() - K * H_sel;
  fused.cov = I_KH * P_pred * I_KH.transpose() + K * R_obs * K.transpose();
  ApplyPoseJoseph(flow_gate_cov, K, H_sel, R_obs);

  return fused;
}
