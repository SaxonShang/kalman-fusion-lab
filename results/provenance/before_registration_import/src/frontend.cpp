#include "frontend.h"

#include "data.h"
#include "frontend_debug_visualizer.h"
#include "dataset_loader.h"
#include "sio_frontend_modules.h"
#include "sio_pipeline_helpers.h"
#include "utils.h"

#include <Eigen/Dense>
#include <array>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

  // 将点云变换到机体系下
  void TransformPointCloudInPlace(const Eigen::Affine3d &T_body_cam,
                                  pcl::PointCloud<PointTypeRGB>::Ptr &cloud) {
    if (!cloud || cloud->empty()) {
      return;
    }

    for (auto &pt : cloud->points) {
      const Eigen::Vector3d p_body = T_body_cam * Eigen::Vector3d(pt.x, pt.y, pt.z);
      pt.x = static_cast<float>(p_body.x());
      pt.y = static_cast<float>(p_body.y());
      pt.z = static_cast<float>(p_body.z());
    }
  }
  // 判断点云点是否满足跟踪目标的有效性要求（深度范围、置信度、坐标有效性）
  bool IsTrackingPointValid(const PointTypeRGB &point, float confidence,
                            const sio_utils::TrackingTargetParams &cfg) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
      return false;
    }
    if (point.z < cfg.min_depth_m || point.z > cfg.max_depth_m) {
      return false;
    }
    if (!std::isfinite(confidence) || confidence <= cfg.min_confidence) {
      return false;
    }
    return true;
  }

  // 体素下采样：与 ESIKF target_cache 一致，先过滤有效点再按体素网格保留代表点。
  TrackingTargetData BuildVoxelTrackingTargetData(
    const pcl::PointCloud<PointTypeRGB>::Ptr &cloud, const std::vector<float> &point_confidences,
    const std::vector<Eigen::Matrix3d> &point_covariances,
    const sio_utils::TrackingTargetParams &cfg, size_t *depth_valid_count_out = nullptr) {
    TrackingTargetData result;
    result.cloud.reset(new pcl::PointCloud<PointTypeRGB>());
    if (depth_valid_count_out) {
      *depth_valid_count_out = 0;
    }

    if (!cloud || cloud->empty() || !(cfg.voxel_size_m > 0.0f)) {
      result.cloud->width = 0;
      result.cloud->height = 1;
      result.cloud->is_dense = true;
      return result;
    }

    pcl::PointCloud<PointTypeRGB>::Ptr filtered_cloud(new pcl::PointCloud<PointTypeRGB>());
    std::vector<float> filtered_confidences;
    std::vector<Eigen::Matrix3d> filtered_covariances;
    std::vector<size_t> filtered_original_indices;
    filtered_cloud->points.reserve(cloud->size());
    filtered_confidences.reserve(cloud->size());
    filtered_covariances.reserve(cloud->size());
    filtered_original_indices.reserve(cloud->size());

    for (size_t i = 0; i < cloud->size(); ++i) {
      const float conf = (i < point_confidences.size()) ? point_confidences[i] : 1.0f;
      if (!IsTrackingPointValid(cloud->points[i], conf, cfg)) {
        continue;
      }
      filtered_cloud->points.push_back(cloud->points[i]);
      filtered_confidences.push_back(conf);
      filtered_covariances.push_back((i < point_covariances.size())
                                       ? point_covariances[i]
                                       : (Eigen::Matrix3d::Identity() * 1e-4));
      filtered_original_indices.push_back(i);
    }

    if (depth_valid_count_out) {
      *depth_valid_count_out = filtered_cloud->size();
    }

    if (filtered_cloud->empty()) {
      result.cloud->width = 0;
      result.cloud->height = 1;
      result.cloud->is_dense = true;
      return result;
    }

    pcl::PointCloud<PointTypeRGB>::Ptr voxel_cloud;
    std::vector<size_t> voxel_source_indices;
    if (!sio_pipeline::BuildIndexAlignedTargetCloud(
          filtered_cloud, filtered_confidences, cfg.voxel_size_m, voxel_cloud,
          result.point_confidences, voxel_source_indices)
        || !voxel_cloud || voxel_cloud->empty()) {
      result.cloud->width = 0;
      result.cloud->height = 1;
      result.cloud->is_dense = true;
      return result;
    }

    result.cloud = voxel_cloud;
    result.point_covariances.reserve(voxel_source_indices.size());
    result.source_indices.reserve(voxel_source_indices.size());
    for (const size_t filtered_idx : voxel_source_indices) {
      if (filtered_idx < filtered_covariances.size()) {
        result.point_covariances.push_back(filtered_covariances[filtered_idx]);
      } else {
        result.point_covariances.push_back(Eigen::Matrix3d::Identity() * 1e-4);
      }
      if (filtered_idx < filtered_original_indices.size()) {
        result.source_indices.push_back(filtered_original_indices[filtered_idx]);
      }
    }

    result.cloud->width = static_cast<uint32_t>(result.cloud->points.size());
    result.cloud->height = 1;
    result.cloud->is_dense = cloud->is_dense;
    return result;
  }

  pcl::PointCloud<PointTypeRGB>::Ptr BuildCloudByIndices(
    const pcl::PointCloud<PointTypeRGB>::Ptr &cloud, const std::vector<int> &indices) {
    auto filtered = pcl::PointCloud<PointTypeRGB>::Ptr(new pcl::PointCloud<PointTypeRGB>());
    if (!cloud || cloud->empty() || indices.empty()) {
      return filtered;
    }
    filtered->points.reserve(indices.size());
    for (const int idx : indices) {
      if (idx >= 0 && idx < static_cast<int>(cloud->size())) {
        filtered->points.push_back(cloud->points[static_cast<size_t>(idx)]);
      }
    }
    filtered->width = static_cast<uint32_t>(filtered->points.size());
    filtered->height = 1;
    filtered->is_dense = cloud->is_dense;
    return filtered;
  }

} // namespace

SIOFrontend::SIOFrontend(SVIO_SLAM *slam)
    : slam_(slam), is_running_(false),
      flow_gate_cov_(Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Zero()),
      pointcloud_cur_(new pcl::PointCloud<PointTypeRGB>()) {
  const auto project_cfg = sio_utils::LoadProjectRuntimeConfig();
  if (project_cfg.plan_disp.empty() || project_cfg.plan_raft.empty()) {
    throw std::runtime_error("plan_disp and plan_raft must be configured.");
  }
  if (!std::filesystem::exists(project_cfg.plan_disp)
      || !std::filesystem::exists(project_cfg.plan_raft)) {
    throw std::runtime_error(
      "TensorRT plan file not found: disp=" + project_cfg.plan_disp.string() + " raft="
      + project_cfg.plan_raft.string());
  }
  const CameraIntrinsics cam_inference =
    sio_utils::LoadInferenceCameraIntrinsics(project_cfg.params_file);
  processor_ = std::make_unique<DynamicCloudProcessor>(
      project_cfg.plan_disp.string(), project_cfg.plan_raft.string(), cam_inference);
  debug_viz_.reset(new FrontendDebugVisualizer());
}

SIOFrontend::~SIOFrontend() {
  stop();
}

void SIOFrontend::LogTargetCacheRebuild(int frame_id, int keyframe_id, size_t raw_points,
                                      size_t depth_valid_points, size_t voxel_cache_points,
                                      float voxel_size_m) const {
  if (!enable_track_console_) {
    return;
  }
  std::cout << "[TargetCache] rebuild frame=" << frame_id << " kf_id=" << keyframe_id
            << " raw=" << raw_points << " depth_valid=" << depth_valid_points
            << " voxel_cache=" << voxel_cache_points << " voxel_m=" << voxel_size_m << "\n";
}
//cache 日志打印
void SIOFrontend::LogKfTrackCache(int frame_id, size_t full_ref, size_t static_ref,
                                  size_t cache_base, size_t cache_esikf,
                                  const ESIKFDebugSummary &esikf_debug) const {
  if (!enable_track_console_) {
    return;
  }
  std::cout << "[TargetCache] kf_track frame=" << frame_id << " full_ref=" << full_ref
            << " static_ref=" << static_ref << " cache_base=" << cache_base
            << " cache_esikf=" << cache_esikf << " esikf_corr=" << esikf_debug.final_corr
            << " src_N=" << esikf_debug.source_N << " tgt_N=" << esikf_debug.target_N
            << " knn=" << esikf_debug.knn_hit << " dist=" << esikf_debug.dist_pass
            << " max_corr=" << esikf_debug.max_corr_m
            << (esikf_debug.skipped ? " SKIPPED" : "")
            << (esikf_debug.converged ? " converged" : " not_converged") << "\n";
}

void SIOFrontend::RebuildTargetCache(const pcl::PointCloud<PointTypeRGB>::Ptr &cloud,
                                     const std::vector<float> &point_confidences,
                                     const std::vector<Eigen::Matrix3d> &point_covariances,
                                     int frame_id, int keyframe_id) {
  static const sio_utils::TrackingTargetParams tracking_cfg =
    sio_utils::LoadTrackingTargetParams(sio_utils::LocateParamsFile());
  const size_t raw_points = cloud ? cloud->size() : 0;
  size_t depth_valid_points = 0;
  // -------------- 构建跟踪目标：体素降采样（与 GICP 共用 tracking_target_voxel_size）
  const auto tracking_target = BuildVoxelTrackingTargetData(
    cloud, point_confidences, point_covariances, tracking_cfg, &depth_valid_points);

  target_cache_.cloud = tracking_target.cloud;
  target_cache_.point_confidences = tracking_target.point_confidences;
  target_cache_.point_covariances = tracking_target.point_covariances;
  target_cache_.source_indices = tracking_target.source_indices;
  target_cache_.kdtree.setInputCloud(target_cache_.cloud);

  if (!target_cache_.cloud || target_cache_.cloud->empty()) {
    target_cache_.normals->clear();
    target_cache_.color_gradient.clear();
    target_cache_.point_covariances.clear();
    target_cache_.source_indices.clear();
    LogTargetCacheRebuild(frame_id, keyframe_id, raw_points, depth_valid_points, 0,
                          tracking_cfg.voxel_size_m);
    return;
  }

  feat_proc_.EstimateNormals(target_cache_.cloud, target_cache_.normals);
  feat_proc_.PrepareColorGradient(target_cache_.normals, target_cache_.color_gradient);
  SaveTargetCacheBase();
  LogTargetCacheRebuild(frame_id, keyframe_id, raw_points, depth_valid_points,
                        target_cache_.cloud->size(), tracking_cfg.voxel_size_m);
}

void SIOFrontend::SaveTargetCacheBase() {
  target_cache_base_.cloud.reset(new pcl::PointCloud<PointTypeRGB>(*target_cache_.cloud));
  target_cache_base_.normals.reset(
    new pcl::PointCloud<pcl::PointXYZRGBNormal>(*target_cache_.normals));
  target_cache_base_.color_gradient = target_cache_.color_gradient;
  target_cache_base_.point_confidences = target_cache_.point_confidences;
  target_cache_base_.point_covariances = target_cache_.point_covariances;
  target_cache_base_.source_indices = target_cache_.source_indices;
  target_cache_base_.kdtree.setInputCloud(target_cache_base_.cloud);
  target_cache_base_valid_ = target_cache_.cloud && !target_cache_.cloud->empty();
}

void SIOFrontend::RestoreTargetCacheFromBase() {
  if (!target_cache_base_valid_ || !target_cache_base_.cloud || target_cache_base_.cloud->empty()) {
    return;
  }
  target_cache_.cloud.reset(new pcl::PointCloud<PointTypeRGB>(*target_cache_base_.cloud));
  target_cache_.normals.reset(
    new pcl::PointCloud<pcl::PointXYZRGBNormal>(*target_cache_base_.normals));
  target_cache_.color_gradient = target_cache_base_.color_gradient;
  target_cache_.point_confidences = target_cache_base_.point_confidences;
  target_cache_.point_covariances = target_cache_base_.point_covariances;
  target_cache_.source_indices = target_cache_base_.source_indices;
  target_cache_.kdtree.setInputCloud(target_cache_.cloud);
}

void SIOFrontend::ResetKeyframeDynamicReference(
  const pcl::PointCloud<PointTypeRGB>::Ptr &body_cloud, const StereoCloud &camera_cloud,
  const std::vector<float> &confidences) {
  if (!body_cloud || body_cloud->empty() || camera_cloud.empty()) {
    prev_keyframe_full_cloud_body_.reset();
    prev_keyframe_full_cloud_camera_.clear();
    prev_keyframe_full_confidences_.clear();
    return;
  }

  prev_keyframe_full_cloud_body_.reset(new pcl::PointCloud<PointTypeRGB>(*body_cloud));
  prev_keyframe_full_cloud_camera_ = camera_cloud;
  prev_keyframe_full_confidences_ = confidences;
}

void SIOFrontend::FilterTargetCacheByStaticIndices(const std::vector<int> &static_indices) {
  if (!target_cache_.cloud || target_cache_.cloud->empty()
      || target_cache_.source_indices.empty()) {
    return;
  }

  std::vector<int> new_index_by_old;
  if (!static_indices.empty()) {
    const int max_index = *std::max_element(static_indices.begin(), static_indices.end());
    if (max_index >= 0) {
      new_index_by_old.assign(static_cast<size_t>(max_index) + 1, -1);
    }
  }
  for (size_t i = 0; i < static_indices.size(); ++i) {
    const int old_index = static_indices[i];
    if (old_index >= 0 && old_index < static_cast<int>(new_index_by_old.size())) {
      new_index_by_old[old_index] = static_cast<int>(i);
    }
  }

  pcl::PointCloud<PointTypeRGB>::Ptr filtered_cloud(new pcl::PointCloud<PointTypeRGB>());
  pcl::PointCloud<PointTypeNormal>::Ptr filtered_normals(new pcl::PointCloud<PointTypeNormal>());
  std::vector<Eigen::Vector3d> filtered_color_gradient;
  std::vector<float> filtered_confidences;
  std::vector<Eigen::Matrix3d> filtered_covariances;
  std::vector<size_t> filtered_source_indices;

  const size_t cache_size =
    std::min({target_cache_.cloud->size(), target_cache_.normals->size(),
              target_cache_.color_gradient.size(), target_cache_.point_confidences.size(),
              target_cache_.point_covariances.size(), target_cache_.source_indices.size()});
  filtered_cloud->points.reserve(cache_size);
  filtered_normals->points.reserve(cache_size);
  filtered_color_gradient.reserve(cache_size);
  filtered_confidences.reserve(cache_size);
  filtered_covariances.reserve(cache_size);
  filtered_source_indices.reserve(cache_size);

  for (size_t i = 0; i < cache_size; ++i) {
    const size_t old_source_index = target_cache_.source_indices[i];
    if (old_source_index >= new_index_by_old.size()) {
      continue;
    }
    const int new_source_index = new_index_by_old[old_source_index];
    if (new_source_index < 0) {
      continue;
    }

    filtered_cloud->points.push_back(target_cache_.cloud->points[i]);
    filtered_normals->points.push_back(target_cache_.normals->points[i]);
    filtered_color_gradient.push_back(target_cache_.color_gradient[i]);
    filtered_confidences.push_back(target_cache_.point_confidences[i]);
    filtered_covariances.push_back(target_cache_.point_covariances[i]);
    filtered_source_indices.push_back(static_cast<size_t>(new_source_index));
  }

  filtered_cloud->width = static_cast<uint32_t>(filtered_cloud->points.size());
  filtered_cloud->height = 1;
  filtered_cloud->is_dense = target_cache_.cloud->is_dense;
  filtered_normals->width = static_cast<uint32_t>(filtered_normals->points.size());
  filtered_normals->height = 1;
  filtered_normals->is_dense = target_cache_.normals->is_dense;

  target_cache_.cloud = filtered_cloud;
  target_cache_.normals = filtered_normals;
  target_cache_.color_gradient.swap(filtered_color_gradient);
  target_cache_.point_confidences.swap(filtered_confidences);
  target_cache_.point_covariances.swap(filtered_covariances);
  target_cache_.source_indices.swap(filtered_source_indices);
  target_cache_.kdtree.setInputCloud(target_cache_.cloud);
}

void SIOFrontend::TransformPointCloudToBody(const StereoCloud &stereo_camera,
                                            const Eigen::Affine3d &T_body_cam) {
  // 相机系 XYZRGB 只拷贝 x,y,z,rgb，w=1，供协方差与 Stereo-GICP。u、v 留在 StereoCloud。
  pointcloud_cur_camera_ = StereoCloudToXyzrgb(stereo_camera);
  pointcloud_cur_.reset(new pcl::PointCloud<PointTypeRGB>(*pointcloud_cur_camera_));
  TransformPointCloudInPlace(T_body_cam, pointcloud_cur_);
  point_covariances_cur_body_
    .clear(); // 清理当前帧点云在机体系下的协方差缓存，后续在需要时重新计算（例如首帧初始化和关键帧创建时）
}

namespace {

constexpr double kBackendMatchTsEps = 1e-3;

void ApplyPoseFromMatrix4d(ImuStatesGroup &state, const Eigen::Matrix4d &T) {
  state.rot_end = T.block<3, 3>(0, 0);
  state.pos_end = T.block<3, 1>(0, 3);
}
// 应用绝对状态 
void ApplyAbsoluteBackendState(ImuStatesGroup &dst, const ImuStatesGroup &src) {
  dst.rot_end = src.rot_end;
  dst.pos_end = src.pos_end;
  dst.vel_end = src.vel_end;
  dst.bias_g = src.bias_g;
  dst.bias_a = src.bias_a;
  dst.cov = src.cov;
  if (src.gravity.norm() > 1e-6) {
    dst.gravity = src.gravity;
  }
}
// 从相对边缘更新当前 IMU 状态
void ComposeFromRelativeEdge(ImuStatesGroup &curr, const ImuStatesGroup &prev,
                             const KeyFrameData &curr_kf, const ImuStatesGroup &bias_src,
                             const Eigen::Matrix4d &delta_T) {
  if (curr_kf.has_relative_edge) {
    const Eigen::Matrix4d T_prev = svio_utils::ImuStatesToMatrix4d(prev);
    ApplyPoseFromMatrix4d(curr, T_prev * curr_kf.relative_T_prev_curr);
    curr.vel_end = prev.vel_end + prev.rot_end * curr_kf.relative_v_prev_curr_body;
  } else {
    ApplyPoseFromMatrix4d(curr, delta_T * svio_utils::ImuStatesToMatrix4d(curr));
    curr.vel_end = delta_T.block<3, 3>(0, 0) * curr.vel_end;
  }
  curr.bias_g = bias_src.bias_g;
  curr.bias_a = bias_src.bias_a;
}
// 匹配后端更新
bool MatchesBackendUpdate(const KeyFrameData &kf, const BackendStateUpdate &upd) {
  if (upd.frontend_keyframe_id >= 0 && kf.frontend_keyframe_id == upd.frontend_keyframe_id) {
    return true;
  }
  return std::abs(kf.timestamp - upd.timestamp) < kBackendMatchTsEps;
}

} // namespace

void SIOFrontend::ApplyBackendStateUpdates() {
  BackendStateUpdate upd; 
  while (backend_.TryPopStateUpdate(upd)) {
    std::shared_ptr<KeyFrameData> matched;
    size_t match_idx = std::numeric_limits<size_t>::max();
    {
      std::lock_guard<std::mutex> lock(mtx_frontend_);
      for (size_t i = 0; i < frontend_keyframe_deque_.size(); ++i) {
        auto &kf = frontend_keyframe_deque_[i];
        if (kf && MatchesBackendUpdate(*kf, upd)) {
          matched = kf;
          match_idx = i;
          break;
        }
      }
    }

    const bool match_prev = MatchesBackendUpdate(prev_keyframe_, upd);
    if (!matched && !match_prev) {
      continue;
    }

    const ImuStatesGroup &old_anchor =
      matched ? matched->imu_states : prev_keyframe_.imu_states;
    const Eigen::Matrix4d T_old = svio_utils::ImuStatesToMatrix4d(old_anchor);
    const Eigen::Matrix4d T_star = svio_utils::ImuStatesToMatrix4d(upd.state);
    const Eigen::Matrix4d delta_T = T_star * T_old.inverse();

    if (matched) {
      ApplyAbsoluteBackendState(matched->imu_states, upd.state);

      {
        std::lock_guard<std::mutex> lock(mtx_frontend_);
        for (size_t i = match_idx + 1; i < frontend_keyframe_deque_.size(); ++i) {
          ComposeFromRelativeEdge(frontend_keyframe_deque_[i]->imu_states,
                                  frontend_keyframe_deque_[i - 1]->imu_states,
                                  *frontend_keyframe_deque_[i], upd.state, delta_T);
        }
      }

      if (match_prev) {
        ApplyAbsoluteBackendState(prev_keyframe_.imu_states, upd.state);
      } else if (prev_keyframe_.frontend_keyframe_id > matched->frontend_keyframe_id) {
        std::lock_guard<std::mutex> lock(mtx_frontend_);
        if (!frontend_keyframe_deque_.empty() &&
            frontend_keyframe_deque_.back()->frontend_keyframe_id ==
              prev_keyframe_.frontend_keyframe_id) {
          prev_keyframe_.imu_states = frontend_keyframe_deque_.back()->imu_states;
        } else if (prev_keyframe_.has_relative_edge && !frontend_keyframe_deque_.empty()) {
          ComposeFromRelativeEdge(prev_keyframe_.imu_states,
                                  frontend_keyframe_deque_.back()->imu_states, prev_keyframe_,
                                  upd.state, delta_T);
        } else {
          ApplyPoseFromMatrix4d(
            prev_keyframe_.imu_states,
            delta_T * svio_utils::ImuStatesToMatrix4d(prev_keyframe_.imu_states));
          prev_keyframe_.imu_states.vel_end =
            delta_T.block<3, 3>(0, 0) * prev_keyframe_.imu_states.vel_end;
          prev_keyframe_.imu_states.bias_g = upd.state.bias_g;
          prev_keyframe_.imu_states.bias_a = upd.state.bias_a;
        }
      }
    } else if (match_prev) {
      ApplyAbsoluteBackendState(prev_keyframe_.imu_states, upd.state);
    }

    // 当前跟踪状态：δT 左乘绝对量，切换偏置供后续预积分（不重积中间帧）
    if (imu_states_.timestamp + kBackendMatchTsEps >= upd.timestamp) {
      ApplyPoseFromMatrix4d(imu_states_,
                            delta_T * svio_utils::ImuStatesToMatrix4d(imu_states_));
      imu_states_.vel_end = delta_T.block<3, 3>(0, 0) * imu_states_.vel_end;
      imu_states_.bias_g = upd.state.bias_g;
      imu_states_.bias_a = upd.state.bias_a;
    }
    if (prev_frame_.timestamp + kBackendMatchTsEps >= upd.timestamp) {
      ApplyPoseFromMatrix4d(prev_frame_.imu_states,
                            delta_T * svio_utils::ImuStatesToMatrix4d(prev_frame_.imu_states));
      prev_frame_.imu_states.vel_end =
        delta_T.block<3, 3>(0, 0) * prev_frame_.imu_states.vel_end;
      prev_frame_.imu_states.bias_g = upd.state.bias_g;
      prev_frame_.imu_states.bias_a = upd.state.bias_a;
    }
  }
}

// Main frontend thread: process frames, manage keyframes, and fuse IMU+vision
// data
void SIOFrontend::run() {
  is_running_.store(true);

  const auto project_cfg = sio_utils::LoadProjectRuntimeConfig();
  // ── 相机内参：推理分辨率（双目/ESIKF）与裁减未缩放分辨率（后端 3DGS）──
  const CameraIntrinsics cam_inference =
    sio_utils::LoadInferenceCameraIntrinsics(project_cfg.params_file);
  const CameraIntrinsics cam_render =
    sio_utils::LoadCroppedCameraIntrinsics(project_cfg.params_file);
  const sio_utils::StereoImageGeometry stereo_geom =
    sio_utils::LoadStereoImageGeometry(project_cfg.params_file);
  processor_->SetCameraIntrinsics(cam_inference); // 双目重建 / 动态滤波
  processor_->ConfigureStereoPreprocess(stereo_geom);
  feat_proc_.SetCameraIntrinsics(cam_inference); // 协方差预计算
  backend_.SetCameraIntrinsics(cam_render);      // 后端 3DGS 渲染（1280×640）
  backend_.SetInferenceCameraIntrinsics(cam_inference);
  backend_.SetStereoImageGeometry(stereo_geom);

  backend_.Start();
  const std::filesystem::path params_cfg_path =
    sio_utils::LocateParamsFile(project_cfg);
  DebugLoggerConfig debug_cfg;
  if (!params_cfg_path.empty()) {
    LoadDebugLoggerConfig(params_cfg_path.string(), debug_cfg);
    imu_preprocessor_.LoadConfig(params_cfg_path.string());
  }
  if (!debug_cfg.fusion_csv_path.empty()) {
    fusion_debug_logger_.Open(debug_cfg.fusion_csv_path);
  }
  if (!debug_cfg.imu_csv_path.empty()) {
    imu_debug_logger_.Open(debug_cfg.imu_csv_path);
  }
  enable_debug_console_ = debug_cfg.console;
  enable_track_console_ = debug_cfg.track_console || debug_cfg.console;
  FrontendDebugVisualConfig viz_cfg;
  LoadFrontendDebugVisualConfig(params_cfg_path.string(), viz_cfg);
  debug_viz_->Configure(viz_cfg);

  const sio_utils::TrackingTargetParams tracking_cfg =
    sio_utils::LoadTrackingTargetParams(params_cfg_path);
  processor_->SetDynamicFilterParams(sio_utils::LoadDynamicFilterParams(params_cfg_path));
  color_icp_.ConfigureGICP(tracking_cfg.voxel_size_m,
                           tracking_cfg.gicp_max_correspondence_distance);

  const Eigen::Affine3d T_body_cam =
    sio_utils::LoadImuBodyFromCameraAffine(params_cfg_path);
  feat_proc_.SetBodyFromCameraExtrinsic(T_body_cam.linear(), T_body_cam.translation());
  backend_.SetBodyFromCameraExtrinsic(T_body_cam.linear(), T_body_cam.translation());

  const std::string img_dir = project_cfg.dataset_dir.string();
  const std::string imu_file = project_cfg.imu_file.string();

  DatasetLoader loader(img_dir, imu_file);
  ImagePreprocessor image_preprocessor;
  if (!params_cfg_path.empty()) {
    image_preprocessor.LoadConfig(params_cfg_path.string());
  }

  std::vector<std::shared_ptr<FrameData>> frame_queue; // 本地测试队列

  //---------------初始化IMU---------------
  while (true) {
    auto frame_data = loader.GrabNextFrame();
    if (!frame_data) {
      break;
    }

    frame_queue.push_back(frame_data);
    // 判断imu初始化是否完成，如果没有完成，就继续读取下一帧，直到 imu
    // 初始化完成后才进入正常的前端流程
    if (!imu_initialized_) {
      // 获取imu数据，用于imu状态的初始化和后续预积分
      for (const auto &imu_msg : frame_data->imu) {
        auto processed = imu_preprocessor_.Process(imu_msg);
        if (!processed) {
          continue;
        }
        imu_buffer_.push_back(std::make_shared<sensor_msgs::msg::Imu>(*processed));
      }
      // 判断imu的数据量够不够初始化，如果不够就继续读取下一帧
      if (imu_buffer_.size() < 200) {
        std::cout << "frame " << frame_data->frame_id << " has insufficient IMU data ("
                  << imu_buffer_.size() << " messages), skipping...\n";
        continue;
      }
      // 检查imu数据的连续性，如果不连续，清理imu缓存队列，重新开始缓存

      if (!isImuDataContinuous()) {
        imu_buffer_.clear();
        std::cout << "frame " << frame_data->frame_id
                  << " has discontinuous IMU data, clearing buffer...\n";
        continue;
      }

      // 用 imu_buffer_ 的静止数据初始化 IMU 状态
      imu_states_ = imu_int_.ImuInit(imu_buffer_);
      // imu初始化完成，清空 imu_buffer_，清空frame_deque_,frame id
      // 重置，之后开启正常的前端流程
      imu_buffer_.clear();
      frame_queue.clear();
      imu_initialized_ = true;
      loader.ResetFrameId(); // 初始化完成，下一帧 frame_id 从 0 开始
      std::cout << "IMU initialized at frame " << frame_data->frame_id
                << " with gyro_bias=" << imu_states_.bias_g.transpose()
                << "  gravity_norm=" << imu_states_.gravity.norm() << "\n";
      continue;
    }

    // 每帧只在读入后做一次双目图像预处理，后续全管线复用处理后的图像。
    // 中心裁减已在 DatasetLoader（文件 / ZED）读入时完成。
    if (frame_data->imgl.empty() || frame_data->imgr.empty()) {
      std::cerr << "[Frontend] Empty stereo image at frame " << frame_data->frame_id
                << ", skip frame." << std::endl;
      continue;
    }
    {
      auto [pre_left, pre_right] =
        image_preprocessor.ProcessStereoAdaptiveLighting(frame_data->imgl, frame_data->imgr);
      if (!pre_left.empty()) {
        frame_data->imgl = pre_left;
      }
      if (!pre_right.empty()) {
        frame_data->imgr = pre_right;
      }
    }

    // 将新的 imu 数据加入缓存：
    // 1) imu_buffer_：跨帧连续传播用
    // 2) keyframe manager 内部缓存：前端关键帧落盘时保留对应 IMU 段
    for (const auto &imu_msg : frame_data->imu) {
      auto processed = imu_preprocessor_.Process(imu_msg);
      if (!processed) {
        continue;
      }
      auto imu_copy = std::make_shared<sensor_msgs::msg::Imu>(*processed);
      imu_buffer_.push_back(imu_copy);
      kf_manager_.AddImuData(imu_copy);
      if (imu_buffer_.size() > 500) {
        imu_buffer_.pop_front();
      }
    }

    //----------------进入frontend线程---------------------

    StereoCloud stereo_cam; // 当前帧相机系视差点，u/v/has_pixel 供下一关键帧几何光流
    cv::Mat h_disp;          // 当前帧噪点滤波后的视差图
    cv::Mat confidence_map;  // 预留给前端双目网络输出的置信度图
    std::vector<float>
      current_point_confidences; // 当前点云在相机系下的置信度，供后续关键帧创建时使用

    // 构建当前帧点云的 lambda
    // 函数，获取相机坐标系下的点云、视差图、置信度图和点云置信度，供首帧初始化和后续关键帧重建使用。
    auto build_current_pointcloud = [this, frame_data, &h_disp, &confidence_map, &stereo_cam,
                                     &current_point_confidences]() {
      h_disp = processor_->computeFilteredDisparity(frame_data->imgl, frame_data->imgr);
      stereo_cam = processor_->generatePointCloud(frame_data->imgl, h_disp, confidence_map,
                                                 &current_point_confidences);
    };

    // --------------首帧初始化 --------------
    if (frame_data->frame_id == 0) {
      build_current_pointcloud(); // 首帧初始化时构建点云，后续帧在需要时才构建，避免冗余计算
      TransformPointCloudToBody(
        stereo_cam,
        T_body_cam); // 将当前帧点云从相机系转换到 IMU（body）系，供后续关键帧创建和跟踪使用
      // 预计算当前帧点云在imu（ body ）系下的协方差，供后续关键帧创建时使用
      feat_proc_.ComputePointCovariancesBody(pointcloud_cur_camera_, current_point_confidences,
                                             T_body_cam.linear(), point_covariances_cur_body_);

      prev_keyframe_ = kf_manager_.CreateFirstKeyFrame(frame_data, pointcloud_cur_, imu_states_,
                                                       h_disp, current_point_confidences,
                                                       point_covariances_cur_body_, confidence_map);
      flow_gate_cov_.setZero();
      {
        std::lock_guard<std::mutex> lock(mtx_frontend_);
        frontend_keyframe_deque_.push_back(std::make_shared<KeyFrameData>(prev_keyframe_));
      }

      // 预计算上一关键帧点云的 KD 树、法向量和颜色梯度，供后续 GICP / ESIKF
      // 使用
      RebuildTargetCache(prev_keyframe_.pointcloud, prev_keyframe_.point_confidences,
                         prev_keyframe_.point_covariances, frame_data->frame_id,
                         prev_keyframe_.keyframe_id);
      color_icp_.BuildGicpKeyframeTargetPrep(target_cache_.cloud, target_cache_.source_indices,
                                             prev_keyframe_.timestamp);
      ResetKeyframeDynamicReference(prev_keyframe_.pointcloud, stereo_cam,
                                    prev_keyframe_.point_confidences);

      imu_states_.timestamp = frame_data->end_time;
      if (debug_viz_->enabled()) {
        FrameDebugVisualInput viz;
        viz.frame_id = frame_data->frame_id;
        viz.is_keyframe = true;
        viz.imu_pred = imu_states_;
        viz.imu_kf = imu_states_;
        viz.stereo_body = prev_keyframe_.pointcloud;
        debug_viz_->ShowFrame(viz);
      }
      continue;
    }

    // ════════════════════════════════════════════════════════════════════════════════
    // ------------IMU 预积分与状态传播：三个重要概念的区分
    // ════════════════════════════════════════════════════════════════════════════════
    //
    // 先消费后端 WindowOptimize 回传：重锚匹配关键帧绝对状态，沿后验相对边复合后续 KF，
    // 并切换 bias 供本帧及之后的预积分（不对中间 ESIKF 帧重积 IMU）。
    ApplyBackendStateUpdates();

    // 【概念 1】IMU 预积分量 (Preintegration Increment):
    //   Δ量：从 propagation_anchor 时刻到当前帧时刻，IMU 测量累积的相对增量
    //         包括 ΔR (相对旋转)、Δv (相对速度增量)、Δp (相对位移增量)
    //   这个量由 ImuPreintegrationCov 内部计算，不直接暴露给外部
    //
    // 【概念 2】IMU 传播的当前先验状态 (Predicted State at Current Frame):
    //   绝对量：当前帧在世界系下的完整状态估计
    //   包括 rot_end、pos_end、vel_end、bias_g、bias_a、gravity、协方差
    //   计算方式：x_k^- = f(propagation_anchor.imu_states, Δ IMU)
    //   存储在：imu_pred_state_curr_ (ImuStatesGroup)
    //   这是"先验"，后续会被 GICP/ESIKF 观测校正
    //   光流门控不用这份绝对协方差，而用 flow_gate_cov_：关键帧处置零后，
    //   与下面同一次预测、以及随后的 ESKF/ESIKF 同步更新。
    //
    // 【概念 3】当前相对上一关键帧的位姿变换 (Relative Transform):
    //   T_kf_curr = T_w_kf^{-1} · T_w_curr（small_gicp T_target_source：p_kf = T_kf_curr · p_curr）
    //   动态滤除用 camera 系 T_curr_prevkf_cam（p_curr = T_curr_prevkf_cam · p_kf_cam）
    //
    // 传播流程：
    //   imu_states_(k-1)^+ + [imu_buffer] ---> imu_pred_state_curr_(k)^-
    //                      (锚点)           (先验状态)
    //
    constexpr double kSameFrameEps = 1e-6; // 设置一个小的时间阈值来判断两帧是否为同一帧
    const bool is_prev_frame_same_as_prev_keyframe =
      std::abs(prev_frame_.timestamp - prev_keyframe_.timestamp) < kSameFrameEps;

    // 若上一帧就是上一关键帧，则直接以关键帧为预积分起点；
    // 否则以上一帧 ESKF 后验状态为起点做连续传播。
    KeyFrameData propagation_anchor = prev_keyframe_;
    // 用于预积分传播的状态锚点，默认为上一关键帧，但如果上一帧不是上一关键帧，则使用上一帧的状态作为锚点，以保证连续传播的正确性。
    if (!is_prev_frame_same_as_prev_keyframe) {
      propagation_anchor.timestamp =
        (imu_states_.timestamp > 0.0) ? imu_states_.timestamp : prev_frame_.timestamp;
      propagation_anchor.imu_states = imu_states_;
    }

    // -------计算当前帧的先验状态（包含 IMU 预积分）--------------------------------
    // 计算锚点到当前帧的预积分增量
    imu_pred_state_curr_ =
      imu_int_.ImuPreintegrationCov(propagation_anchor, imu_buffer_, frame_data->end_time,
                                    &flow_gate_cov_);

    imu_preint_increment_ = imu_int_.GetLastPreintegrationIncrement();

    // 共享世界系先验：anchor(上一帧 ESKF) + 本帧 IMU → T_w_curr；相对 keyframe 导出 body/cam 两路。
    const Eigen::Matrix4d T_w_kf =
      svio_utils::ImuStatesToMatrix4d(prev_keyframe_.imu_states);
    const Eigen::Matrix4d T_w_curr =
      svio_utils::ImuStatesToMatrix4d(imu_pred_state_curr_);

    Eigen::Matrix4d T_imu_cam = Eigen::Matrix4d::Identity();
    T_imu_cam.block<3, 3>(0, 0) = T_body_cam.linear();
    T_imu_cam.block<3, 1>(0, 3) = T_body_cam.translation();

    // body 系 T_kf_curr：p_kf = T_kf_curr * p_curr（GICP / ESKF）
    const Eigen::Matrix4d T_kf_curr_body =
      svio_utils::RelativeTransform(T_w_kf, T_w_curr);

    // camera 系
    const Eigen::Matrix4d T_w_kf_cam = svio_utils::WorldCamFromBody(T_w_kf, T_imu_cam);
    const Eigen::Matrix4d T_w_curr_cam = svio_utils::WorldCamFromBody(T_w_curr, T_imu_cam);
    const Eigen::Matrix4d T_prevkf_curr_cam =
      svio_utils::RelativeTransform(T_w_kf_cam, T_w_curr_cam);
    const Eigen::Matrix4d T_curr_prevkf_cam = T_prevkf_curr_cam.inverse();

    const size_t prev_cloud_full_size =
      prev_keyframe_full_cloud_body_ ? prev_keyframe_full_cloud_body_->size() : 0;

    // 双目重建与动态剔除互不依赖，可以并行执行。
    std::thread frontend_thread(build_current_pointcloud);

    // 每帧：关键帧点云 + 当前帧图像/位姿 → 静态点索引（位姿协方差驱动的马氏动态门控）
    std::vector<int> static_prev_indices;
    std::vector<int> frame_dynamic_indices;
    std::vector<int> frame_invisible_indices;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr frame_invisible_cloud;
    bool dynamic_filter_ok = false;
    const Eigen::Matrix<double, 6, 6> pose_cov_cam =
      svio_utils::FlowGatePoseCovInCamera(imu_pred_state_curr_, flow_gate_cov_, T_imu_cam);
    std::thread dynamic_thread([this, frame_data, &T_curr_prevkf_cam, &static_prev_indices,
                                &frame_dynamic_indices, &frame_invisible_indices,
                                &frame_invisible_cloud, &pose_cov_cam, &dynamic_filter_ok]() {
      if (prev_keyframe_full_cloud_camera_.empty()) {
        return;
      }

      dynamic_filter_ok = processor_->removeDynamicObjects(
        prev_keyframe_.image_left, prev_keyframe_full_cloud_camera_, frame_data->imgl,
        T_curr_prevkf_cam, prev_keyframe_.disp_filtered, &prev_keyframe_full_confidences_,
        nullptr, &static_prev_indices, &frame_dynamic_indices, &pose_cov_cam,
        &frame_invisible_cloud, &frame_invisible_indices);
    });

    frontend_thread.join();

    // 将当前帧点云从相机坐标系转换到机体系，供后续 GICP / ESIKF 使用
    TransformPointCloudToBody(stereo_cam, T_body_cam);

    dynamic_thread.join();

    // 空静态不是失败。成功且静态为空：可见点都是动态，不把它们写回静态。
    // 失败时 removeDynamicObjects 已保留不可见、清空动态，并把未检验可见点放入静态。
    prev_keyframe_invisible_cloud_ = frame_invisible_cloud;

    const auto build_viz_base = [&]() {
      FrameDebugVisualInput viz;
      viz.frame_id = frame_data->frame_id;
      viz.imu_pred = imu_pred_state_curr_;
      viz.imu_kf = prev_keyframe_.imu_states;
      viz.stereo_body = pointcloud_cur_;
      viz.kf_full_body = prev_keyframe_full_cloud_body_;
      viz.static_indices = static_prev_indices;
      viz.dynamic_indices = frame_dynamic_indices;
      viz.invisible_indices = frame_invisible_indices;
      viz.source_body = pointcloud_cur_;
      viz.reg_target =
        BuildCloudByIndices(prev_keyframe_full_cloud_body_, static_prev_indices);
      viz.T_kf_curr_imu_prior = T_kf_curr_body;
      viz.has_imu_prior = pointcloud_cur_ && !pointcloud_cur_->empty();
      return viz;
    };

    // 仍然以"上一前端关键帧"为参考做关键帧判定

    const double time_diff = frame_data->end_time - prev_keyframe_.timestamp;
    // 评估前端关键帧决策
    const FrontKeyframeDecision kf_decision =
      kf_manager_.EvaluateFrontKeyframeDecision(imu_pred_state_curr_, prev_keyframe_.imu_states,
                                                time_diff);
    const double last_imu_ts =
      imu_buffer_.empty() ? frame_data->end_time : svio_utils::MsgTimeSec(imu_buffer_.back());
    const double dt_tail = frame_data->end_time - last_imu_ts;
    int imu_between_count = 0;
    // 计算IMU预积分增量
    for (const auto &imu_msg : imu_buffer_) {
      if (!imu_msg) {
        continue;
      }
      const double imu_t = svio_utils::MsgTimeSec(imu_msg);
      if (imu_t > propagation_anchor.timestamp && imu_t <= frame_data->end_time) {
        ++imu_between_count;
      }
    }
    const auto imu_debug_record = BuildImuKeyframeDebugRecord(
      frame_data->frame_id, frame_data->end_time, frame_data->imu, kf_decision,
      imu_pred_state_curr_, imu_preint_increment_, imu_states_.gravity.norm(),
      imu_preint_increment_.dt, dt_tail, imu_between_count, propagation_anchor.timestamp,
      prev_keyframe_.timestamp, last_imu_ts);
    imu_debug_logger_.WriteRecord(imu_debug_record);
    if (enable_debug_console_) {
      std::cout << ImuDebugLogger::FormatConsoleLine(imu_debug_record) << std::endl;
    }
    if (enable_track_console_) {
      std::cout << ImuDebugLogger::FormatPreintSummaryLine(
                     frame_data->frame_id, imu_preint_increment_,
                     is_prev_frame_same_as_prev_keyframe, imu_between_count, kf_decision)
                << std::endl;
    }

    const auto write_fusion_debug = [&](bool is_keyframe, const ColorICP::GICPResult &gicp_out,
                                        const ImuStatesGroup &state_before_fuse,
                                        const ImuStatesGroup &state_after_fuse,
                                        const ESIKFDebugSummary &esikf_debug = {},
                                        const Eigen::Matrix4d &T_w_gicp_obs =
                                          Eigen::Matrix4d::Identity()) {
      VisualFusionDebugRecord fusion_rec;
      fusion_rec.frame_id = frame_data->frame_id;
      fusion_rec.frame_end_ts = frame_data->end_time;
      fusion_rec.is_keyframe = is_keyframe;
      fusion_rec.source_cloud_size = pointcloud_cur_ ? pointcloud_cur_->size() : 0;
      fusion_rec.target_cloud_size = static_prev_indices.size();
      if (prev_cloud_full_size > 0) {
        const size_t not_static = prev_cloud_full_size - static_prev_indices.size();
        const size_t invisible_count = frame_invisible_indices.size();
        const size_t frame_dynamic_count =
          (not_static > invisible_count) ? (not_static - invisible_count) : 0;
        fusion_rec.dynamic_removed_ratio =
          static_cast<double>(frame_dynamic_count) / static_cast<double>(prev_cloud_full_size);
      }
      FillImuPriorFields(imu_pred_state_curr_, T_prevkf_curr_cam, &fusion_rec);
      const Eigen::Matrix4d T_w_curr_fused = svio_utils::ImuStatesToMatrix4d(state_after_fuse);
      const Eigen::Matrix4d T_prevkf_curr_fused =
        svio_utils::RelativeTransform(T_w_kf, T_w_curr_fused);
      FillImuFusedFields(state_after_fuse, T_prevkf_curr_fused, &fusion_rec);
      fusion_rec.gicp_inliers = gicp_out.num_inliers;
      fusion_rec.gicp_iters = gicp_out.iterations;
      fusion_rec.gicp_converged = gicp_out.converged;
      fusion_rec.gicp_ds_target = gicp_out.downsample_target;
      fusion_rec.gicp_ds_source = gicp_out.downsample_source;
      fusion_rec.gicp_target_prep_cached = gicp_out.target_prep_cached;
      if (!is_keyframe) {
        ComputeRelativeTransformDelta(T_kf_curr_body, gicp_out.T.cast<double>(),
                                      &fusion_rec.gicp_corr_trans_m, &fusion_rec.gicp_corr_rot_deg);
        fusion_rec.gicp_cov_trace = gicp_out.covariance.trace();
        ImuStatesGroup gicp_obs_state = imu_pred_state_curr_;
        gicp_obs_state.rot_end = T_w_gicp_obs.block<3, 3>(0, 0);
        gicp_obs_state.pos_end = T_w_gicp_obs.block<3, 1>(0, 3);
        ComputeImuStatePoseDelta(imu_pred_state_curr_, gicp_obs_state,
                                 &fusion_rec.eskf_innov_trans_m, &fusion_rec.eskf_innov_rot_deg);
        ComputeImuStatePoseDelta(imu_pred_state_curr_, state_after_fuse,
                                 &fusion_rec.eskf_fused_trans_m, &fusion_rec.eskf_fused_rot_deg);
      }
      fusion_rec.esikf_skipped = esikf_debug.skipped;
      fusion_rec.esikf_converged = esikf_debug.converged;
      fusion_rec.esikf_iters = esikf_debug.num_iters;
      fusion_rec.esikf_final_corr = static_cast<double>(esikf_debug.final_corr);
      if (is_keyframe && !esikf_debug.skipped) {
        fusion_rec.esikf_pose_delta_m = esikf_debug.pose_delta_m;
        fusion_rec.esikf_pose_delta_deg = esikf_debug.pose_delta_deg;
      }
      fusion_rec.target_cache_size = target_cache_.cloud ? target_cache_.cloud->size() : 0;
      fusion_debug_logger_.WriteRecord(fusion_rec);
      if (enable_debug_console_) {
        std::cout << FusionDebugLogger::FormatConsoleLine(fusion_rec) << std::endl;
      }
    };

    if (!kf_decision.is_keyframe) {
      if (target_cache_future_.valid()) {
        target_cache_future_.wait();
      }

      if (static_prev_indices.empty()) {
        // 没有静态 target。成功：可见点均已标动态，动态索引保持不动。
        // 失败：没有可暂作静态的可见点。都不做本帧静态云配准，状态沿用 IMU 先验。
        if (enable_track_console_) {
          std::cout << "[dynamic_filter] frame=" << frame_data->frame_id
                    << " ok=" << (dynamic_filter_ok ? 1 : 0)
                    << " static=0 dynamic=" << frame_dynamic_indices.size()
                    << " invisible=" << frame_invisible_indices.size()
                    << " skip registration\n";
        }
        ColorICP::GICPResult skipped_gicp;
        skipped_gicp.T = T_kf_curr_body.cast<float>();
        write_fusion_debug(false, skipped_gicp, imu_pred_state_curr_, imu_pred_state_curr_);
        imu_states_ = imu_pred_state_curr_;
        imu_states_.timestamp = frame_data->end_time;
        prev_frame_.timestamp = frame_data->end_time;
        prev_frame_.image_left = frame_data->imgl;
        prev_frame_.image_right = frame_data->imgr;
        prev_frame_.pointcloud = pointcloud_cur_;
        prev_frame_.imu_states = imu_states_;
        prev_frame_.point_confidences = current_point_confidences;
        if (debug_viz_->enabled()) {
          FrameDebugVisualInput viz = build_viz_base();
          viz.is_keyframe = false;
          debug_viz_->ShowFrame(viz);
        }
        continue;
      }

      // --------------非关键帧：Stereo-GICP 位姿观测 + IMU 先验做 ESKF 松耦合
      // 当前帧完整相机点云 ↔ 上一关键帧静态点云（投影关联 + 立体 Σ）。
      const auto filtered_target_cloud =
        BuildCloudByIndices(prev_keyframe_full_cloud_body_, static_prev_indices);
      const auto filtered_target_cloud_cam =
        StereoCloudToXyzrgbByIndices(prev_keyframe_full_cloud_camera_, static_prev_indices);

      CameraIntrinsics model_cam = cam_inference;
      if (!prev_keyframe_.image_left.empty()) {
        model_cam.image_width = prev_keyframe_.image_left.cols;
        model_cam.image_height = prev_keyframe_.image_left.rows;
      } else if (!prev_keyframe_.disp_filtered.empty()) {
        model_cam.image_width = prev_keyframe_.disp_filtered.cols;
        model_cam.image_height = prev_keyframe_.disp_filtered.rows;
      }

      auto gicp_out = color_icp_.EstimatePoseStereoGICP(
        pointcloud_cur_camera_, T_prevkf_curr_cam, model_cam, cv::Mat(), T_imu_cam,
        filtered_target_cloud_cam);
      const bool used_stereo_gicp = gicp_out.num_inliers >= 20;
      if (!used_stereo_gicp) {
        gicp_out = color_icp_.EstimatePoseGICPWithCacheFilter(
          pointcloud_cur_, T_kf_curr_body, prev_keyframe_.timestamp, static_prev_indices,
          filtered_target_cloud);
      }

      if (enable_track_console_) {
        double gicp_corr_trans_m = 0.0;
        double gicp_corr_rot_deg = 0.0;
        const Eigen::Matrix4d T_gicp_kf_curr = gicp_out.T.cast<double>();
        ComputeRelativeTransformDelta(T_kf_curr_body, T_gicp_kf_curr, &gicp_corr_trans_m,
                                      &gicp_corr_rot_deg);
        std::cout << "[GICP] frame=" << frame_data->frame_id
                  << " method=" << (used_stereo_gicp ? "stereo" : "small_gicp")
                  << " static_tgt=" << static_prev_indices.size()
                  << " ds_tgt=" << gicp_out.downsample_target
                  << " ds_src=" << gicp_out.downsample_source
                  << " inliers=" << gicp_out.num_inliers
                  << " max_corr=" << tracking_cfg.gicp_max_correspondence_distance
                  << " converged=" << (gicp_out.converged ? 1 : 0) << " corr_delta="
                  << gicp_corr_trans_m << "m/" << gicp_corr_rot_deg << "deg\n";
        LogPoseFusionDebug(frame_data->frame_id, T_kf_curr_body, T_gicp_kf_curr,
                           imu_pred_state_curr_, gicp_out.covariance);
      }

      // T 已映回 body：prevkf(body) ← curr(body)，乘 T_w_kf 得世界系观测
      const Eigen::Matrix4d T_w_curr_gicp = T_w_kf * gicp_out.T.cast<double>();

      Eigen::Matrix<double, 6, 6> R_obs_aligned;
      if (used_stereo_gicp) {
        // Stereo-GICP 左扰动协方差已映到 body；按 robotic-front 对齐到 ESKF 混合坐标。
        R_obs_aligned = svio_utils::AlignStereoGicpCovarianceToEskfPose(
          gicp_out.covariance, prev_keyframe_.imu_states.rot_end,
          gicp_out.T.cast<double>());
      } else {
        // small_gicp Σ: T_target_source 的 se(3) 右扰动 [δθ_curr, δt_curr]（body）
        R_obs_aligned = svio_utils::AlignGicpCovarianceToEskfPose(
          gicp_out.covariance, prev_keyframe_.imu_states.rot_end, gicp_out.T.cast<double>());
      }


      // ESKF 松耦合：在 IMU 先验基础上用 GICP 观测做一次滤波更新，得到当前帧的后验状态

      const ImuStatesGroup imu_before_eskf = imu_pred_state_curr_;
      imu_states_ =
        estimator_.ESKF(imu_pred_state_curr_, T_w_curr_gicp.cast<float>(), R_obs_aligned,
                        &flow_gate_cov_);
      write_fusion_debug(false, gicp_out, imu_before_eskf, imu_states_, {}, T_w_curr_gicp);

      if (debug_viz_->enabled()) {
        FrameDebugVisualInput viz = build_viz_base();
        viz.is_keyframe = false;
        viz.has_gicp = true;
        viz.T_kf_curr_gicp = gicp_out.T.cast<double>();
        viz.has_eskf = true;
        viz.T_kf_curr_eskf =
          svio_utils::RelativeTransform(T_w_kf, svio_utils::ImuStatesToMatrix4d(imu_states_));
        debug_viz_->ShowFrame(viz);
      }

      // 更新上一帧状态为当前帧的后验，供下一帧继续传播
      imu_states_.timestamp = frame_data->end_time;
      prev_frame_.timestamp = frame_data->end_time;
      prev_frame_.image_left = frame_data->imgl;
      prev_frame_.image_right = frame_data->imgr;
      prev_frame_.pointcloud = pointcloud_cur_;
      prev_frame_.imu_states = imu_states_;
      prev_frame_.point_confidences = current_point_confidences;
      continue;
    }

    // 前端关键帧：在 IMU 先验基础上做几何+颜色联合更新，生成新关键帧，并判断是否升级为后端关键帧

    else {
      //----前端关键帧
      if (target_cache_future_.valid()) {
        target_cache_future_.wait();
      }

      // 静态非空才从完整缓存滤出本帧静态点。静态为空时保留上一份缓存：
      // 成功表示可见点均为动态；失败表示没有未检验可见点。都不把动态索引并入静态。
      if (!static_prev_indices.empty()) {
        RestoreTargetCacheFromBase();
        FilterTargetCacheByStaticIndices(static_prev_indices);
      }
      // 静态为空，或过滤后没有剩余点云/置信度：跳过本帧前端关键帧，沿用 IMU 先验。
      ESIKFDebugSummary esikf_debug;
      if (static_prev_indices.empty() || target_cache_.normals->empty()
          || target_cache_.point_confidences.empty()) {
        esikf_debug.skipped = true;
        if (static_prev_indices.empty() && enable_track_console_) {
          std::cout << "[dynamic_filter] frame=" << frame_data->frame_id
                    << " ok=" << (dynamic_filter_ok ? 1 : 0)
                    << " static=0 dynamic=" << frame_dynamic_indices.size()
                    << " invisible=" << frame_invisible_indices.size()
                    << " keep previous target cache\n";
        }
        write_fusion_debug(true, ColorICP::GICPResult{}, imu_pred_state_curr_,
                           imu_pred_state_curr_, esikf_debug);
        const size_t cache_base_size =
          (target_cache_base_valid_ && target_cache_base_.cloud) ? target_cache_base_.cloud->size()
                                                                 : 0;
        LogKfTrackCache(frame_data->frame_id, prev_cloud_full_size, static_prev_indices.size(),
                        cache_base_size, 0, esikf_debug);
        imu_states_ = imu_pred_state_curr_;
        imu_states_.timestamp = frame_data->end_time;
        prev_frame_.timestamp = frame_data->end_time;
        prev_frame_.image_left = frame_data->imgl;
        prev_frame_.image_right = frame_data->imgr;
        prev_frame_.pointcloud = pointcloud_cur_;
        prev_frame_.imu_states = imu_states_;
        prev_frame_.point_confidences = current_point_confidences;
        if (debug_viz_->enabled()) {
          FrameDebugVisualInput viz = build_viz_base();
          viz.is_keyframe = true;
          debug_viz_->ShowFrame(viz);
        }
        continue;
      }

      // 前端关键帧：从 target_cache_.normals 提取法向量数组，供 ESIKF 使用
      std::vector<Eigen::Vector3d> target_normals;
      target_normals.reserve(target_cache_.normals->size());
      for (const auto &pt_n : target_cache_.normals->points) {
        target_normals.emplace_back(pt_n.normal_x, pt_n.normal_y, pt_n.normal_z);
      }

      // 计算当前帧点云在机体系下的协方差
      feat_proc_.ComputePointCovariancesBody(pointcloud_cur_camera_, current_point_confidences,
                                             T_body_cam.linear(), point_covariances_cur_body_);

      // 关键帧视差 → 深度图，供 ESIKF Stereo-GICP 投影关联
      const cv::Mat target_depth_cam = sio_utils::DisparityToDepthMeters(
        prev_keyframe_.disp_filtered, cam_inference, tracking_cfg.min_depth_m,
        tracking_cfg.max_depth_m);

      // ESIKF 紧耦合：投影关联 + Stereo-GICP 马氏几何项，锚点为上一关键帧
      const ImuStatesGroup imu_before_esikf = imu_pred_state_curr_;
      estimator_.ESIKF(pointcloud_cur_,
                       std::make_shared<pcl::KdTreeFLANN<PointTypeRGB>>(target_cache_.kdtree),
                       target_cache_.color_gradient, target_normals, imu_pred_state_curr_,
                       prev_keyframe_.imu_states.rot_end, prev_keyframe_.imu_states.pos_end,
                       current_point_confidences, target_cache_.point_confidences, 0.5,
                       &flow_gate_cov_, T_imu_cam, &esikf_debug, tracking_cfg.voxel_size_m,
                       point_covariances_cur_body_, target_cache_.point_covariances,
                       tracking_cfg.esikf_max_correspondence_distance, target_depth_cam,
                       cam_inference);

      imu_states_ = imu_pred_state_curr_;
      write_fusion_debug(true, ColorICP::GICPResult{}, imu_before_esikf, imu_states_, esikf_debug);

      if (debug_viz_->enabled()) {
        FrameDebugVisualInput viz = build_viz_base();
        viz.is_keyframe = true;
        viz.has_esikf = !esikf_debug.skipped;
        viz.esikf_target = target_cache_.cloud;
        viz.T_kf_curr_esikf =
          svio_utils::RelativeTransform(T_w_kf, svio_utils::ImuStatesToMatrix4d(imu_states_));
        debug_viz_->ShowFrame(viz);
      }

      const size_t cache_base_size =
        (target_cache_base_valid_ && target_cache_base_.cloud) ? target_cache_base_.cloud->size()
                                                               : 0;
      const size_t cache_esikf_size = target_cache_.cloud ? target_cache_.cloud->size() : 0;
      LogKfTrackCache(frame_data->frame_id, prev_cloud_full_size, static_prev_indices.size(),
                      cache_base_size, cache_esikf_size, esikf_debug);

      // 更新上一帧状态为当前帧的后验，供下一帧继续传播
      imu_states_.timestamp = frame_data->end_time;
      prev_frame_.timestamp = frame_data->end_time;
      prev_frame_.image_left = frame_data->imgl;
      prev_frame_.image_right = frame_data->imgr;
      prev_frame_.pointcloud = pointcloud_cur_;
      prev_frame_.imu_states = imu_states_;
      prev_frame_.point_confidences = current_point_confidences;

      // 生成前端关键帧：相对边锚在上一前端关键帧 prev_keyframe_（非已更新的 prev_frame_）
      const auto keyframe_imu_buffer = kf_manager_.GetImuBufferForKeyFrame();
      std::shared_ptr<KeyFrameData> new_kf = kf_manager_.CreateFrontendKeyFrame(
        prev_keyframe_, frame_data, keyframe_imu_buffer, pointcloud_cur_, imu_states_, h_disp,
        current_point_confidences, point_covariances_cur_body_, confidence_map);
      {
        std::lock_guard<std::mutex> lock(mtx_frontend_);
        frontend_keyframe_deque_.push_back(new_kf);
      }

      // 判断上一前端关键帧是否需要升级为后端关键帧（空间+旋转+时间复合策略）
      const double backend_time_diff =
        (prev_backend_keyframe_.timestamp > 0.0)
          ? (prev_keyframe_.timestamp - prev_backend_keyframe_.timestamp)
          : std::numeric_limits<double>::infinity();

      if (kf_manager_.IsBackKeyFrame(prev_keyframe_, frontend_keyframe_deque_.size(),
                                     backend_time_diff)) {
        auto frontend_kf_for_backend = std::make_shared<KeyFrameData>(prev_keyframe_);
        std::shared_ptr<KeyFrameData> backend_kf = kf_manager_.CreateBackendKeyFrame(
          prev_backend_keyframe_, frontend_kf_for_backend, imu_buffer_);
        prev_backend_keyframe_ = *backend_kf;
        backend_.Wake(backend_kf);
      }

      // 更新上一前端关键帧
      // 异步构建用于 ESIKF / GICP 的缓存，避免阻塞前端线程
      prev_keyframe_ = *new_kf;
      flow_gate_cov_.setZero();
      ResetKeyframeDynamicReference(new_kf->pointcloud, stereo_cam, new_kf->point_confidences);
      color_icp_.InvalidateGicpTargetPrepCache();
      target_cache_future_ = std::async(std::launch::async, [this, new_kf, log_frame_id = frame_data->frame_id]() {
        RebuildTargetCache(new_kf->pointcloud, new_kf->point_confidences, new_kf->point_covariances,
                           log_frame_id, new_kf->keyframe_id);
        color_icp_.BuildGicpKeyframeTargetPrep(target_cache_.cloud, target_cache_.source_indices,
                                               new_kf->timestamp);
      });

      // 前端关键帧队列加长，覆盖后端异步优化延迟，便于 ID/时间戳匹配回传
      {
        std::lock_guard<std::mutex> lock(mtx_frontend_);
        constexpr size_t kFrontendKfHistory = 30;
        while (frontend_keyframe_deque_.size() > kFrontendKfHistory) {
          frontend_keyframe_deque_.pop_front();
        }
      }
    }
  }

  is_running_.store(false);
  backend_.Stop();
}

// Check if IMU buffer has continuous data without large gaps
bool SIOFrontend::isImuDataContinuous() const {
  // 期望 IMU 频率约 200 Hz，周期约 5 ms
  // 允许最大相邻时间戳间隔为 2 倍周期（10 ms）
  constexpr double kMaxGapSec = 0.01;

  for (size_t i = 1; i < imu_buffer_.size(); ++i) {
    const auto &prev = imu_buffer_[i - 1];
    const auto &curr = imu_buffer_[i];
    double t_prev = prev->header.stamp.sec + prev->header.stamp.nanosec * 1e-9;
    double t_curr = curr->header.stamp.sec + curr->header.stamp.nanosec * 1e-9;
    if (t_curr - t_prev > kMaxGapSec) {
      std::cerr << "[IMU] discontinuity detected: gap = " << (t_curr - t_prev) * 1000.0
                << " ms at index " << i << "\n";
      return false;
    }
  }
  return true;
}
