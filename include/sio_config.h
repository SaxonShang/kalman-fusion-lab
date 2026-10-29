#pragma once

#include "camera_intrinsics.h"
#include "sio_stereo_geometry.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>

namespace sio_utils {

struct DynamicFilterParams {
  float tau_px_fixed = 12.0f;
  float raft_noise_base_px = 2.0f;
  float raft_noise_near_px = 6.0f;
  float geo_flow_min_sq = 0.25f;
  // 与点云重建共用深度范围：z in [min_depth_m, max_depth_m]
  float min_depth_m = 0.1f;
  float max_depth_m = 30.0f;
};

struct DisparityDepthBounds {
  float min_disparity = 0.0f; // fb / max_depth，小于此值视为超过 max_depth
  float max_disparity = 0.0f; // fb / min_depth，大于此值视为小于 min_depth
};

inline DisparityDepthBounds ComputeDisparityDepthBounds(float fx, float baseline, float min_depth_m,
                                                      float max_depth_m) {
  const float fb = std::max(1e-6f, fx * baseline);
  const float z_min = std::max(min_depth_m, 1e-3f);
  const float z_max = std::max(max_depth_m, z_min + 1e-3f);
  DisparityDepthBounds bounds;
  bounds.min_disparity = fb / z_max;
  bounds.max_disparity = fb / z_min;
  return bounds;
}

inline bool IsDisparityInDepthRange(float disparity, const DisparityDepthBounds &bounds) {
  return std::isfinite(disparity) && disparity >= bounds.min_disparity
         && disparity <= bounds.max_disparity;
}

inline float ComputeFlowThresholdPx(float depth_m, const DynamicFilterParams &cfg, float /*fx*/) {
  const float z = std::max(depth_m, 1e-3f);
  const float tau_noise = cfg.raft_noise_base_px + cfg.raft_noise_near_px / z;
  return std::max(cfg.tau_px_fixed, tau_noise);
}

struct TrackingTargetParams {
  size_t max_points = 50000;
  float min_depth_m = 0.1f;
  float max_depth_m = 30.0f;
  float min_confidence = 0.0f;
  float voxel_size_m = 0.05f;                    // ESIKF target_cache 与 GICP 共用体素
  float gicp_max_correspondence_distance = 0.10f; // GICP 最大对应距离 (m)
  float esikf_max_correspondence_distance = 0.15f; // ESIKF 最大对应距离 (m)
};

struct ProjectRuntimeConfig {
  std::filesystem::path config_path;
  std::filesystem::path project_root;
  std::filesystem::path dataset_dir;
  std::filesystem::path imu_file;
  std::filesystem::path plan_disp;
  std::filesystem::path plan_raft;
  std::filesystem::path params_file;
  std::string topic = "/sio/pipeline_debug_cloud";
  std::string frame_id = "map";
};

std::filesystem::path LocateProjectConfigPath();
ProjectRuntimeConfig LoadProjectRuntimeConfig();
std::filesystem::path LocateParamsFile();
std::filesystem::path LocateParamsFile(const ProjectRuntimeConfig &project_cfg);

CameraIntrinsics LoadCameraIntrinsics(const std::filesystem::path &params_file);
StereoImageGeometry LoadStereoImageGeometry(const std::filesystem::path &params_file);
CameraIntrinsics LoadCroppedCameraIntrinsics(const std::filesystem::path &params_file);
CameraIntrinsics LoadInferenceCameraIntrinsics(const std::filesystem::path &params_file);
TrackingTargetParams LoadTrackingTargetParams(const std::filesystem::path &params_file);
DynamicFilterParams LoadDynamicFilterParams(const std::filesystem::path &params_file);
Eigen::Affine3d LoadImuBodyFromCameraAffine(const std::filesystem::path &params_file);
Eigen::Matrix4d LoadImuCameraExtrinsic(const std::filesystem::path &params_file);

// 在 YAML 树中深度查找首个匹配的标量 key（兼容嵌套节）。
bool ParseYamlScalarDouble(const std::filesystem::path &file_path, const std::string &key,
                           double &out_value);

} // namespace sio_utils
