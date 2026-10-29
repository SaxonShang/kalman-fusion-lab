#include "sio_config.h"
#include "sio_stereo_geometry.h"

#include <yaml-cpp/yaml.h>

#include <iostream>
#include <stdexcept>
#include <vector>

namespace sio_utils {
namespace {

std::filesystem::path ResolveRelativePath(const std::filesystem::path &value,
                                          const std::filesystem::path &base_dir) {
  if (value.empty()) {
    return {};
  }
  if (value.is_absolute()) {
    return value.lexically_normal();
  }
  return (base_dir / value).lexically_normal();
}

Eigen::Vector3d ReadVector3(const YAML::Node &node, const Eigen::Vector3d &default_value) {
  if (!node || !node.IsSequence() || node.size() < 3) {
    return default_value;
  }
  return Eigen::Vector3d(node[0].as<double>(), node[1].as<double>(), node[2].as<double>());
}

Eigen::Matrix3d RpyToRotation(const Eigen::Vector3d &rpy) {
  return (Eigen::AngleAxisd(rpy.z(), Eigen::Vector3d::UnitZ()) *
          Eigen::AngleAxisd(rpy.y(), Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(rpy.x(), Eigen::Vector3d::UnitX()))
    .toRotationMatrix();
}

YAML::Node LoadYamlFileOrEmpty(const std::filesystem::path &file_path) {
  if (file_path.empty() || !std::filesystem::exists(file_path)) {
    return {};
  }
  try {
    return YAML::LoadFile(file_path.string());
  } catch (const std::exception &e) {
    std::cerr << "[sio_config] Failed to load YAML: " << file_path << " (" << e.what() << ")\n";
    return {};
  }
}

bool FindScalarKeyDeep(const YAML::Node &node, const std::string &key, double &out_value) {
  if (!node || !node.IsMap()) {
    return false;
  }

  if (node[key]) {
    out_value = node[key].as<double>();
    return true;
  }

  for (const auto &entry : node) {
    if (FindScalarKeyDeep(entry.second, key, out_value)) {
      return true;
    }
  }
  return false;
}

void MergeProjectRuntimeConfig(ProjectRuntimeConfig &cfg, const YAML::Node &root,
                               const std::filesystem::path &base_dir) {
  if (!root || !root.IsMap()) {
    return;
  }

  if (root["dataset_dir"]) {
    cfg.dataset_dir = ResolveRelativePath(root["dataset_dir"].as<std::string>(), base_dir);
  }
  if (root["imu_file"]) {
    cfg.imu_file = ResolveRelativePath(root["imu_file"].as<std::string>(), base_dir);
  }
  if (root["plan_disp"]) {
    cfg.plan_disp = ResolveRelativePath(root["plan_disp"].as<std::string>(), base_dir);
  }
  if (root["plan_raft"]) {
    cfg.plan_raft = ResolveRelativePath(root["plan_raft"].as<std::string>(), base_dir);
  }
  if (root["params_file"]) {
    cfg.params_file =
      ResolveRelativePath(root["params_file"].as<std::string>(), base_dir);
  } else if (cfg.params_file.empty()) {
    cfg.params_file = ResolveRelativePath("config/params.yaml", base_dir);
  }
  if (root["topic"]) {
    cfg.topic = root["topic"].as<std::string>();
  }
  if (root["frame_id"]) {
    cfg.frame_id = root["frame_id"].as<std::string>();
  }
}

std::filesystem::path LocateParamsFileFallback() {
  const std::vector<std::filesystem::path> candidates = {
    std::filesystem::current_path() / "config" / "params.yaml",
    std::filesystem::current_path() / "params.yaml",
    std::filesystem::current_path().parent_path() / "config" / "params.yaml",
  };

  for (const auto &candidate : candidates) {
    if (std::filesystem::exists(candidate)) {
      return candidate.lexically_normal();
    }
  }
  return {};
}

} // namespace

std::filesystem::path LocateProjectConfigPath() {
  const std::filesystem::path source_hint =
#ifdef SIO_SLAM_PROJECT_CONFIG_PATH
    std::filesystem::path(SIO_SLAM_PROJECT_CONFIG_PATH);
#else
    std::filesystem::path();
#endif

  const std::vector<std::filesystem::path> candidates = {
    source_hint,
    std::filesystem::current_path() / "project_config.yaml",
    std::filesystem::current_path() / "config" / "project_config.yaml",
    std::filesystem::current_path().parent_path() / "project_config.yaml",
    std::filesystem::current_path().parent_path() / "config" / "project_config.yaml",
  };

  for (const auto &candidate : candidates) {
    if (!candidate.empty() && std::filesystem::exists(candidate)) {
      return candidate.lexically_normal();
    }
  }
  return {};
}

ProjectRuntimeConfig LoadProjectRuntimeConfig() {
  ProjectRuntimeConfig cfg;
  cfg.config_path = LocateProjectConfigPath();
  if (cfg.config_path.empty()) {
    throw std::runtime_error(
      "Project config not found. Please add project_config.yaml in the project root.");
  }

  cfg.project_root = cfg.config_path.parent_path().lexically_normal();

  const YAML::Node root = LoadYamlFileOrEmpty(cfg.config_path);
  MergeProjectRuntimeConfig(cfg, root, cfg.project_root);

  const std::filesystem::path local_override_path =
    cfg.project_root / "project_config.local.yaml";
  if (std::filesystem::exists(local_override_path)) {
    const YAML::Node local_root = LoadYamlFileOrEmpty(local_override_path);
    MergeProjectRuntimeConfig(
      cfg, local_root, local_override_path.parent_path().lexically_normal());
  }

  return cfg;
}

std::filesystem::path LocateParamsFile(const ProjectRuntimeConfig &project_cfg) {
  if (!project_cfg.params_file.empty()
      && std::filesystem::exists(project_cfg.params_file)) {
    return project_cfg.params_file;
  }
  return LocateParamsFileFallback();
}

std::filesystem::path LocateParamsFile() {
  try {
    return LocateParamsFile(LoadProjectRuntimeConfig());
  } catch (...) {
    return LocateParamsFileFallback();
  }
}

CameraIntrinsics LoadCameraIntrinsics(const std::filesystem::path &params_file) {
  CameraIntrinsics cam{500.0f, 500.0f, 320.0f, 240.0f, 0.12f};
  const YAML::Node root = LoadYamlFileOrEmpty(params_file);
  const YAML::Node node = root["camera intrinsic"];
  if (!node) {
    return cam;
  }

  if (node["fx"]) {
    cam.fx = static_cast<float>(node["fx"].as<double>());
  }
  if (node["fy"]) {
    cam.fy = static_cast<float>(node["fy"].as<double>());
  }
  if (node["cx"]) {
    cam.cx = static_cast<float>(node["cx"].as<double>());
  }
  if (node["cy"]) {
    cam.cy = static_cast<float>(node["cy"].as<double>());
  }
  if (node["baseline"]) {
    cam.baseline = static_cast<float>(node["baseline"].as<double>());
  }
  return cam;
}

StereoImageGeometry LoadStereoImageGeometry(const std::filesystem::path &params_file) {
  StereoImageGeometry geom;
  const YAML::Node root = LoadYamlFileOrEmpty(params_file);
  const YAML::Node node = root["camera intrinsic"];
  if (!node) {
    return geom;
  }

  if (node["source_width"]) {
    geom.source_width = node["source_width"].as<int>();
  }
  if (node["source_height"]) {
    geom.source_height = node["source_height"].as<int>();
  }
  if (node["inference_width"]) {
    geom.inference_width = node["inference_width"].as<int>();
  }
  if (node["inference_height"]) {
    geom.inference_height = node["inference_height"].as<int>();
  }
  return geom;
}

CameraIntrinsics LoadCroppedCameraIntrinsics(const std::filesystem::path &params_file) {
  const CameraIntrinsics source_cam = LoadCameraIntrinsics(params_file);
  const StereoImageGeometry geom = LoadStereoImageGeometry(params_file);
  return AdjustIntrinsicsForCrop(source_cam, geom);
}

CameraIntrinsics LoadInferenceCameraIntrinsics(const std::filesystem::path &params_file) {
  const CameraIntrinsics source_cam = LoadCameraIntrinsics(params_file);
  const StereoImageGeometry geom = LoadStereoImageGeometry(params_file);
  return AdjustIntrinsicsForInference(source_cam, geom);
}

TrackingTargetParams LoadTrackingTargetParams(const std::filesystem::path &params_file) {
  TrackingTargetParams cfg;
  const YAML::Node root = LoadYamlFileOrEmpty(params_file);
  const YAML::Node node = root["slam parameters"];
  if (!node) {
    return cfg;
  }

  if (node["tracking_target_max_points"]) {
    const double raw = node["tracking_target_max_points"].as<double>();
    if (raw > 0.0) {
      cfg.max_points = static_cast<size_t>(raw);
    }
  }
  if (node["tracking_target_min_depth_m"]) {
    const double raw = node["tracking_target_min_depth_m"].as<double>();
    if (raw >= 0.0) {
      cfg.min_depth_m = static_cast<float>(raw);
    }
  }
  if (node["tracking_target_max_depth_m"]) {
    const double raw = node["tracking_target_max_depth_m"].as<double>();
    if (raw > 0.0) {
      cfg.max_depth_m = static_cast<float>(raw);
    }
  }
  if (node["tracking_target_min_confidence"]) {
    const double raw = node["tracking_target_min_confidence"].as<double>();
    if (raw >= 0.0) {
      cfg.min_confidence = static_cast<float>(raw);
    }
  }
  if (node["tracking_target_voxel_size"]) {
    const double raw = node["tracking_target_voxel_size"].as<double>();
    if (raw > 0.0) {
      cfg.voxel_size_m = static_cast<float>(raw);
    }
  }
  if (node["tracking_target_max_correspondence_distance"]) {
    const double raw = node["tracking_target_max_correspondence_distance"].as<double>();
    if (raw > 0.0) {
      cfg.gicp_max_correspondence_distance = static_cast<float>(raw);
    }
  }
  if (node["esikf_max_correspondence_distance"]) {
    const double raw = node["esikf_max_correspondence_distance"].as<double>();
    if (raw > 0.0) {
      cfg.esikf_max_correspondence_distance = static_cast<float>(raw);
    }
  }

  if (cfg.max_depth_m < cfg.min_depth_m) {
    std::swap(cfg.max_depth_m, cfg.min_depth_m);
  }
  cfg.voxel_size_m = std::max(0.01f, cfg.voxel_size_m);
  cfg.gicp_max_correspondence_distance =
    std::max(cfg.voxel_size_m, cfg.gicp_max_correspondence_distance);
  cfg.esikf_max_correspondence_distance =
    std::max(cfg.voxel_size_m, cfg.esikf_max_correspondence_distance);
  return cfg;
}

DynamicFilterParams LoadDynamicFilterParams(const std::filesystem::path &params_file) {
  DynamicFilterParams cfg;
  const YAML::Node root = LoadYamlFileOrEmpty(params_file);

  const YAML::Node slam = root["slam parameters"];
  if (slam) {
    if (slam["tracking_target_min_depth_m"]) {
      cfg.min_depth_m = static_cast<float>(slam["tracking_target_min_depth_m"].as<double>());
    }
    if (slam["tracking_target_max_depth_m"]) {
      cfg.max_depth_m = static_cast<float>(slam["tracking_target_max_depth_m"].as<double>());
    }
  }

  const YAML::Node node = root["dynamic_filter"];
  if (!node) {
    if (cfg.max_depth_m < cfg.min_depth_m) {
      std::swap(cfg.max_depth_m, cfg.min_depth_m);
    }
    return cfg;
  }

  auto read_f = [&](const char *key, float &out) {
    if (node[key]) {
      out = static_cast<float>(node[key].as<double>());
    }
  };

  read_f("tau_px_fixed", cfg.tau_px_fixed);
  read_f("raft_noise_base_px", cfg.raft_noise_base_px);
  read_f("raft_noise_near_px", cfg.raft_noise_near_px);
  read_f("geo_flow_min_sq", cfg.geo_flow_min_sq);
  read_f("min_depth_m", cfg.min_depth_m);
  read_f("max_depth_m", cfg.max_depth_m);

  cfg.tau_px_fixed = std::max(0.1f, cfg.tau_px_fixed);
  cfg.raft_noise_base_px = std::max(0.0f, cfg.raft_noise_base_px);
  cfg.raft_noise_near_px = std::max(0.0f, cfg.raft_noise_near_px);
  cfg.geo_flow_min_sq = std::max(0.0f, cfg.geo_flow_min_sq);
  cfg.min_depth_m = std::max(1e-3f, cfg.min_depth_m);
  cfg.max_depth_m = std::max(cfg.min_depth_m + 1e-3f, cfg.max_depth_m);
  return cfg;
}

Eigen::Affine3d LoadImuBodyFromCameraAffine(const std::filesystem::path &params_file) {
  Eigen::Affine3d T_body_cam = Eigen::Affine3d::Identity();
  const YAML::Node root = LoadYamlFileOrEmpty(params_file);
  const YAML::Node node = root["T_imu_cam"];
  if (!node) {
    return T_body_cam;
  }

  const Eigen::Vector3d rotation_rpy = ReadVector3(node["rotation"], Eigen::Vector3d::Zero());
  const Eigen::Vector3d translation = ReadVector3(node["translation"], Eigen::Vector3d::Zero());
  if (!node["rotation"] && !node["translation"]) {
    return T_body_cam;
  }

  T_body_cam.linear() = RpyToRotation(rotation_rpy);
  T_body_cam.translation() = translation;
  // p_body = T_body_cam * p_cam（与 utils.h 中 T_imu_cam / T_body_cam 约定一致）
  return T_body_cam;
}

Eigen::Matrix4d LoadImuCameraExtrinsic(const std::filesystem::path &params_file) {
  const Eigen::Affine3d T_body_cam = LoadImuBodyFromCameraAffine(params_file);
  Eigen::Matrix4d T_imu_cam = Eigen::Matrix4d::Identity();
  T_imu_cam.block<3, 3>(0, 0) = T_body_cam.linear();
  T_imu_cam.block<3, 1>(0, 3) = T_body_cam.translation();
  return T_imu_cam;
}

bool ParseYamlScalarDouble(const std::filesystem::path &file_path, const std::string &key,
                           double &out_value) {
  const YAML::Node root = LoadYamlFileOrEmpty(file_path);
  if (!root) {
    return false;
  }
  return FindScalarKeyDeep(root, key, out_value);
}

} // namespace sio_utils
