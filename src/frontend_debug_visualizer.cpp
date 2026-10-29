#include "frontend_debug_visualizer.h"

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <yaml-cpp/yaml.h>

namespace {

using CloudPtr = pcl::PointCloud<PointTypeRGB>::Ptr;

CloudPtr CloneOrEmpty(const CloudPtr &cloud) {
  auto out = pcl::PointCloud<PointTypeRGB>::Ptr(new pcl::PointCloud<PointTypeRGB>());
  if (cloud && !cloud->empty()) {
    *out = *cloud;
  }
  return out;
}

const char *DisplayEnv() {
  const char *display = std::getenv("DISPLAY");
  return (display && display[0] != '\0') ? display : nullptr;
}

} // namespace

FrontendDebugVisualizer::FrontendDebugVisualizer() = default;
FrontendDebugVisualizer::~FrontendDebugVisualizer() = default;

bool LoadFrontendDebugVisualConfig(const std::string &yaml_path,
                                   FrontendDebugVisualConfig &cfg) {
  if (yaml_path.empty()) {
    return false;
  }
  try {
    const YAML::Node root = YAML::LoadFile(yaml_path);
    if (!root["debug"]) {
      return false;
    }
    const YAML::Node node = root["debug"];
    cfg.enable = node["visualize"].as<bool>(false);
    cfg.pause_ms = node["visualize_pause_ms"].as<int>(0);
    cfg.voxel_m = node["visualize_voxel_m"].as<float>(0.08f);
    cfg.max_points = node["visualize_max_points"].as<int>(8000);
    return true;
  } catch (const std::exception &e) {
    std::cerr << "[FrontendViz] Failed to load visualize config: " << e.what() << std::endl;
    return false;
  }
}

void FrontendDebugVisualizer::Configure(const FrontendDebugVisualConfig &cfg) {
  cfg_ = cfg;
  logged_stop_ = false;
  reset_camera_once_ = false;
  if (!cfg_.enable) {
    viewer_.reset();
    traj_world_.clear();
    window_opened_ = false;
    return;
  }

  if (!DisplayEnv()) {
    std::cerr << "[FrontendViz] WARNING: DISPLAY is not set — PCL window cannot appear. "
                 "Run from a local desktop terminal or `export DISPLAY=:0` / `:1`.\n";
  } else {
    std::cout << "[FrontendViz] enabled on DISPLAY=" << DisplayEnv()
              << " pause_ms=" << cfg_.pause_ms << "\n";
  }
}

void FrontendDebugVisualizer::EnsureViewer() {
  if (viewer_) {
    return;
  }
  if (!DisplayEnv()) {
    std::cerr << "[FrontendViz] Skip window creation: DISPLAY is unset.\n";
    cfg_.enable = false;
    return;
  }

  viewer_.reset(new pcl::visualization::PCLVisualizer("SIO Frontend Debug", true));
  viewer_->setBackgroundColor(0.05, 0.05, 0.08);
  viewer_->setSize(1600, 960);
  viewer_->setPosition(80, 40);
  viewer_->createViewPort(0.0, 0.5, 0.333, 1.0, vp_imu_);
  viewer_->createViewPort(0.333, 0.5, 0.666, 1.0, vp_stereo_);
  viewer_->createViewPort(0.666, 0.5, 1.0, 1.0, vp_dynamic_);
  viewer_->createViewPort(0.0, 0.0, 0.333, 0.5, vp_gicp_);
  viewer_->createViewPort(0.333, 0.0, 0.666, 0.5, vp_eskf_);
  viewer_->createViewPort(0.666, 0.0, 1.0, 0.5, vp_esikf_);
  viewer_->addText("IMU prior traj (world)", 10, 10, 14, 1.0, 1.0, 1.0, "title_imu", vp_imu_);
  viewer_->addText("Stereo recon (curr body)", 10, 10, 14, 1.0, 1.0, 1.0, "title_stereo", vp_stereo_);
  viewer_->addText("Dynamic filter (kf body)", 10, 10, 14, 1.0, 1.0, 1.0, "title_dyn", vp_dynamic_);
  viewer_->addText("GICP align (kf body)", 10, 10, 14, 1.0, 1.0, 1.0, "title_gicp", vp_gicp_);
  viewer_->addText("ESKF align (kf body)", 10, 10, 14, 1.0, 1.0, 1.0, "title_eskf", vp_eskf_);
  viewer_->addText("ESIKF align (kf body)", 10, 10, 14, 1.0, 1.0, 1.0, "title_esikf", vp_esikf_);

  RefreshWindow(200);
  window_opened_ = !viewer_->wasStopped();
  if (window_opened_) {
    std::cout << "[FrontendViz] Window opened (1600x960). Keep this terminal focused on the same "
                 "desktop session as your monitor.\n";
  } else if (!logged_stop_) {
    logged_stop_ = true;
    std::cerr << "[FrontendViz] Window failed to open (wasStopped=true). Check DISPLAY / OpenGL.\n";
    cfg_.enable = false;
  }
}

void FrontendDebugVisualizer::ResetViewportCameras() {
  if (!viewer_) {
    return;
  }
  viewer_->resetCamera();
}

void FrontendDebugVisualizer::RefreshWindow(int spin_ms) {
  if (!viewer_) {
    return;
  }
  const int ms = std::max(spin_ms, 10);
  viewer_->spinOnce(static_cast<double>(ms));
}

FrontendDebugVisualizer::CloudPtr
FrontendDebugVisualizer::PrepareCloud(const CloudPtr &cloud) const {
  CloudPtr out = CloneOrEmpty(cloud);
  if (out->empty()) {
    return out;
  }
  if (cfg_.voxel_m > 0.0f) {
    pcl::VoxelGrid<PointTypeRGB> voxel;
    voxel.setInputCloud(out);
    voxel.setLeafSize(cfg_.voxel_m, cfg_.voxel_m, cfg_.voxel_m);
    CloudPtr filtered(new pcl::PointCloud<PointTypeRGB>());
    voxel.filter(*filtered);
    out = filtered;
  }
  if (cfg_.max_points > 0 && static_cast<int>(out->size()) > cfg_.max_points) {
    CloudPtr sampled(new pcl::PointCloud<PointTypeRGB>());
    sampled->points.reserve(static_cast<size_t>(cfg_.max_points));
    const double step =
        static_cast<double>(out->size()) / static_cast<double>(cfg_.max_points);
    for (int i = 0; i < cfg_.max_points; ++i) {
      const size_t idx = static_cast<size_t>(std::min(
          out->size() - 1, static_cast<size_t>(static_cast<int>(i * step))));
      sampled->points.push_back(out->points[idx]);
    }
    sampled->width = static_cast<uint32_t>(sampled->size());
    sampled->height = 1;
    out = sampled;
  }
  return out;
}

FrontendDebugVisualizer::CloudPtr
FrontendDebugVisualizer::TransformCloud(const CloudPtr &cloud, const Eigen::Matrix4d &T) const {
  CloudPtr out(new pcl::PointCloud<PointTypeRGB>());
  if (!cloud || cloud->empty()) {
    return out;
  }
  pcl::transformPointCloud(*cloud, *out, T.cast<float>());
  out->width = static_cast<uint32_t>(out->size());
  out->height = 1;
  return out;
}

FrontendDebugVisualizer::CloudPtr
FrontendDebugVisualizer::ColorizeCloud(const CloudPtr &cloud, uint8_t r, uint8_t g,
                                     uint8_t b) const {
  CloudPtr out = PrepareCloud(cloud);
  for (auto &pt : out->points) {
    pt.r = r;
    pt.g = g;
    pt.b = b;
  }
  return out;
}

void FrontendDebugVisualizer::ClearViewport(int viewport) {
  viewer_->removeAllPointClouds(viewport);
}

void FrontendDebugVisualizer::ClearImuViewport(int viewport) {
  viewer_->removeAllPointClouds(viewport);
  viewer_->removeAllShapes(viewport);
  viewer_->removeCoordinateSystem("imu_kf_pose", viewport);
  viewer_->removeCoordinateSystem("imu_pred_pose", viewport);
}

void FrontendDebugVisualizer::ShowRgbCloud(int viewport, const std::string &id,
                                         const CloudPtr &cloud, int point_size) {
  if (!cloud || cloud->empty()) {
    return;
  }
  pcl::visualization::PointCloudColorHandlerRGBField<PointTypeRGB> rgb(cloud);
  if (!viewer_->updatePointCloud(cloud, rgb, id)) {
    viewer_->addPointCloud(cloud, rgb, id, viewport);
  }
  viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                           point_size, id, viewport);
}

void FrontendDebugVisualizer::ShowMonoCloud(int viewport, const std::string &id,
                                            const CloudPtr &cloud, double r, double g, double b,
                                            int point_size) {
  if (!cloud || cloud->empty()) {
    return;
  }
  pcl::visualization::PointCloudColorHandlerCustom<PointTypeRGB> color(cloud, r, g, b);
  if (!viewer_->updatePointCloud(cloud, color, id)) {
    viewer_->addPointCloud(cloud, color, id, viewport);
  }
  viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                           point_size, id, viewport);
}

void FrontendDebugVisualizer::ShowRegistration(int viewport, const std::string &prefix,
                                               const CloudPtr &target,
                                               const CloudPtr &source_body,
                                               const Eigen::Matrix4d &T_kf_curr) {
  ClearViewport(viewport);
  const CloudPtr tgt = PrepareCloud(target);
  const CloudPtr src = PrepareCloud(TransformCloud(source_body, T_kf_curr));
  ShowMonoCloud(viewport, prefix + "_tgt", tgt, 180, 180, 180, 1);
  ShowRgbCloud(viewport, prefix + "_src", src, 2);
}

void FrontendDebugVisualizer::ShowImuTrajectory(int viewport,
                                              const FrameDebugVisualInput &input) {
  ClearImuViewport(viewport);
  traj_world_.push_back(input.imu_pred.pos_end);
  while (traj_world_.size() > 500) {
    traj_world_.pop_front();
  }

  for (size_t i = 1; i < traj_world_.size(); ++i) {
    const Eigen::Vector3d &p0 = traj_world_[i - 1];
    const Eigen::Vector3d &p1 = traj_world_[i];
    const pcl::PointXYZ pt0(static_cast<float>(p0.x()), static_cast<float>(p0.y()),
                            static_cast<float>(p0.z()));
    const pcl::PointXYZ pt1(static_cast<float>(p1.x()), static_cast<float>(p1.y()),
                            static_cast<float>(p1.z()));
    viewer_->addLine(pt0, pt1, 1.0, 0.78, 0.2, "imu_traj_line_" + std::to_string(i), viewport);
  }

  CloudPtr traj_cloud(new pcl::PointCloud<PointTypeRGB>());
  traj_cloud->points.reserve(traj_world_.size());
  for (size_t i = 0; i < traj_world_.size(); ++i) {
    const auto &p = traj_world_[i];
    PointTypeRGB pt;
    pt.x = static_cast<float>(p.x());
    pt.y = static_cast<float>(p.y());
    pt.z = static_cast<float>(p.z());
    const double t = traj_world_.size() > 1
                       ? static_cast<double>(i) / static_cast<double>(traj_world_.size() - 1)
                       : 0.0;
    pt.r = static_cast<uint8_t>(255 * t);
    pt.g = static_cast<uint8_t>(80 + 120 * (1.0 - t));
    pt.b = static_cast<uint8_t>(50);
    traj_cloud->points.push_back(pt);
  }
  traj_cloud->width = static_cast<uint32_t>(traj_cloud->size());
  traj_cloud->height = 1;
  ShowRgbCloud(viewport, "imu_traj_pts", traj_cloud, 6);

  const Eigen::Vector3d kf_p = input.imu_kf.pos_end;
  viewer_->addCoordinateSystem(0.3, kf_p.x(), kf_p.y(), kf_p.z(), "imu_kf_pose", viewport);
  const Eigen::Vector3d pred_p = input.imu_pred.pos_end;
  viewer_->addCoordinateSystem(0.3, pred_p.x(), pred_p.y(), pred_p.z(), "imu_pred_pose", viewport);
}

void FrontendDebugVisualizer::ShowDynamicFilter(int viewport,
                                              const FrameDebugVisualInput &input) {
  ClearViewport(viewport);
  if (!input.kf_full_body || input.kf_full_body->empty()) {
    return;
  }

  CloudPtr static_cloud(new pcl::PointCloud<PointTypeRGB>());
  CloudPtr dynamic_cloud(new pcl::PointCloud<PointTypeRGB>());
  CloudPtr invisible_cloud(new pcl::PointCloud<PointTypeRGB>());
  const size_t n = input.kf_full_body->size();
  std::vector<uint8_t> is_dynamic(n, 0);
  std::vector<uint8_t> is_invisible(n, 0);
  for (const int idx : input.invisible_indices) {
    if (idx >= 0 && idx < static_cast<int>(n)) {
      is_invisible[static_cast<size_t>(idx)] = 1;
    }
  }
  for (const int idx : input.dynamic_indices) {
    if (idx >= 0 && idx < static_cast<int>(n) && !is_invisible[static_cast<size_t>(idx)]) {
      is_dynamic[static_cast<size_t>(idx)] = 1;
    }
  }
  for (const int idx : input.static_indices) {
    if (idx >= 0 && idx < static_cast<int>(n) && !is_invisible[static_cast<size_t>(idx)]) {
      static_cloud->points.push_back(input.kf_full_body->points[static_cast<size_t>(idx)]);
    }
  }
  for (size_t i = 0; i < n; ++i) {
    if (is_invisible[i]) {
      invisible_cloud->points.push_back(input.kf_full_body->points[i]);
    } else if (is_dynamic[i]) {
      dynamic_cloud->points.push_back(input.kf_full_body->points[i]);
    }
  }
  static_cloud->width = static_cast<uint32_t>(static_cloud->size());
  static_cloud->height = 1;
  dynamic_cloud->width = static_cast<uint32_t>(dynamic_cloud->size());
  dynamic_cloud->height = 1;
  invisible_cloud->width = static_cast<uint32_t>(invisible_cloud->size());
  invisible_cloud->height = 1;

  ShowRgbCloud(viewport, "dyn_static", PrepareCloud(static_cloud), 2);
  ShowMonoCloud(viewport, "dyn_removed", PrepareCloud(dynamic_cloud), 220, 60, 60, 2);
  // 与 extractStaticPointCloud 不可见点 RGB (0, 200, 255) 一致。
  ShowMonoCloud(viewport, "dyn_invisible", PrepareCloud(invisible_cloud), 0, 200, 255, 2);
}

void FrontendDebugVisualizer::WaitUser() {
  if (cfg_.pause_ms > 0) {
    RefreshWindow(cfg_.pause_ms);
    return;
  }
  std::cout << "[FrontendViz] frame shown — press Enter to continue, close window to stop...\n";
  std::string line;
  std::getline(std::cin, line);
  RefreshWindow(50);
}

void FrontendDebugVisualizer::ShowFrame(const FrameDebugVisualInput &input) {
  if (!cfg_.enable) {
    return;
  }
  EnsureViewer();
  if (!viewer_ || viewer_->wasStopped()) {
    if (!logged_stop_) {
      logged_stop_ = true;
      std::cerr << "[FrontendViz] Window closed or unavailable; disabling visualize.\n";
    }
    cfg_.enable = false;
    return;
  }

  static int last_logged_frame = -1;
  if (input.frame_id != last_logged_frame) {
    last_logged_frame = input.frame_id;
    std::cout << "[FrontendViz] show frame " << input.frame_id
              << (input.is_keyframe ? " (KF)" : "") << "\n";
  }

  ShowImuTrajectory(vp_imu_, input);

  ClearViewport(vp_stereo_);
  ShowRgbCloud(vp_stereo_, "stereo_body", PrepareCloud(input.stereo_body), 1);

  ShowDynamicFilter(vp_dynamic_, input);

  if (input.has_gicp) {
    ShowRegistration(vp_gicp_, "gicp", input.reg_target, input.source_body, input.T_kf_curr_gicp);
  } else {
    ClearViewport(vp_gicp_);
  }

  if (input.has_eskf) {
    ShowRegistration(vp_eskf_, "eskf", input.reg_target, input.source_body, input.T_kf_curr_eskf);
  } else {
    ClearViewport(vp_eskf_);
  }

  if (input.has_esikf) {
    ShowRegistration(vp_esikf_, "esikf", input.esikf_target, input.source_body,
                     input.T_kf_curr_esikf);
  } else {
    ClearViewport(vp_esikf_);
  }

  if (!reset_camera_once_) {
    ResetViewportCameras();
    reset_camera_once_ = true;
  }
  RefreshWindow(window_opened_ ? 100 : 50);
  window_opened_ = true;
  WaitUser();
}
