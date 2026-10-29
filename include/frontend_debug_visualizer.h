#pragma once

#include "data.h"

#include <Eigen/Dense>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace pcl {
namespace visualization {
class PCLVisualizer;
}
} // namespace pcl

struct FrontendDebugVisualConfig {
  bool enable = false;
  int pause_ms = 0; // 0=每帧按 Enter 继续，>0=自动刷新间隔(ms)
  float voxel_m = 0.08f;
  int max_points = 8000;
};

bool LoadFrontendDebugVisualConfig(const std::string &yaml_path, FrontendDebugVisualConfig &cfg);

struct FrameDebugVisualInput {
  int frame_id = 0;
  bool is_keyframe = false;

  ImuStatesGroup imu_pred;
  ImuStatesGroup imu_kf;

  pcl::PointCloud<PointTypeRGB>::Ptr stereo_body;
  pcl::PointCloud<PointTypeRGB>::Ptr kf_full_body;
  std::vector<int> static_indices;
  std::vector<int> dynamic_indices;
  std::vector<int> invisible_indices;

  pcl::PointCloud<PointTypeRGB>::Ptr reg_target;
  pcl::PointCloud<PointTypeRGB>::Ptr source_body;

  Eigen::Matrix4d T_kf_curr_imu_prior = Eigen::Matrix4d::Identity();
  bool has_imu_prior = false;

  Eigen::Matrix4d T_kf_curr_gicp = Eigen::Matrix4d::Identity();
  bool has_gicp = false;

  Eigen::Matrix4d T_kf_curr_eskf = Eigen::Matrix4d::Identity();
  bool has_eskf = false;

  pcl::PointCloud<PointTypeRGB>::Ptr esikf_target;
  Eigen::Matrix4d T_kf_curr_esikf = Eigen::Matrix4d::Identity();
  bool has_esikf = false;
};

class FrontendDebugVisualizer {
public:
  FrontendDebugVisualizer();
  ~FrontendDebugVisualizer();
  FrontendDebugVisualizer(const FrontendDebugVisualizer &) = delete;
  FrontendDebugVisualizer &operator=(const FrontendDebugVisualizer &) = delete;

  void Configure(const FrontendDebugVisualConfig &cfg);
  bool enabled() const { return cfg_.enable; }

  void ShowFrame(const FrameDebugVisualInput &input);

private:
  using CloudPtr = pcl::PointCloud<PointTypeRGB>::Ptr;

  void EnsureViewer();
  void ResetViewportCameras();
  void RefreshWindow(int spin_ms);
  CloudPtr PrepareCloud(const CloudPtr &cloud) const;
  CloudPtr TransformCloud(const CloudPtr &cloud, const Eigen::Matrix4d &T) const;
  CloudPtr ColorizeCloud(const CloudPtr &cloud, uint8_t r, uint8_t g, uint8_t b) const;
  void ClearViewport(int viewport);
  void ClearImuViewport(int viewport);
  void ShowRgbCloud(int viewport, const std::string &id, const CloudPtr &cloud, int point_size = 1);
  void ShowMonoCloud(int viewport, const std::string &id, const CloudPtr &cloud, double r, double g,
                     double b, int point_size = 1);
  void ShowRegistration(int viewport, const std::string &prefix, const CloudPtr &target,
                        const CloudPtr &source_body, const Eigen::Matrix4d &T_kf_curr);
  void ShowImuTrajectory(int viewport, const FrameDebugVisualInput &input);
  void ShowDynamicFilter(int viewport, const FrameDebugVisualInput &input);
  void WaitUser();

  FrontendDebugVisualConfig cfg_;
  std::unique_ptr<pcl::visualization::PCLVisualizer> viewer_;
  std::deque<Eigen::Vector3d> traj_world_;
  int vp_imu_ = 0;
  int vp_stereo_ = 1;
  int vp_dynamic_ = 2;
  int vp_gicp_ = 3;
  int vp_eskf_ = 4;
  int vp_esikf_ = 5;
  bool window_opened_ = false;
  bool logged_stop_ = false;
  bool reset_camera_once_ = false;
};
