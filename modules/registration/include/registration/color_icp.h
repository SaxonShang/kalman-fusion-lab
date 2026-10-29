#pragma once
// Color ICP Registration
// Hanzhe Teng, Feb 2022
// Adapted for SIO_SLAM

#include <Eigen/Geometry>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <string>
#include <limits>
#include <vector>

#include <yaml-cpp/yaml.h>
#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/registration/registration_helper.hpp>

#include "registration/stereo_gicp.h"
#include "profiler/simple_profiler.h"

namespace Eigen {
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
} // namespace Eigen

class ColorICP {
public:
  using PointT = pcl::PointXYZRGBNormal;
  using NormalT = pcl::PointXYZRGBNormal;
  using PointCloudPtr = pcl::PointCloud<PointT>::Ptr;

  // 默认读取项目配置；模块重放显式指定冻结参数快照。
  explicit ColorICP(const std::string &params_file = "");
  ~ColorICP() = default;

  // 主接口 1：Colored ICP 配准
  // source: 当前帧点云（PointXYZRGBNormal）
  // target: 上一关键帧点云（PointXYZRGBNormal，含法向）
  // target_color_gradient: 预计算的颜色梯度（与 target 一一对齐）
  // thread_kdtrees: 预构建的每线程 KD-Tree（避免函数内重复建树）
  // initial_guess: IMU 预积分提供的初始位姿
  Eigen::Matrix4d ColorICPRegistration(
      const PointCloudPtr &source, const PointCloudPtr &target,
      const std::vector<Eigen::Vector3d> &target_color_gradient,
      const std::vector<pcl::search::KdTree<PointT>::Ptr> &thread_kdtrees,
      Eigen::Matrix4d initial_guess);

  enum class RegistrationType {
    GICP = 0,
    ICP = 1,
    PLANE_ICP = 2,
    STEREO_GICP = 3
  };

  enum class CovarianceConvention { RightSource, LeftTarget };

  // 切换配准方法时，目标端法向/协方差缓存的需求也会变化，必须失效并重建。
  void SetRegistrationType(RegistrationType type);
  void SetRegistrationType(const std::string &type_str);
  RegistrationType GetRegistrationType() const { return registration_type_; }
  std::string GetRegistrationTypeString() const;

  // GICP 配准结果：变换矩阵 + 6×6 位姿协方差（从 Hessian 逆得到）
  struct GICPResult {
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity(); // target ← source（body 系）
    Eigen::Matrix<double, 6, 6>
        covariance = Eigen::Matrix<double, 6, 6>::Identity() * 1e-2;
    CovarianceConvention covariance_convention = CovarianceConvention::RightSource;
    bool covariance_from_hessian = false;
    int num_inliers = 0;
    int num_iterations = 0;
    bool converged = false;
    double error = 0.0;
    double rmse = std::numeric_limits<double>::quiet_NaN(); // Stereo only; native GICP does not report it.
    double fitness = 0.0; // 有效对应 / 源点数；投影路径用这个当置信度
    double preprocess_ms = 0.0;
    double align_ms = 0.0;
    int downsampled_target_points = 0;
    int downsampled_source_points = 0;
    bool target_prep_cached = false;
    std::string registration_type = "GICP";
    std::string status;
    std::vector<sio_stereo_gicp::IterationTrace> stereo_trace;
  };

  // 覆盖 GICP 体素降采样与对应距离（ROS 参数可在 frontend 启动时注入）
  void ConfigureGICP(double voxel_size, double max_correspondence_distance);
  // Stereo-GICP 早停阈值与协方差对角地板。rot_eps 弧度，trans_eps / floor_trans 米，floor_rot 度。
  void ConfigureStereoGicpSolver(double gn_rot_eps, double gn_trans_eps,
                                 double cov_floor_trans_m,
                                 double cov_floor_rot_deg);
  void InvalidateGicpTargetPrepCache();

  // 关键帧：对 target_cache 点云（已体素化）预计算 small_gicp 点/法向/协方差，与 source_indices 对齐。
  bool BuildGicpKeyframeTargetPrep(
      const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cache_cloud,
      const std::vector<size_t> &source_indices, double keyframe_ts);

  // 主接口 2：GICP 配准（small_gicp），同时返回协方差供 ESKF 使用
  GICPResult EstimatePoseGICP(
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_cloud,
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_cloud,
      const Eigen::Matrix4d &init_guess = Eigen::Matrix4d::Identity(),
      double target_keyframe_ts = -1.0
      SIO_PROFILER_PARAMS);

  // 普通帧：按 static_indices 从关键帧 prep 索引筛选 target，仅重建子集 KD-tree；失败时 fallback 稠密 target。
  GICPResult EstimatePoseGICPWithCacheFilter(
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_cloud,
      const Eigen::Matrix4d &init_guess, double target_keyframe_ts,
      const std::vector<int> &static_indices,
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr fallback_target_cloud
      SIO_PROFILER_PARAMS);

  // 相机系投影 Stereo-GICP。T / covariance 返回 body 系，方便 ESKF 复用现网路径。
  GICPResult EstimatePoseStereoGICP(
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_cloud_camera,
      const cv::Mat &target_disp,
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_cloud_camera,
      const CameraIntrinsics &model_camera, const Eigen::Matrix4d &T_imu_cam,
      const Eigen::Matrix4d &T_curr_cam_to_prevkf_cam
      SIO_PROFILER_PARAMS);

private:
  // 6 维李代数增量 -> 4x4 SE(3)
  static Eigen::Matrix4d
  TransformVector6dToMatrix4d(const Eigen::Vector6d &input);

private:
  // 共享 KD-Tree（目标点云搜索器）
  pcl::search::KdTree<
      PointT, pcl::KdTreeFLANN<PointT, flann::L2_Simple<float>>>::Ptr kdtree_;

  pcl::PointCloud<PointT>::Ptr source_cloud_;
  pcl::PointCloud<PointT>::Ptr target_cloud_;

  // target 上预计算的颜色梯度，与 target 点一一对齐
  std::vector<Eigen::Vector3d> target_color_gradient_;

  // RGB -> 亮度加权系数（可配置为 NTSC / sRGB / 均匀）
  Eigen::Vector3d rgb_to_intensity_weight_;

  // 当前累计变换（由 ColorICPRegistration 更新）
  Eigen::Matrix4d transformation_;

  // 从 YAML 加载的算法参数
  YAML::Node params_;

  int num_threads_;
  int gradient_min_neighbors_;
  int gradient_max_neighbors_;
  double nn_search_epsilon_;
  bool verbose_iteration_;

  // 缓存的配准参数（构造时从 YAML 读取一次，避免每次配准/迭代重复解析）
  int max_iterations_;
  double transformation_epsilon_;
  double max_corres_dist_;
  double color_icp_lambda_;
  double gicp_voxel_size_ = 0.05;
  double gicp_max_correspondence_distance_ = 0.10;
  double stereo_gn_rot_eps_ = 1e-6;
  double stereo_gn_trans_eps_ = 1e-6;
  double stereo_cov_floor_trans_m_ = 0.0;
  double stereo_cov_floor_rot_deg_ = 0.0;
  RegistrationType registration_type_ = RegistrationType::GICP;

  struct GicpKeyframePrep {
    double keyframe_ts = -1.0;
    double voxel_size = 0.0;
    small_gicp::PointCloud::Ptr cloud;
    std::vector<size_t> source_indices;
    std::shared_ptr<small_gicp::KdTree<small_gicp::PointCloud>> kdtree;
    bool valid = false;
  } gicp_keyframe_prep_;

  GICPResult AlignGICP(
      const small_gicp::PointCloud::Ptr &target_cloud_prep,
      const std::shared_ptr<small_gicp::KdTree<small_gicp::PointCloud>>
          &target_kdtree,
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_cloud,
      const Eigen::Matrix4d &init_guess, bool target_prep_from_keyframe_cache
      SIO_PROFILER_PARAMS);
};
