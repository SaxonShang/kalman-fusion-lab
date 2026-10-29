#pragma once
/**
 * @file color_icp.h
 * @brief Color ICP / small_gicp / 投影 Stereo-GICP 配准入口。
 *
 * - ColorICPRegistration：Colored ICP（几何+光度）
 * - EstimatePoseGICP*：欧氏 GICP（含关键帧 prep 缓存）
 * - EstimatePoseStereoGICP：投影关联独立配准，供普通帧 ESKF
 */

#include "profiler/simple_profiler.h"
#include "camera_intrinsics.h"

#include <Eigen/Geometry>
#include <memory>
#include <opencv2/core.hpp>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/registration/registration_helper.hpp>
#include <string>
#include <vector>

namespace Eigen {
  typedef Eigen::Matrix<double, 6, 6> Matrix6d;
  typedef Eigen::Matrix<double, 6, 1> Vector6d;
} // namespace Eigen

class ColorICP {
public:
  using PointT = pcl::PointXYZRGBNormal;
  using NormalT = pcl::PointXYZRGBNormal;
  using PointCloudPtr = pcl::PointCloud<PointT>::Ptr;

  explicit ColorICP(const std::string &params_file = "");
  ~ColorICP() = default;

  /**
   * @brief Colored ICP 配准（几何点到面 + 切平面光度）。
   * @param source 当前帧（含法向）
   * @param target 上一关键帧（含法向）
   * @param target_color_gradient 与 target 对齐的颜色梯度
   * @param target_kdtree 目标 KD 树（调用方复用）
   * @param initial_guess IMU 预积分初值（target←source）
   * @return 优化后的 4×4 变换
   */
  Eigen::Matrix4d ColorICPRegistration(
    const PointCloudPtr &source, const PointCloudPtr &target,
    const std::vector<Eigen::Vector3d> &target_color_gradient,
    const pcl::search::KdTree<PointT> &target_kdtree,
    Eigen::Matrix4d initial_guess);

  /** @brief GICP/StereoGICP 统一结果：位姿 + Hessian 逆协方差。 */
  struct GICPResult {
    Eigen::Matrix4f T;                      ///< target ← source
    Eigen::Matrix<double, 6, 6> covariance; ///< [rot(3), trans(3)] 左扰动系
    int num_inliers = 0;
    int iterations = 0;
    bool converged = false;
    int downsample_target = 0;
    int downsample_source = 0;
    bool target_prep_cached = false;
    bool covariance_from_hessian = false;
    double error = 0.0;
  };

  /**
   * @brief 配置欧氏 GICP 体素与最大对应距离。
   * @param voxel_size 下采样叶长 (m)
   * @param max_correspondence_distance 最大对应距离 (m)
   */
  void ConfigureGICP(double voxel_size, double max_correspondence_distance);

  /** @brief 使关键帧 small_gicp prep 缓存失效。 */
  void InvalidateGicpTargetPrepCache();

  /**
   * @brief 关键帧上预计算 small_gicp 点/法向/协方差。
   * @param cache_cloud target_cache 点云
   * @param source_indices 相对完整参考云的下标
   * @param keyframe_ts 关键帧时间戳，用于缓存命中
   * @return 成功构建为 true
   */
  bool BuildGicpKeyframeTargetPrep(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cache_cloud,
                                   const std::vector<size_t> &source_indices,
                                   double keyframe_ts);

  /**
   * @brief 欧氏 GICP（small_gicp），协方差供 ESKF。
   * @param target_cloud,source_cloud 目标/源点云
   * @param init_guess 初值
   * @param target_keyframe_ts ≥0 时尝试复用关键帧 prep
   */
  GICPResult EstimatePoseGICP(pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_cloud,
                              pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_cloud,
                              const Eigen::Matrix4d &init_guess = Eigen::Matrix4d::Identity(),
                              double target_keyframe_ts = -1.0 SIO_PROFILER_PARAMS);

  /**
   * @brief 投影 Stereo-GICP：相机系源点 ↔ 关键帧深度（+可选 splat 目标）。
   * @param source_cam 当前帧相机系点云
   * @param T_curr_cam_to_prevkf_cam 初值：prevkf_cam ← curr_cam
   * @param camera 内参
   * @param target_depth 关键帧左目深度
   * @param T_imu_cam 外参，用于把 T/cov 转到 body
   * @param target_cam 非空则只从该静态云 splat 深度
   * @return body 系 T 与协方差
   */
  GICPResult EstimatePoseStereoGICP(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &source_cam,
    const Eigen::Matrix4d &T_curr_cam_to_prevkf_cam, const CameraIntrinsics &camera,
    const cv::Mat &target_depth, const Eigen::Matrix4d &T_imu_cam,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &target_cam = nullptr SIO_PROFILER_PARAMS);

  /**
   * @brief 按 static_indices 筛选关键帧 prep 子集做 GICP（动态滤除后 fallback）。
   * @param static_indices 相对完整参考的静态下标
   * @param fallback_target_cloud prep 未命中时的目标云
   */
  GICPResult EstimatePoseGICPWithCacheFilter(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_cloud, const Eigen::Matrix4d &init_guess,
    double target_keyframe_ts, const std::vector<int> &static_indices,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr fallback_target_cloud SIO_PROFILER_PARAMS);

private:
  /** @brief 6 维李代数增量 → 4×4 SE(3)。 */
  static Eigen::Matrix4d TransformVector6dToMatrix4d(const Eigen::Vector6d &input);

  /**
   * @brief 在已 prep 的 target 上对齐 source（small_gicp 核心）。
   * @param target_prep_from_keyframe_cache 是否来自关键帧缓存
   */
  GICPResult AlignGICP(const small_gicp::PointCloud::Ptr &target_cloud_prep,
                       const std::shared_ptr<small_gicp::KdTree<small_gicp::PointCloud>>
                         &target_kdtree,
                       pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_cloud,
                       const Eigen::Matrix4d &init_guess, bool target_prep_from_keyframe_cache
                         SIO_PROFILER_PARAMS);

private:
  // 共享 KD-Tree（目标点云搜索器）
  pcl::search::KdTree<PointT, pcl::KdTreeFLANN<PointT, flann::L2_Simple<float>>>::Ptr kdtree_;

  pcl::PointCloud<PointT>::Ptr source_cloud_;
  pcl::PointCloud<PointT>::Ptr target_cloud_;

  // target 上预计算的颜色梯度，与 target 点一一对齐
  std::vector<Eigen::Vector3d> target_color_gradient_;

  // RGB -> 亮度加权系数（可配置为 NTSC / sRGB / 均匀）
  Eigen::Vector3d rgb_to_intensity_weight_;

  // 当前累计变换（由 ColorICPRegistration 更新）
  Eigen::Matrix4d transformation_;

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

  struct GicpKeyframePrep {
    double keyframe_ts = -1.0;
    double voxel_size = 0.0;
    small_gicp::PointCloud::Ptr cloud;
    std::vector<size_t> source_indices;
    std::shared_ptr<small_gicp::KdTree<small_gicp::PointCloud>> kdtree;
    bool valid = false;
  } gicp_keyframe_prep_;
};
