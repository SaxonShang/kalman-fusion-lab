#pragma once

/**
 * @file frontend.h
 * @brief SIOFrontend：前端主循环，编排双目重建、配准、IMU 融合与关键帧推进。
 *
 * 典型一帧：取图/IMU → 视差点云 →（关键帧 ESIKF | 普通帧 StereoGICP+ESKF）
 * → 关键帧判定 → 推后端窗口优化。
 */

#include "backend.h"
#include "color_icp.h"
#include "data.h"
#include "fusion_debug.h"
#include "frontend_debug_visualizer.h"
#include "imu_debug.h"
#include "imu_preprocessor.h"
#include "sio_frontend_modules.h"

#include <atomic>
#include <future>
#include <mutex>
#include "profiler/simple_profiler.h"

class SVIO_SLAM;

/** @brief 跟踪目标点云及其置信度/协方差/源索引打包。 */
struct TrackingTargetData {
  pcl::PointCloud<PointTypeRGB>::Ptr cloud;
  std::vector<float> point_confidences;
  std::vector<size_t> source_indices;
  std::vector<Eigen::Matrix3d> point_covariances;
};

class SIOFrontend {
public:
  /**
   * @brief 构造前端并绑定 SLAM 节点（取参、取传感器队列）。
   * @param slam 外部 SVIO_SLAM 管理节点
   */
  explicit SIOFrontend(SVIO_SLAM *slam);
  ~SIOFrontend();

  /** @brief 阻塞运行前端主循环，直至 stop()。 */
  void run();

  /** @brief 请求停止前端与后端线程。 */
  void stop() {
    is_running_.store(false);
    backend_.Stop();
  }

private:
  SVIO_SLAM *slam_;

  std::atomic<bool> is_running_;
  bool ESIKF_fused_ = false;   ///< 是否已完成首帧 ESIKF 融合
  bool imu_initialized_ = false;

  std::unique_ptr<DynamicCloudProcessor> processor_;
  FeatureProcessor feat_proc_;
  ImuIntegrator imu_int_;
  StateEstimator estimator_;
  KeyframeManager kf_manager_;
  ColorICP color_icp_;
  ImuPreprocessor imu_preprocessor_;
  FusionDebugLogger fusion_debug_logger_;
  ImuDebugLogger imu_debug_logger_;
  std::unique_ptr<FrontendDebugVisualizer> debug_viz_;
  bool enable_debug_console_ = false;
  bool enable_track_console_ = false;

  std::shared_ptr<FrameData> processing_frame_;
  std::shared_ptr<FrontFrameData> front_frame_cur_;
  KeyFrameData prev_frame_;
  KeyFrameData prev_keyframe_;
  KeyFrameData prev_backend_keyframe_;

  ImuStatesGroup imu_states_;          ///< 最新滤波后验（世界系）
  ImuStatesGroup imu_pred_state_curr_; ///< 当前帧 IMU 先验
  /// 光流门控并行协方差（全 21×21）。关键帧时刻置零，其后与主滤波同步做
  /// IMU 预测和 GICP/ESIKF 的 Joseph 更新。只供 Σ_f，不替换 imu_states_.cov。
  Eigen::Matrix<double, DIM_STATE, DIM_STATE> flow_gate_cov_;
  Pose6D imu_preint_increment_;        ///< 最近一次预积分增量

  pcl::PointCloud<PointTypeRGB>::Ptr pointcloud_cur_;        ///< 机体系点云
  pcl::PointCloud<PointTypeRGB>::Ptr pointcloud_cur_camera_; ///< 相机系点云
  std::vector<Eigen::Matrix3d> point_covariances_cur_body_; ///< 与 pointcloud_cur_ 对齐

  /** @brief 上一前端关键帧预计算跟踪缓存（切换前复用）。 */
  struct TargetCache {
    pcl::PointCloud<PointTypeRGB>::Ptr cloud;
    pcl::KdTreeFLANN<PointTypeRGB> kdtree;
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr normals;
    std::vector<Eigen::Vector3d> color_gradient;
    std::vector<float> point_confidences;
    std::vector<Eigen::Matrix3d> point_covariances;
    std::vector<size_t> source_indices; ///< 相对完整参考点云的下标

    TargetCache()
      : cloud(new pcl::PointCloud<PointTypeRGB>())
      , normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>) {
    }
  } target_cache_;

  pcl::PointCloud<PointTypeRGB>::Ptr prev_keyframe_full_cloud_body_;
  StereoCloud prev_keyframe_full_cloud_camera_; ///< 相机系动态滤除参考，u/v/has_pixel 保留到几何光流
  std::vector<float> prev_keyframe_full_confidences_;
  pcl::PointCloud<PointTypeRGB>::Ptr prev_keyframe_invisible_cloud_; ///< 相机系不可见点（青色）
  TargetCache target_cache_base_;
  bool target_cache_base_valid_ = false;

  std::deque<sensor_msgs::msg::Imu::SharedPtr> imu_buffer_;

  std::deque<std::shared_ptr<KeyFrameData>> frontend_keyframe_deque_;
  std::mutex mtx_frontend_;
  std::future<void> target_cache_future_;

  SIOBackend backend_;

  /**
   * @brief 由目标点云重建跟踪缓存：体素下采样、法向、颜色梯度、KD 树。
   * @param cloud 机体系目标点云
   * @param point_confidences,point_covariances 与输入对齐；下采样后重映射
   * @param frame_id,keyframe_id 日志用，可 -1
   */
  void RebuildTargetCache(const pcl::PointCloud<PointTypeRGB>::Ptr &cloud,
                          const std::vector<float> &point_confidences,
                          const std::vector<Eigen::Matrix3d> &point_covariances,
                          int frame_id = -1, int keyframe_id = -1);

  /** @brief 打印 target cache 重建统计。 */
  void LogTargetCacheRebuild(int frame_id, int keyframe_id, size_t raw_points,
                             size_t depth_valid_points, size_t voxel_cache_points,
                             float voxel_size_m) const;

  /** @brief 打印关键帧跟踪缓存与 ESIKF 漏斗摘要。 */
  void LogKfTrackCache(int frame_id, size_t full_ref, size_t static_ref, size_t cache_base,
                       size_t cache_esikf, const ESIKFDebugSummary &esikf_debug) const;

  /**
   * @brief 按动态滤除静态索引过滤已有 cache，避免重复估法向/梯度。
   * @param static_indices 相对完整参考点云的静态下标
   */
  void FilterTargetCacheByStaticIndices(const std::vector<int> &static_indices);

  /** @brief 保存动态滤除前的 base cache 快照。 */
  void SaveTargetCacheBase();
  /** @brief 从 base 快照恢复 target_cache_。 */
  void RestoreTargetCacheFromBase();

  /**
   * @brief 关键帧切换时重置动态滤除用的完整参考点云。
   * @param body_cloud 机体系 XYZRGB（GICP / 可视化）
   * @param camera_cloud 相机系 StereoCloud，供几何光流读 u、v
   * @param confidences 点置信度
   */
  void ResetKeyframeDynamicReference(const pcl::PointCloud<PointTypeRGB>::Ptr &body_cloud,
                                     const StereoCloud &camera_cloud,
                                     const std::vector<float> &confidences);

  /**
   * @brief 由相机系 StereoCloud 得到 w=1 的相机系 XYZRGB，并变换出机体系 pointcloud_cur_。
   *        StereoCloud 的 u、v、has_pixel 不参与该变换。
   * @param stereo_camera 相机系视差点云
   * @param T_body_cam 相机→机体
   */
  void TransformPointCloudToBody(const StereoCloud &stereo_camera,
                                 const Eigen::Affine3d &T_body_cam);

  /**
   * @brief 消费后端 WindowOptimize 后验：匹配 frontend_keyframe_id/timestamp，
   *        写回绝对量，后续 KF 沿相对边复合，并切换未来预积分偏置。
   */
  void ApplyBackendStateUpdates();

  /** @brief imu_buffer_ 相邻时间戳是否连续（无超阈值跳变）。 */
  bool isImuDataContinuous() const;
};