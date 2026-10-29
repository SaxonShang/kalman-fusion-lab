#pragma once

/**
 * @file sio_frontend_modules.h
 * @brief 前端算法模块：双目重建/动态滤除、状态估计、特征预处理、关键帧管理。
 *
 * 管线位置（由 SIOFrontend 编排）：
 *   DynamicCloudProcessor → FeatureProcessor / ColorICP(StereoGICP)
 *   → ImuIntegrator → StateEstimator(ESIKF|ESKF) → KeyframeManager → Backend
 */

#include "data.h"

#include <Eigen/Core>
#include <algorithm>
#include <deque>
#include <memory>
#include <mutex>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/msg/imu.hpp>
#include <string>
#include <vector>
#include "profiler/simple_profiler.h"
#include "color_icp.h"
#include "image_preprocessor.h"
#include "imu_debug.h"
#include "imu_integrator.h"
#include "fusion_debug.h"
#include "sio_config.h"
#include "sio_stereo_geometry.h"

class TrtEngine;

struct CUstream_st;
using cudaStream_t = CUstream_st *;


/** @brief 双目深度重建 + RAFT/几何光流动态物体滤除。 */
class DynamicCloudProcessor {
public:
  /**
   * @param plan_disp 视差 TensorRT plan 路径
   * @param plan_raft 光流 TensorRT plan 路径
   * @param cam 推理分辨率下左目内参
   */
  DynamicCloudProcessor(const std::string &plan_disp, const std::string &plan_raft,
                        const CameraIntrinsics &cam);
  ~DynamicCloudProcessor();

  /** @brief 运行时更新内参（须与裁减+缩放后图像对齐）。 */
  void SetCameraIntrinsics(const CameraIntrinsics &cam) {
    cam_ = cam;
  }

  /**
   * @brief 配置中心裁减+缩放几何（如 1280×720→896×448，与 params.yaml 一致）。
   * @param geom 立体图像几何描述
   */
  void ConfigureStereoPreprocess(const sio_utils::StereoImageGeometry &geom);

  /** @brief 设置动态滤除门控参数。 */
  void SetDynamicFilterParams(const sio_utils::DynamicFilterParams &params) {
    dynamic_filter_cfg_ = params;
  }

  /** @brief 读取当前动态滤除参数。 */
  const sio_utils::DynamicFilterParams &GetDynamicFilterParams() const {
    return dynamic_filter_cfg_;
  }

  /**
   * @brief 双目视差推理并做一致性滤波（常在 ROS 回调线程）。
   * @param cpu_left,cpu_right BGR 左右图
   * @return 滤波后视差图 CV_32FC1
   */
  cv::Mat computeFilteredDisparity(const cv::Mat &cpu_left, const cv::Mat &cpu_right);

  /**
   * @brief 视差图反投影为相机系 StereoCloud。
   * @param rgb_img 与视差对齐的彩色图
   * @param h_disp 视差图
   * @param confidence_map 可选置信度图；非空时写入 point_confidences
   * @param point_confidences 非空则逐点输出置信度
   */
  StereoCloud generatePointCloud(const cv::Mat &rgb_img, const cv::Mat &h_disp,
                                 const cv::Mat &confidence_map = cv::Mat(),
                                 std::vector<float> *point_confidences = nullptr);

  /**
   * @brief 动态剔除：位姿协方差→Σ_f 马氏门控 + 投影几何光流 vs RAFT。
   * @param prev_left,curr_left 相邻帧左目图
   * @param prev_pcd 上一关键帧参考点云（相机系 StereoCloud，u/v/has_pixel 有效）
   * @param T_curr_prevkf p_curr = T * p_prevkf（关键帧相机→当前相机）
   * @param prev_disp 上一关键帧视差
   * @param prev_confidences 可选上一帧点置信度
   * @param static_confidences,static_indices,dynamic_indices 可选输出
   * @param pose_cov_cam 可选相机系 6×6 位姿协方差 [δω; δt]
   * @param invisible_cloud 可选不可见点云。has_pixel==0、视野外、相机后方或当前视图被遮挡者；
   *        XYZRGB 着青色 (0, 200, 255)，w=1。既不进入静态云，也不写入 dynamic_indices。
   *        d<=0 的像素没有点，不会出现在这里
   * @param invisible_indices 可选，与 invisible_cloud 对齐的参考点云下标
   * @param static_cloud 可选输出：静态 StereoCloud（u、v、has_pixel 保留）
   * @return true 左图可用、光流已生成且门控已跑完。静态集可为空（可见点全部标为动态），空不是失败。
   *         false 上一帧或当前左图为空，或裁减 ROI 放不下。不可见索引保留几何光流结果，
   *         动态索引为空，未检验的可见点暂写入静态集。三套索引互斥。
   */
  bool removeDynamicObjects(
    const cv::Mat &prev_left, const StereoCloud &prev_pcd, const cv::Mat &curr_left,
    Eigen::Matrix4d T_curr_prevkf, const cv::Mat &prev_disp,
    const std::vector<float> *prev_confidences = nullptr,
    std::vector<float> *static_confidences = nullptr, std::vector<int> *static_indices = nullptr,
    std::vector<int> *dynamic_indices = nullptr,
    const Eigen::Matrix<double, 6, 6> *pose_cov_cam = nullptr,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr *invisible_cloud = nullptr,
    std::vector<int> *invisible_indices = nullptr, StereoCloud *static_cloud = nullptr);

private:
  std::unique_ptr<TrtEngine> algo_disp;
  std::unique_ptr<TrtEngine> algo_raft;
  cudaStream_t stream_disp_;  ///< 视差推理 CUDA 流
  cudaStream_t stream_raft_;  ///< 光流推理 CUDA 流
  CameraIntrinsics cam_;
  sio_utils::StereoImageGeometry stereo_geom_{};
  sio_utils::StereoCropRect stereo_crop_{};
  sio_utils::DynamicFilterParams dynamic_filter_cfg_{};
  bool stereo_preprocess_enabled_ = false;

  /** @brief TensorRT 引擎预热，避免首帧尖峰延迟。 */
  void warmup();

  /** @brief CPU 端立体预处理：裁减/缩放到推理分辨率。 */
  cv::Mat preprocessStereoImageCpu(const cv::Mat &cpu_img) const;

  /**
   * @brief 上传视差网络输入张量到 GPU。
   * @param scale 可选强度缩放
   */
  void uploadDispTensor(const cv::Mat &bgr_inference, float *d_trt_input, float scale) const;

  /** @brief 上传 RAFT 光流网络输入张量到 GPU。 */
  void uploadRaftTensor(const cv::Mat &bgr_inference, float *d_trt_input) const;

  /**
   * @brief 视差局部一致性滤波。
   * @param window_size 邻域窗口
   * @param diff_thresh 视差差阈值
   * @param count_thresh 邻域内一致点数下限
   */
  void filterDisparityMap(const cv::Mat &input_disp, cv::Mat &output_disp, int window_size,
                          float diff_thresh, int count_thresh);

  /**
   * @brief 视差→点云核心实现。有效视差像素写入 StereoPoint 的 u、v，has_pixel=1，point.w=1。
   *        d<=0（含邻域滤波清零）或深度超范围不建点，因而既不进静态云也不标动态。
   * @param zmin,zmax 深度裁剪 (m)
   */
  StereoCloud disparityToPointCloud(const cv::Mat &rgb_img, const cv::Mat &h_disp, float zmin,
                                    float zmax, const cv::Mat &confidence_map = cv::Mat(),
                                    std::vector<float> *point_confidences = nullptr);

  /** @brief 动态滤除内部粗配准（GICP），估计帧间相对位姿。 */
  Eigen::Matrix4f estimatePoseGICP(pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_cloud,
                                   pcl::PointCloud<pcl::PointXYZRGB>::Ptr source_cloud);

  /**
   * @brief 由刚体变换生成投影几何光流图。
   *        参考像素读 StereoPoint 的 u、v、has_pixel。无像素、相机后方、出当前图像或被遮挡的点为不可见，既非静态也非动态。
   *        当前像素仍用 T 投影，同一当前像素只保留更近的点。
   * @param T 目标←源的 4×4 变换
   * @param width,height 图像尺寸
   */
  GeoFlowResult generate_geometric_flow_cpp(const StereoCloud &pcd, const Eigen::Matrix4f &T,
                                            int width, int height, const CameraIntrinsics &camera);

  /** @brief 将 RAFT GPU 输出整理为 CV_32FC2 光流图。 */
  cv::Mat postprocessRaftFlow(float *d_trt_out_flow, int width, int height);

  /**
   * @brief 自适应门控：比较 RAFT 与几何光流，输出动态掩码。
   * @param pose_cov_cam 非空时用马氏距离门控
   */
  cv::Mat detectDynamicObjectsAdaptive(
    const cv::Mat &raft_flow, const cv::Mat &geo_flow, const cv::Mat &h_disp,
    const CameraIntrinsics &camera,
    const Eigen::Matrix<double, 6, 6> *pose_cov_cam = nullptr);

  /**
   * @brief 按动态掩码抽取静态点及可选索引/置信度。
   * @param valid_indices,pixel_coords 点→像素映射
   * @param invisible_indices has_pixel==0、视野外、相机后方或当前视图被遮挡的已建点
   * @param dynamic_mask 非零为动态
   * @param invisible_cloud 可选输出：上述不可见点的 XYZRGB 点云，青色 (0, 200, 255)，w=1，非静态、非动态
   * @param out_invisible_indices 可选输出：不可见点在 original_pcd 中的下标
   */
  StereoCloud extractStaticPointCloud(
    const StereoCloud &original_pcd, const std::vector<int> &valid_indices,
    const std::vector<cv::Point> &pixel_coords, const std::vector<int> &invisible_indices,
    const cv::Mat &dynamic_mask, const std::vector<float> *original_confidences = nullptr,
    std::vector<float> *static_confidences = nullptr, std::vector<int> *static_indices = nullptr,
    std::vector<int> *dynamic_indices = nullptr,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr *invisible_cloud = nullptr,
    std::vector<int> *out_invisible_indices = nullptr);
};

// IMU 预积分 — 见 imu_integrator.h

/**
 * @brief 融合状态估计：关键帧紧耦合 ESIKF / 普通帧松耦合 ESKF。
 *
 * 观测关联见 BuildObservationSubclouds（欧氏）与
 * BuildProjectiveObservationSubclouds（投影 Stereo-GICP）。
 */
class StateEstimator {
public:
  /** @brief 单点几何/光度观测因子（供 ColorPointObserver 累加）。 */
  struct PointObservationFactor {
    // Stereo-GICP 几何项：e = p' - q（目标系 3D），J 对 [δθ_body, δp_world]，W = C^{-1}
    Eigen::Matrix<double, 3, 6> J_geo_mat = Eigen::Matrix<double, 3, 6>::Zero();
    Eigen::Vector3d r_geo_vec = Eigen::Vector3d::Zero();
    Eigen::Matrix3d W_geo = Eigen::Matrix3d::Zero();
    bool has_stereo_geo = false;

    // 兼容标量几何（调试 / 旧测试）；Stereo-GICP 路径下 r_geo≈‖e‖，J_geo 为退化占位
    Eigen::Matrix<double, 6, 1> J_geo = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 1> J_color = Eigen::Matrix<double, 6, 1>::Zero();
    double r_geo = 0.0;
    double r_color = 0.0;
    double match_dist2 = 0.0;
    double grad_strength2 = 0.0;
    double src_pos_var = 0.0;
    double tgt_pos_var = 0.0;
    double pos_var_sum = 0.0;
    double inv_geo_noise = 0.0;
    double inv_color_noise = 0.0;
  };

  /**
   * @brief 欧氏 KNN 关联：在已变换到目标系的源点上搜对应，筛法向有效对。
   * @param source 源点云（目标坐标系，如 prevkf body）
   * @param target_kdtree 目标 KD 树
   * @param target_color_gradient,target_normals 与 target 对齐
   * @param source_point_confidences,target_point_confidences 可空，缺省按 1
   * @param source_corr,target_corr 输出一一对应子点云
   * @param target_corr_color_gradient,target_corr_normals 输出属性
   * @param source_corr_confidences,target_corr_confidences 输出置信度
   * @param max_dist_sq 最大对应距离平方 (m²)
   * @param source_point_covariances,target_point_covariances 可选点协方差
   * @param source_corr_covariances,target_corr_covariances 非空则填充对应协方差
   * @param funnel_out 可选漏斗统计
   */
  void BuildObservationSubclouds(
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
    std::vector<float> &target_corr_confidences,
    float max_dist_sq = 0.5f,
    const std::vector<Eigen::Matrix3d> &source_point_covariances = {},
    const std::vector<Eigen::Matrix3d> &target_point_covariances = {},
    std::vector<Eigen::Matrix3d> *source_corr_covariances = nullptr,
    std::vector<Eigen::Matrix3d> *target_corr_covariances = nullptr,
    CorrespondenceFunnelStats *funnel_out = nullptr);

  /**
   * @brief 投影关联：body→相机投影，命中像素中心光线反投影，再映回 body。
   *
   * search_radius 小于 0 时按深度自适应；颜色/法向从 target KD-tree 最近邻取。
   * 目标协方差用 0.5 px 匹配噪声在相机系现算后转到 body。
   *
   * @param source_body_in_target 已在锚点机体系的源点
   * @param target_kdtree 锚点目标 KD 树（取颜色/法向）
   * @param target_depth_cam 关键帧左目深度 CV_32FC1
   * @param cam 内参
   * @param T_body_cam 外参 相机→body（4×4）
   * @param search_radius 像素半径；小于 0 自适应
   * @param zmin_m,zmax_m,sigma_d_px,z_gate_sigma,edge_sigma 门控参数
   * 其余输出参数同 BuildObservationSubclouds
   */
  void BuildProjectiveObservationSubclouds(
    const pcl::PointCloud<PointTypeRGB>::Ptr &source_body_in_target,
    const pcl::KdTreeFLANN<PointTypeRGB>::Ptr &target_kdtree,
    const std::vector<Eigen::Vector3d> &target_color_gradient,
    const std::vector<Eigen::Vector3d> &target_normals,
    const std::vector<float> &source_point_confidences,
    const std::vector<float> &target_point_confidences,
    const cv::Mat &target_depth_cam, const CameraIntrinsics &cam,
    const Eigen::Matrix4d &T_body_cam, pcl::PointCloud<PointTypeRGB>::Ptr &source_corr,
    pcl::PointCloud<PointTypeRGB>::Ptr &target_corr,
    std::vector<Eigen::Vector3d> &target_corr_color_gradient,
    std::vector<Eigen::Vector3d> &target_corr_normals,
    std::vector<float> &source_corr_confidences,
    std::vector<float> &target_corr_confidences,
    const std::vector<Eigen::Matrix3d> &source_point_covariances = {},
    std::vector<Eigen::Matrix3d> *source_corr_covariances = nullptr,
    std::vector<Eigen::Matrix3d> *target_corr_covariances = nullptr,
    CorrespondenceFunnelStats *funnel_out = nullptr, int search_radius = -1,
    float zmin_m = 0.5f, float zmax_m = 15.0f, float sigma_d_px = 0.3f,
    float z_gate_sigma = 3.0f, float edge_sigma = 3.0f);

  /**
   * @brief 单点观测因子：几何 e=p'-q、J_geo、W=C^{-1}；可选切平面光度。
   * @param source_pt 已在目标系的源点 p'
   * @param target_pt 目标点 q（含颜色）
   * @param target_color_gradient,target_normal 光度用
   * @param source_origin_in_target 源原点在目标系平移
   * @param R_target_to_world 锚点机体→世界
   * @param source_confidence,target_confidence 标量置信度
   * @param factor 输出因子
   * @param R_state_to_target 当前机体→锚点机体
   * @param sensor_origin_in_state 传感器原点在状态系（常为 t_imu_cam 或 0）
   * @param source_cov,target_cov 非空则用相机立体 Σ；否则用置信度对角近似
   * @return 几何项有效为 true
   */
  bool BuildPointObservationFactor(
    const PointTypeRGB &source_pt, const PointTypeRGB &target_pt,
    const Eigen::Vector3d &target_color_gradient, const Eigen::Vector3d &target_normal,
    const Eigen::Vector3d &source_origin_in_target, const Eigen::Matrix3d &R_target_to_world,
    double source_confidence, double target_confidence, PointObservationFactor &factor,
    const Eigen::Matrix3d &R_state_to_target = Eigen::Matrix3d::Identity(),
    const Eigen::Vector3d &sensor_origin_in_state = Eigen::Vector3d::Zero(),
    const Eigen::Matrix3d *source_cov = nullptr,
    const Eigen::Matrix3d *target_cov = nullptr) const;

  /**
   * @brief 累加全部对应点的 6×6 观测信息 H_pose 与信息向量 b_pose（含 Huber）。
   * @param source_corr,target_corr 一一对应子点云
   * @param target_color_gradient,target_normals 对应属性
   * @param source_origin_in_target,R_target_to_world 见 BuildPointObservationFactor
   * @param H_pose,b_pose 累加输出（调用方先清零）
   * @param lambda 几何权重 ∈[0,1]；光度为 (1-λ)
   * @param source_corr_covariances,target_corr_covariances 可选
   */
  void ColorPointObserver(
    const pcl::PointCloud<PointTypeRGB>::Ptr &source_corr,
    const pcl::PointCloud<PointTypeRGB>::Ptr &target_corr,
    const std::vector<Eigen::Vector3d> &target_color_gradient,
    const std::vector<Eigen::Vector3d> &target_normals,
    const Eigen::Vector3d &source_origin_in_target, const Eigen::Matrix3d &R_target_to_world,
    Eigen::Matrix<double, 6, 6> &H_pose, Eigen::Matrix<double, 6, 1> &b_pose,
    const std::vector<float> &source_corr_confidences = {},
    const std::vector<float> &target_corr_confidences = {}, double lambda = 0.5,
    const Eigen::Matrix3d &R_state_to_target = Eigen::Matrix3d::Identity(),
    const Eigen::Vector3d &sensor_origin_in_state = Eigen::Vector3d::Zero(),
    const std::vector<Eigen::Matrix3d> &source_corr_covariances = {},
    const std::vector<Eigen::Matrix3d> &target_corr_covariances = {});

  /**
   * @brief 关键帧紧耦合 iESKF：投影关联 + 几何/光度虚拟观测，迭代更新名义状态。
   * @param pointcloud_cur 当前帧 body 系点云
   * @param target_kdtree 锚点关键帧目标
   * @param target_color_gradient,target_normals 目标属性
   * @param imu_states 输入输出：IMU 先验 / 后验（就地更新均值；结束时写 Joseph cov）
   * @param R_anchor,t_anchor 锚点关键帧世界位姿
   * @param lambda 几何权重
   * @param flow_gate_cov 非空时用本步最终的 K、H、R 对这份并行协方差做 Joseph 更新；不改主滤波的 P
   * @param T_imu_cam 外参
   * @param target_depth_cam,cam 非空且有效时走投影关联，否则欧氏
   * @param source_voxel_size 源点下采样叶长 (m)
   * @param esikf_max_correspondence_distance 欧氏回退最大距离 (m)
   */
  void ESIKF(const pcl::PointCloud<PointTypeRGB>::Ptr &pointcloud_cur,
             const pcl::KdTreeFLANN<PointTypeRGB>::Ptr &target_kdtree,
             const std::vector<Eigen::Vector3d> &target_color_gradient,
             const std::vector<Eigen::Vector3d> &target_normals, ImuStatesGroup &imu_states,
             const Eigen::Matrix3d &R_anchor, const Eigen::Vector3d &t_anchor,
             const std::vector<float> &source_point_confidences = {},
             const std::vector<float> &target_point_confidences = {}, double lambda = 0.5,
             Eigen::Matrix<double, DIM_STATE, DIM_STATE> *flow_gate_cov = nullptr,
             const Eigen::Matrix4d &T_imu_cam = Eigen::Matrix4d::Identity(),
             ESIKFDebugSummary *debug_out = nullptr, double source_voxel_size = 0.05,
             const std::vector<Eigen::Matrix3d> &source_point_covariances = {},
             const std::vector<Eigen::Matrix3d> &target_point_covariances = {},
             double esikf_max_correspondence_distance = 0.15,
             const cv::Mat &target_depth_cam = cv::Mat(),
             const CameraIntrinsics &cam = CameraIntrinsics{}
               SIO_PROFILER_PARAMS);

  /**
   * @brief 非关键帧松耦合 ESKF：一次位姿观测更新。
   * @param imu_propagated IMU 传播先验（含 P_pred）
   * @param T_gicp 世界系绝对位姿观测
   * @param R_gicp 已对齐到混合误差 [δθ_body, δp_world] 的 6×6 观测噪声
   * @param flow_gate_cov 非空时用同一次 K、H、R 更新这份并行协方差，不改返回的绝对协方差
   * @param debug_out 可选输出本次实际创新、正则化 S、K、修正量和 NIS，不改变更新计算
   * @return 后验 ImuStatesGroup（Joseph 协方差）
   */
  ImuStatesGroup ESKF(const ImuStatesGroup &imu_propagated, const Eigen::Matrix4f &T_gicp,
                      const Eigen::Matrix<double, 6, 6> &R_gicp,
                      Eigen::Matrix<double, DIM_STATE, DIM_STATE> *flow_gate_cov = nullptr,
                      ESKFDebugSummary *debug_out = nullptr);
};

/** @brief 点云法向/颜色梯度/立体点协方差与欧氏对应搜索。 */
class FeatureProcessor {
public:
  /** @brief 设置立体内参（协方差建模用）。 */
  void SetCameraIntrinsics(const CameraIntrinsics &cam) {
    cam_ = cam;
  }

  /**
   * @brief 设置外参：点_cam → 点_body = R·p + t。
   * @param R_body_cam,t_body_cam 相机→机体
   */
  void SetBodyFromCameraExtrinsic(const Eigen::Matrix3d &R_body_cam,
                                  const Eigen::Vector3d &t_body_cam) {
    R_body_cam_ = R_body_cam;
    t_body_cam_ = t_body_cam;
  }

  /**
   * @brief 视差方差基数 σ_d² 基准（再按置信度缩放）。
   * @param disp_var_base 须 > 0
   */
  void SetDisparityVarBase(double disp_var_base) {
    disp_var_base_ = std::max(1e-6, disp_var_base);
  }

  /**
   * @brief 估计法向并输出带法向点云。
   * @param cloud 输入 RGB 点云
   * @param cloud_out 输出含法向点云
   */
  void EstimateNormals(const pcl::PointCloud<PointTypeRGB>::Ptr &cloud,
                       pcl::PointCloud<PointTypeNormal>::Ptr &cloud_out);

  /**
   * @brief 颜色梯度（简化）：内部自估法向后算切平面梯度。
   * @param target 目标点云
   * @param target_color_gradient 与点一一对齐的 ∇I
   */
  void PrepareColorGradient(const pcl::PointCloud<PointTypeRGB>::Ptr &target,
                            std::vector<Eigen::Vector3d> &target_color_gradient);

  /**
   * @brief 颜色梯度（完整）：Open3D Colored ICP 风格，供 ColorICP 复用。
   * @param target 已含法向的点云
   */
  void PrepareColorGradient(const pcl::PointCloud<PointTypeNormal>::Ptr &target,
                            std::vector<Eigen::Vector3d> &target_color_gradient);

  /**
   * @brief 估计几何协方差与颜色方差（邻域统计）。
   * @param geo_covariances 每点 3×3
   * @param color_variances 每点标量光度方差
   * @param point_confidences 可空
   */
  void EstimatePointCovariances(const pcl::PointCloud<PointTypeRGB>::Ptr &cloud,
                                const std::vector<Eigen::Vector3d> &target_color_gradient,
                                std::vector<Eigen::Matrix3d> &geo_covariances,
                                std::vector<double> &color_variances,
                                const std::vector<float> &point_confidences = {}) const;

  /**
   * @brief KD-tree 最近邻对应；超距记为 -1。
   * @param match_indices 与 source 对齐，值为 target 下标或 -1
   * @param max_dist_sq 最大距离平方 (m²)
   * @param funnel_out 可选漏斗统计
   */
  void FindCorrespondences(const pcl::PointCloud<PointTypeRGB>::Ptr &source,
                           const pcl::KdTreeFLANN<PointTypeRGB>::Ptr &target_kdtree,
                           std::vector<int> &match_indices, float max_dist_sq = 0.25f,
                           CorrespondenceFunnelStats *funnel_out = nullptr);

  /**
   * @brief 相机系立体点协方差转到机体系：Σ_b = R Σ_c Rᵀ。
   * @param cloud_cam 相机系点云
   * @param confidences 与点对齐的置信度
   * @param R_body_cam 旋转外参
   * @param out_covariances 输出机体系 Σ
   */
  void ComputePointCovariancesBody(const pcl::PointCloud<PointTypeRGB>::Ptr &cloud_cam,
                                   const std::vector<float> &confidences,
                                   const Eigen::Matrix3d &R_body_cam,
                                   std::vector<Eigen::Matrix3d> &out_covariances) const;

private:
  /**
   * @brief 单点相机系立体协方差（像素/视差噪声经 J_uvd）。
   * @param pc 相机系点
   * @param z 深度 (m)
   * @param confidence 置信度，缩放视差方差
   * @param out_trace_mean 输出 trace(Σ)/3 供调试
   */
  Eigen::Matrix3d ComputeCameraPointCovariance(const Eigen::Vector3d &pc, double z,
                                               double confidence, double &out_trace_mean) const;

  CameraIntrinsics cam_{500.0f, 500.0f, 320.0f, 240.0f, 0.12f};
  Eigen::Matrix3d R_body_cam_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_body_cam_ = Eigen::Vector3d::Zero();
  double disp_var_base_ = 0.25;
};

// ColorICP 实现见 color_icp.h；图像预处理见 image_preprocessor.h

/** @brief 前端/后端关键帧判定、创建与跨帧 IMU 缓冲。 */
class KeyframeManager {
public:
  struct Config {
    double front_trans_thresh_m = 0.4;   ///< 前端平移阈值 (m)
    double front_angle_thresh_deg = 10.0; ///< 前端转角阈值 (deg)
    double front_time_thresh_s = 0.6;    ///< 前端时间阈值 (s)

    double speed_ref_mps = 1.5;          ///< 速度自适应参考 (m/s)
    double adaptive_scale_min = 0.5;     ///< 平移阈值缩放下限
    double adaptive_scale_max = 2.0;     ///< 平移阈值缩放上限

    size_t backend_min_frontend_kfs = 3; ///< 晋升后端最少前端 KF 数
    size_t backend_queue_legacy_trigger = 5;
    double backend_trans_thresh_m = 1.8;
    double backend_angle_thresh_deg = 35.0;
    double backend_time_thresh_s = 4.0;

    std::string params_file_path; ///< 可选 YAML 配置路径
  };

  KeyframeManager();
  /** @param cfg 关键帧阈值配置 */
  explicit KeyframeManager(const Config &cfg);

  /**
   * @brief 创建系统首个关键帧（世界系初值来自 imu_states）。
   * @param frame_data 原始帧
   * @param pcd_cur 机体系点云
   * @param disp_filtered 滤波视差（缓存供后续投影）
   * @param point_confidences,point_covariances,confidence_map 可选属性
   */
  KeyFrameData CreateFirstKeyFrame(const std::shared_ptr<FrameData> &frame_data,
                                   const pcl::PointCloud<PointTypeRGB>::Ptr &pcd_cur,
                                   const ImuStatesGroup &imu_states, const cv::Mat &disp_filtered,
                                   const std::vector<float> &point_confidences = {},
                                   const std::vector<Eigen::Matrix3d> &point_covariances = {},
                                   const cv::Mat &confidence_map = cv::Mat());

  /**
   * @brief 创建前端关键帧，并附带相对上一前端 KF 的 IMU 预积分边。
   * @param prev_frame 上一前端关键帧
   * @param curr_frame 当前原始帧
   * @param imu_buffer [t_prev, t_curr] IMU
   * @param pcd_cur 当前机体系点云
   */
  std::shared_ptr<KeyFrameData> CreateFrontendKeyFrame(
    const KeyFrameData &prev_frame, const std::shared_ptr<FrameData> &curr_frame,
    const std::deque<sensor_msgs::msg::Imu::SharedPtr> &imu_buffer,
    pcl::PointCloud<PointTypeRGB>::Ptr pcd_cur, const ImuStatesGroup &imu_states,
    const cv::Mat &disp_filtered, const std::vector<float> &point_confidences = {},
    const std::vector<Eigen::Matrix3d> &point_covariances = {},
    const cv::Mat &confidence_map = cv::Mat());

  /**
   * @brief 由最新前端 KF 晋升后端关键帧，附带相对上一后端 KF 的预积分。
   * @param prev_backend_keyframe 上一后端关键帧
   * @param latest_frontend_kf 待晋升的前端 KF
   * @param imu_buffer 跨后端 KF 的 IMU
   */
  std::shared_ptr<KeyFrameData> CreateBackendKeyFrame(
    const KeyFrameData &prev_backend_keyframe, std::shared_ptr<KeyFrameData> latest_frontend_kf,
    const std::deque<sensor_msgs::msg::Imu::SharedPtr> &imu_buffer);

  /**
   * @brief 评估是否应建前端关键帧（平移/转角/时间，含速度自适应）。
   * @param imu_progated 当前传播状态
   * @param last_fused 上一融合关键帧状态
   * @param time_diff 距上一前端 KF 的时间 (s)
   */
  FrontKeyframeDecision EvaluateFrontKeyframeDecision(const ImuStatesGroup &imu_progated,
                                                      const ImuStatesGroup &last_fused,
                                                      double time_diff) const;

  /** @brief 是否前端关键帧（EvaluateFrontKeyframeDecision 的布尔封装）。 */
  bool IsFrontKeyFrame(const ImuStatesGroup &imu_progated, const ImuStatesGroup &last_fused,
                       double time_diff) const;

  /**
   * @brief 是否晋升后端关键帧（运动阈值 + 队列深度）。
   * @param time_diff_from_last_backend 距上一后端 KF 时间 (s)
   */
  bool IsBackKeyFrame(const KeyFrameData &latest_frontend_kf, size_t frontend_deque_size,
                      double time_diff_from_last_backend) const;

  /** @brief 仅按前端队列深度的旧版后端触发。 */
  bool IsBackKeyFrame(size_t frontend_deque_size) const;

  /** @brief 上一前端关键帧副本。 */
  KeyFrameData getPrevFrame() const {
    return prev_frame_;
  }
  /** @brief 上一后端关键帧副本。 */
  KeyFrameData getPrevBackendKeyFrame() const {
    return prev_backend_keyframe_;
  }

  /** @brief 向前端 IMU 缓冲追加一条消息。 */
  void AddImuData(const sensor_msgs::msg::Imu::SharedPtr &imu_msg);
  /** @brief 清空前端 IMU 缓冲。 */
  void ClearImuBuffer();
  /** @brief 取出当前缓冲快照（供建 KF 用）。 */
  std::deque<sensor_msgs::msg::Imu::SharedPtr> GetImuBufferForKeyFrame();
  /**
   * @brief 抽取时间窗 [t0, t1] 内的 IMU（含边界）。
   * @param t0,t1 秒级时间戳
   */
  std::deque<sensor_msgs::msg::Imu::SharedPtr> ExtractImuBetween(double t0, double t1);

private:
  /** @brief 从 YAML 加载 Config；成功返回 true。 */
  bool LoadConfig(const std::string &path);
  /** @brief 两旋转矩阵夹角 (rad)。 */
  static double RotationAngleBetween(const Eigen::Matrix3d &R_cur, const Eigen::Matrix3d &R_last);

  Config cfg_;

  KeyFrameData prev_frame_;
  KeyFrameData prev_backend_keyframe_;

  std::deque<sensor_msgs::msg::Imu::SharedPtr> front_imu_buffer_;
  std::mutex mtx_imu_buffer_;
  int frontend_keyframe_id_ = 0;
  int backend_keyframe_id_ = 0;
};
